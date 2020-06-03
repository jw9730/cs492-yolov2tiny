#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512

#define INDEX_ROW_MAJOR_3(i, j, k, I, J, K) (i * (J*K) + j * K + k)
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) (i * (J*K*L) + j * (K*L) + k * L + l)

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void mm_tp(float *I, float *K, float *R, int n_pixels, int kernel_in, int kernel_out){
    int gidx = blockIdx.x;
    int lidx = threadIdx.x;
    if(gidx * THREADS_PER_BLOCK + lidx >= kernel_in){
        return;
    }
    for(int i=0; i<n_pixels; i++){
        for(int j=0; j<kernel_out; j++){
            // vectors to compute dot product
            float * Iix = I + i * kernel_in + gidx * THREADS_PER_BLOCK + lidx;
            float * Kxj = K + j * kernel_in + gidx * THREADS_PER_BLOCK + lidx;
            // target output address
            float * Rij = R + i * kernel_out + j;
            // accumulate
            atomicAdd(Rij, (*Iix) * (*Kxj));
        }
    }
}
extern "C"
void matmul_tp(float * I, float * K, float * R, int n_pixels, int kernel_in, int kernel_out) {
    float *dev_I, *dev_K, *dev_R;
    // I: (n_pixels * kernel_in), row major ordered
    // K: (kernel_in * kernel_out), column major ordered
    // R: (n_pixels * kernel_out), row major ordered
    // todo: compute matrix multiplication between I and K and store the results in R

    // loop over outer dimensions, and compute dot product in chunks of size 512
    // kernel function: multiply-and-accumulate of floats, accumulation can be asynchronous
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, n_pixels * kernel_in * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_K, kernel_in * kernel_out * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, n_pixels * kernel_out * sizeof(float) ) );
    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, n_pixels * kernel_in * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_K, K, kernel_in * kernel_out * sizeof(float), cudaMemcpyHostToDevice ) );
    // launch kernel on GPU
    int BLOCKS = ceil(float(kernel_in)/float(THREADS_PER_BLOCK));
    mm_tp<<<BLOCKS,THREADS_PER_BLOCK>>>(dev_I, dev_K, dev_R, n_pixels, kernel_in, kernel_out);
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, n_pixels * kernel_out * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_K); cudaFree(dev_R);
    // problem: no on-chip data reuse (no sharing across threads)
    // solution: fallback to looped convolution, and enforce input and kernel reuse
}

__global__ void conv(float *I, float *K, float *R, int iw, int ih, int ow, int oh, int kw, int kh, int sw, int sh, int ic, int oc){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // compute block index in output channel dimension
    int BLOCKS_PER_PIXEL = ceil(float(oc)/float(THREADS_PER_BLOCK));
    int cid = bid % BLOCKS_PER_PIXEL;
    int offset = cid * THREADS_PER_BLOCK;
    int n_tid = (oc - offset < THREADS_PER_BLOCK)? (oc - offset) : THREADS_PER_BLOCK;
    // compute output pixel of the block
    int pid = bid - cid;
    int w = pid % oh;
    int h = pid - oh * w;
    printf("bid %d, tid %d, cid %d, offset %d, n_tid %d, pid %d, (w,h)=(%d,%d)\n", bid, tid, cid, offset, n_tid, pid, w, h);
    if (tid >= n_tid) return;

    
    // declare on-chip shared memory
    extern __shared__ float memory[];
    // read input data once per block (shared across threads)
    if(tid == 0){
        for (int i=0; i<kw; i++){
            for (int j=0; j<kh; j++){
                for (int k=0; k<ic; k++){
                    int mem_idx = INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic);
                    int input_idx = INDEX_ROW_MAJOR_3(w*sw+i,h*sh+j,k, iw,ih,ic);
                    memory[mem_idx] = I[input_idx];
                }
            }
        }
    }
    // wait until data is ready
    __syncthreads();
    // apply convolution
    for (int i=0; i<kw; i++){
        for (int j=0; j<kh; j++){
            for (int k=0; k<ic; k++){
                int mem_idx = INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic);
                int kernel_idx = INDEX_ROW_MAJOR_4(i,j,k,offset+tid, kw,kh,ic,oc);
                int output_idx = INDEX_ROW_MAJOR_3(w,h,offset+tid, ow,oh,oc);
                if (k == 0){
                    printf("[%d,%d] %1.5f <- %1.5f, acc %1.5f\n", bid, tid, R[output_idx], memory[mem_idx] * K[kernel_idx], R[output_idx] + memory[mem_idx] * K[kernel_idx]);
                }
                atomicAdd(&R[output_idx], memory[mem_idx] * K[kernel_idx]);
            }
        }
    }
}

extern "C"
void conv2d(float * I, float * K, float * R, int iw, int ih, int ow, int oh, int kw, int kh, int sw, int sh, int ic, int oc) {
    float *dev_I, *dev_K, *dev_R;
    // I: (iw * ih * ic), row major ordered
    // K: (kw * kh * ic * oc), row major ordered
    // R: (ow * oh * oc), row major ordered
    // todo: 2d convolution between I and K

    // loop over outer dimensions, and compute dot product in chunks of size 512
    // kernel function: convolution for a single sliding window
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, iw * ih * ic * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_K, kw * kh * ic * oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, ow * oh * oc * sizeof(float) ) );
    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, iw * ih * ic * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_K, K, kw * kh * ic * oc * sizeof(float), cudaMemcpyHostToDevice ) );
    // how to organiza blocks?
    // maximizing data reuse, spatial locality
    // thread over output channels (input stationary)
    int BLOCKS_PER_PIXEL = ceil(float(oc)/float(THREADS_PER_BLOCK));
    int BLOCKS = ow * oh * BLOCKS_PER_PIXEL;
    int shared_memory_size = kw * kh * ic * sizeof(float);
    printf("# blocks: %d, % blocks per pixel: %d\n", BLOCKS, BLOCKS_PER_PIXEL);
    conv<<<BLOCKS,THREADS_PER_BLOCK, shared_memory_size>>>(dev_I, dev_K, dev_R, iw, ih, ow, oh, kw, kh, sw, sh, ic, oc);
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, ow * oh * oc * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_K); cudaFree(dev_R);
}