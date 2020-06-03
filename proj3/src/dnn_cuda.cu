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
    int oc_bid = bid % (ow * oh);
    int c_ofs = oc_bid * THREADS_PER_BLOCK;
    int n_tid = (oc - c_ofs < THREADS_PER_BLOCK)? oc - c_ofs : THREADS_PER_BLOCK;
    if (tid >= n_tid) return;

    // compute output pixel of the block
    int pos_bid = bid - oc_bid;
    int w = pos_bid % ow;
    int h = pos_bid - oh * w;
    
    // declare on-chip shared memory
    __shared__ float input[kw * kh * ic];
    // read input data once per block (shared across threads)
    if(tid == 0){
        for (int i=0; i<kw; i++){
            for (int j=0; j<kh; j++){
                for (int k=0; k<ic; k++){
                    input[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] = I[INDEX_ROW_MAJOR_3(w*sw+i,h*sh+j,k, iw,ih,ic)];
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
                int input_idx = INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic);
                int kernel_idx = INDEX_ROW_MAJOR_4(i,j,k,c_ofs+tid, kw,kh,ic,oc);
                int output_idx = INDEX_ROW_MAJOR_3(w,h,c_ofs+tid, ow,oh,oc);
                atomicAdd(R +output_idx, input[input_idx] * K[kernel_idx]);
            }
        }
    }
}

extern "C"
void conv2d(float * I, float * K, float * R, int iw, int ih, int ow, int oh, int kw, int kh, int sw, int sh, int ic, int oc) {
    float *dev_I, *dev_K, *dev_R;
    assert ((iw == ow*sw) && (ih == iw*sh));
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
    int BLOCKS_OUT = ceil(float(oc)/float(THREADS_PER_BLOCK));
    int BLOCKS = ow * oh * BLOCKS_OUT;
    conv<<<BLOCKS,THREADS_PER_BLOCK>>>(dev_I, dev_K, dev_R, iw, ih, ow, oh, kw, kh, sw, sh, ic, oc);
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, ow * oh * oc * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_K); cudaFree(dev_R);
}