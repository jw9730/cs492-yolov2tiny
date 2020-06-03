#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512

#define INDEX_ROW_MAJOR_3(i, j, k, I, J, K) (k + K * (j + J * (i)))
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) (l + L * (k + K * (j + J * (i))))

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void conv_ws(float *I, float *K, float *R, int iw, int ih, int ow, int oh, int kw, int kh, int sw, int sh, int ic, int oc){
    int BLOCKS_PER_PIXEL = ceil(float(oc)/float(THREADS_PER_BLOCK));
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cid = blockIdx.x % BLOCKS_PER_PIXEL;
    int pid = blockIdx.x / BLOCKS_PER_PIXEL;
    // compute block index in output channel dimension
    int ofs = cid * THREADS_PER_BLOCK;
    int n_tid = (oc - ofs < THREADS_PER_BLOCK)? (oc - ofs) : THREADS_PER_BLOCK;
    // compute output pixel of the block
    int h = pid % oh;
    int w = pid / oh;
    if (tid >= n_tid) return;
    // declare on-chip shared memory
    extern __shared__ float M[];
    // read input data once per block (shared across threads)
    if(threadIdx.x == 0){
        for (int i=0; i<kw; i++){
            for (int j=0; j<kh; j++){
                for (int k=0; k<ic; k++){
                    M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] = I[INDEX_ROW_MAJOR_3(w*sw+i,h*sh+j,k, iw,ih,ic)];
                }
            }
        }
    }
    // wait until data is ready
    __syncthreads();
    // apply convolution
    float *o = R + INDEX_ROW_MAJOR_3(w,h,ofs+tid, ow,oh,oc);
    for (int i=0; i<kw; i++){
        for (int j=0; j<kh; j++){
            for (int k=0; k<ic; k++){
                atomicAdd(o, M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] * K[INDEX_ROW_MAJOR_4(i,j,k,ofs+tid, kw,kh,ic,oc)]);
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
    int BLOCK_MEMSIZE = kw * kh * ic * sizeof(float);
    conv_ws<<<BLOCKS,THREADS_PER_BLOCK,BLOCK_MEMSIZE>>>(dev_I, dev_K, dev_R, iw, ih, ow, oh, kw, kh, sw, sh, ic, oc);
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, ow * oh * oc * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_K); cudaFree(dev_R);
}