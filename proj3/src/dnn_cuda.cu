#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void mm(float *I, float *K, float *R, int n_pixels, int kernel_in, int kernel_out){
    for(int i=0; i<n_pixels; i++){
        for(int j=0; j<kernel_out; j++){
            // vectors to compute dot product
            float * Iix = I + i * kernel_in + blockIdx.x;
            float * Kxj = K + j * kernel_in + blockIdx.x;
            // target output address
            float * Rij = R + i * kernel_out + j;
            // accumulate
            float tmp = (*Iix) * (*Kxj);
            if (i==0 && j==0){
                printf("[%d, %d]\tblock %d:\t%f <- %f + %f @ %p\n", i, j, blockIdx.x, *Rij + tmp, *Rij, (*Iix) * (*Kxj), Rij);
            }
            *Rij += tmp;
        }
    }
}

extern "C"
void matmul(float * I, float * K, float * R, int n_pixels, int kernel_in, int kernel_out) {
    float *dev_I, *dev_K, *dev_R;
    // I: (n_pixels * kernel_in), row major ordered
    // K: (kernel_in * kernel_out), column major ordered
    // R: (n_pixels * kernel_out), row major ordered
    // todo: compute matrix multiplication between I and K and store the results in R

    // how to effectively eliminate loops?
    // assign blocks
    // within a block, 512 threads can execute in parallel (via shared memory)

    // trial 1:
    // loop over outer dimensions, and compute dot product in chunks of size 512
    // shared memory: gets vectors to compute product, each element consumed by threads
    // kernel function: multiply-and-accumulate of floats, accumulation can be asynchronous

    // copy inputs to device
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, n_pixels * kernel_in * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_K, kernel_in * kernel_out * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, n_pixels * kernel_out * sizeof(float) ) );

    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, n_pixels * kernel_in * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_K, K, kernel_in * kernel_out * sizeof(float), cudaMemcpyHostToDevice ) );
    
    // launch kernel on GPU
    mm<<<kernel_in,1>>>(dev_I, dev_K, dev_R, n_pixels, kernel_in, kernel_out);
    
    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, n_pixels * kernel_out * sizeof(float), cudaMemcpyDeviceToHost ) );

    // cleanup
    cudaFree(dev_I); cudaFree(dev_K); cudaFree(dev_R);
}
