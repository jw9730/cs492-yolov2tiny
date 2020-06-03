#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK 512

__global__ void mul(float *i, float *k, float *r){
    //int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    //r[idx] += i[idx] + k[idx];
    *r += (*i) * (*k);
}

void matmul(float * I, float * K, float * R, int n_pixels, int kernel_in, int kernel_out) {
    // I: (n_pixels * kernel_in), row major ordered
    // K: (kernel_in * kernel_out), column major ordered
    // R: (n_pixels * kernel_out), row major ordered
    assert((I != NULL) && (K != NULL) && (R != NULL));

    // todo:
    // compute matrix multiplication between I and K and store the results in R

    // approach:
    // compute dot products in GPU
    // how to effectively eliminate loops?

    // assign blocks
    // within a block, 512 computations can execute in parallel (via shared memory)

    // trial 1:
    // loop over outer dimensions, and compute a dot product in chunks of size 512
    // shared memory: gets vectors to compute product
    // kernel function: multiply-and-accumulate of floats, accumulation can be asynchronous

    // device copies of inputs
    float * d_I, * d_K, * d_R;

    // allocate space for device copies of input
    cudaMalloc((void **) &d_I, n_pixels * kernel_in * sizeof(float));
    cudaMalloc((void **) &d_K, kernel_in * kernel_out * sizeof(float));
    cudaMalloc((void **) &d_R, n_pixels * kernel_out * sizeof(float));

    // copy inputs to device
    cudaMemcpy(d_I, &I, n_pixels * kernel_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, &K, kernel_in * kernel_out * sizeof(float), cudaMemcpyHostToDevice);
    
    // launch kernel on GPU
    for(int i=0; i<n_pixels; i++){
        for(int j=0; j<kernel_out; j++){
            // vectors to compute dot product
            float * I_ = d_I + i * kernel_in;
            float * K_ = d_K + j * kernel_in;
            // target output address
            float * R_ = d_R + i * kernel_out + j;

            // compute dot product and accumulate the result in target output
            for(int k=0; k<kernel_in; k++){
                mul<<<1,1>>>(I_, K_, R_);
            }
        }
    }
    // synchronize
    cudaDeviceSynchronize();
    // copy result back to host
    cudaMemcpy(&R, d_R, n_pixels * kernel_out * sizeof(float), cudaMemcpyDeviceToHost);
    // free GPU memory
    cudaFree(d_I);
    cudaFree(d_K);
    cudaFree(d_R);
}
