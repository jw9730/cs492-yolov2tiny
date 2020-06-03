#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK 512

void mul(float *i, float *k, float *r, int n_tid){
    int tid = 0;
    while (tid < n_tid){
        r[tid] += i[tid] * k[tid];
        tid += 1;
    }
}

extern "C"
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
    // within a block, 512 threads can execute in parallel (via shared memory)

    // trial 1:
    // loop over outer dimensions, and compute a dot product in chunks of size 512
    // shared memory: gets vectors to compute product
    // kernel function: multiply-and-accumulate of floats, accumulation can be asynchronous

    // copy inputs to device
    
    // launch kernel on GPU
    for(int i=0; i<n_pixels; i++){
        for(int j=0; j<kernel_out; j++){
            // vectors to compute dot product
            float * I_ = I + i * kernel_in;
            float * K_ = K + j * kernel_in;
            // target output address
            float * R_ = R + i * kernel_out + j;

            // compute dot product and accumulate the result in target output
            // block-wise
            int residue = kernel_in;
            int ofs = 0;
            while (residue > 0){
                printf("ofs: %d\n", ofs);
                int n_tid = residue > THREADS_PER_BLOCK? THREADS_PER_BLOCK : residue;
                mul(I_ + ofs, K_ + ofs, R_, n_tid);
                ofs += THREADS_PER_BLOCK;
                residue -= THREADS_PER_BLOCK;
            }
        }
    }
}
