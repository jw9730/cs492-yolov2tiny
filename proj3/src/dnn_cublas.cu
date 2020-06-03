#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C"
void ki_apply(float *K, float *I, float *R, int in_size, int out_size) {
    // K: (in_size, out_size)
    // I: (1, in_size)
    // R: (1, out_size)

    cudaError_t cudaStat;     // cudaMalloc status
    cublasStatus_t stat;      // CUBLAS functions status
    cublasHandle_t handle;    // CUBLAS context

    // on the device
    float * d_I; 
    float * d_K;
    float * d_R;

    cudaStat = cudaMalloc((void**)&d_I, 1 * in_size * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_K, in_size * out_size * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_R, 1 * out_size * sizeof(float));
    stat = cublasCreate(&handle);

    float a = 1.0f;
    float b = 1.0f;

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       1, out_size, in_size,
                       &a, d_I, 1, d_K, in_size,
                       &b, d_R, 1);
    stat = cublasGetMatrix(1, out_size, sizeof(float), d_R, 1, R, 1); // cp d_c - > c

    // free device memory
    cudaFree(d_I);
    cudaFree(d_K);
    cudaFree(d_R);
    // destroy CUBLAS context
    cublasDestroy(handle);
}