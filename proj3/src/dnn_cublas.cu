#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

extern "C"
void ki_apply(float *K, float *I, float *res, int in_size, int out_size) {
    // K: (in_size, out_size)
    // I: (1, in_size)
    // res: (1, out_size)

    cudaError_t cudaStat;     // cudaMalloc status
    cublasStatus_t stat;      // CUBLAS functions status
    cublasHandle_t handle;    // CUBLAS context

    // on the device
    float * d_I; 
    float * d_K;
    float * d_res;

    cudaStat = cudaMalloc(( void**)& d_I , 1*in_size*sizeof(*I));
    assert (cudaStat == 0);
    cudaStat = cudaMalloc(( void**)& d_K, in_size*out_size*sizeof(*K));
    assert (cudaStat == 0);
    cudaStat = cudaMalloc(( void**)& d_res , 1*out_size*sizeof(*res));
    assert (cudaStat == 0);
    stat = cublasCreate(&handle);
    assert (stat == 0);

    float al =1.0f;
    float bet =0.0f;

    stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,out_size,in_size,&al,d_I,1,d_K,in_size,&bet,d_res,1);
    assert (stat == 0);
    stat = cublasGetMatrix(1 ,out_size , sizeof(*res) , d_res ,1 ,res , 1 ); // cp d_c - > c
    assert (stat == 0);

    cudaFree(d_I);
    // free device memory
    cudaFree(d_K);
    // free device memory
    cudaFree(d_res);
    // free device memory
    cublasDestroy( handle );
    // destroy CUBLAS context
}