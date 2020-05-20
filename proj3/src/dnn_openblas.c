#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

void dot_product(float *v1, float *v2, float *res, int n) {
    //printf("dot_product: (v1, v2, res, n) = (%p, %p, %p, %d)\n", v1, v2, res, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, n,
                1.0, v1, n, v2, 1,
                1.0, res, 1);
}

void ki_apply(float *K, float *I, float *res, int in_size, int out_size) {
    // K: (KW * KH * IC, OC)
    // I: (1, KW * KH * IC)
    // res: (1, OC)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, out_size, in_size,
                1.0, I, in_size, K, out_size,
                1.0, res, out_size);
}
