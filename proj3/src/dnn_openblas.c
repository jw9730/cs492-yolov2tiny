#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <assert.h>

void dot_product(float *v1, float *v2, float *res, int n) {
    // n: length of v1 and v2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, n,
                1.0, v1, n, v2, 1,
                1.0, res, 1);
}

void ki_apply(float *K, float *I, float *res, int in_size, int out_size) {
    assert(!K && !I && !res);

    // K: (in_size, out_size)
    // I: (1, in_size)
    // res: (1, out_size)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, out_size, in_size,
                1.0, I, in_size, K, out_size,
                1.0, res, out_size);
}
