#include <stdio.h>
#include <cblas.h>

void dot_product(float *v1, float *v2, float *res, int n) {
    *res = cblas_sdot(n, v1, 1, v2, 1);
}
