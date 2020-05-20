/* gcc compile option */
/* gcc -shared -fPIC -o dnn_openblas.so dnn_openblas.c */

#include <stdio.h>
#include <cblas.h> // use sgemm function in cblas library

void run_for_oc_v2(const float ****ptin, const float ****weights, float ****shared_result,
                int chunk, int k, int parallelism,
                int IC,
                int OW, int OH,
                int SW, int SH,
                int KW, int KH) {
    
    int oc = chunk * parallelism + k;
    int i, j;

    printf("run_for_oc_v2: loop start\n");

    for (int ic=0; ic<IC; ic++) {
        for (int ow=0; ow<OW; ow++) {
            for (int oh=0; oh<OH; oh++) {
                for (int ii=0; ii<KW; ii++) {
                    for (int jj=0; jj<KH; jj++){
                        i = SW * ow + jj;
                        j = SH * oh + ii;
                        shared_result[0][ow][oh][oc] += ptin[0][i][j][ic] * weights[ii][jj][ic][oc];
                    }
                }
            }
        }
    }

    printf("run_for_oc_v2: loop end\n");
}

void run_sgemm(const float **A, const float **B, float **C, int m, int k, int n) {
    // A: (m, k)
    // B: (k, n)
    // C: (m, n)
    // C <- A * B + C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 1, C, n);
}