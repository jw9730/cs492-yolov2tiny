#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>
#include <string.h>
//#define DEBUG

// - __m256: 256-bit vector containing 8 floats

struct args {
    float * x;
    float * y;
    int n_f;
    float * o;
};

void * func(void * aux) {
    struct args * args = (struct args *) aux;
    int n_f = args->n_f;

    __m256 x = _mm256_setzero_ps();
    __m256 y = _mm256_setzero_ps();
    memcpy(&x, args->x, sizeof (float) * n_f);
    memcpy(&y, args->y, sizeof (float) * n_f);
    __m256 o = _mm256_mul_ps(x, y);
    
    float * r = (float *) &o;
    float acc = r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
    *(args->o) += acc;
    
    free(args);
}

void ki_apply(float *K, float *I, float *R, int in_size, int out_size) {
    assert((K != NULL) && (I != NULL) && (R != NULL));
    // K: (in_size * out_size), row major ordered
    // I: (in_size)
    // R: (out_size)
#ifdef DEBUG
    printf("ki_apply: got K %p, I %p, R %p, in_size %d, out_size %d\n", K, I, R, in_size, out_size);
#endif
    // number of chunks
    int n_c = ceil((float)in_size / 8.0);
    // holder for num_elements within a chunk (<= 8)
    int n_f;
    // holder for args struct
    struct args * args = NULL;
    // holder for addresses
    void * K_o = NULL;
    void * R_o = NULL;

    pthread_t tid[out_size * n_c];
    for (int i=0; i<out_size; i++){
        // kernel vector
        K_o = K + i * in_size;
        // output address
        R_o = R + i;
#ifdef DEBUG
        printf("\nki_apply: output idx [%d]/[%d]. Kernel vector M[%p...], out channel M[%p]\n", i, out_size-1, K_o, R_o);
#endif
        // compute dot product between kernel and input
        for (int j=0; j<n_c; j++){
            // allocate an argument holder (will be freed before a thread exits)
            // convert subarrays into 256-bit chunks
            args = malloc(sizeof (struct args));
            memset(args, 0, sizeof (struct args));
            n_f = in_size - 8 * j;
            args->n_f = (n_f > 8) ? 8 : n_f;
            args->x = K_o + 8 * j;
            args->y = I + 8 * j;
            args->o = R_o;

            // run thread
            pthread_create(tid + (i * n_c + j), NULL, func, (void *)(args));
        }
        for (int j=0; j<n_c; j++){
            pthread_join(tid[i * n_c + j], NULL);
        }
    }
}


#ifdef DEBUG
    printf("ki_apply: output [");
    for (int i=0; i<out_size; i++) printf("%3.2f ", R[i]);
    printf("]\n");
#endif

    return;
}
