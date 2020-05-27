#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#define MIN(x, y) ((x)>(y)? (y) : (x))
#define DEBUG

// - __m256: 256-bit vector containing 8 floats

struct args {
    __m256 x;
    __m256 y;
    float * o;
};

__m256 get_chunk(float * v, int n){

#ifdef DEBUG
    printf("get_chunk: v [");
    for (int i=0; i<n; i++) printf("%3.2f ", v[i]);
    printf("]\n");
#endif

    __m256 c = _mm256_setzero_ps();
    memcpy((void *)&c, (void *)v, (size_t)((sizeof (float)) * n));

#ifdef DEBUG
    printf("get_chunk: c [");
    for (int i=0; i<8; i++) printf("%3.2f ", *(float *)&c[i]);
    printf("]\n");
#endif

    return c;
}

void * func(void * aux) {
#ifdef DEBUG
    printf("func: apply operation, aux @ %p\n", aux);
#endif
    struct args * args = (struct args *) aux;
    // compute vector multiplication
    __m256 o = _mm256_mul_ps(args->x, args->y);
    float * r = (float *) &o;
    // accumulate result in output address
    float acc = r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
    *(args->o) += acc;
    free(args);
#ifdef DEBUG
    printf("func: acc += %f\n", acc);
#endif
    return NULL;
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
    struct args * args;
    // holder for addresses
    void * K_o, * R_o;

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
            args = malloc(sizeof (struct args));

            // convert subarrays into 256-bit chunks
            n_f = MIN(in_size - 8 * j, 8);
#ifdef DEBUG
            printf("\nki_apply: chunk idx [%d]/[%d], # elements %d, args @ %p\n", j, n_c-1, n_f, args);
#endif
            args->x = get_chunk(K_o + 8 * j, n_f);
            args->y = get_chunk(I + 8 * j, n_f);
            args->o = R_o;
#ifdef DEBUG
            printf("ki_apply: create thread %d\n", i * n_c + j);
#endif
            // run thread
            pthread_create(tid + (i * n_c + j), NULL, func, (void *)(args));
        }
    }

    for (int i=0; i<out_size; i++){
        for (int j=0; j<n_c; j++){
            pthread_join(tid[i * n_c + j], NULL);
            printf("thread %d ends\n", i * n_c + j);
        }
    }
}
