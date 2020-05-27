#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>
#include <string.h>

// - __m256: 256-bit vector containing 8 floats

struct args {
    __m256 x;
    __m256 y;
    float * o;
};

__m256 get_chunk(float * v, int n){
    __m256 c = _mm256_setzero_ps();
    memcpy(&c, v, (sizeof float) * n);
    return c;
}

void func(void * aux) {
    struct args * args = (struct args *) aux;
    // compute vector multiplication
    __m256 o = _mm256_mul_ps(args->x, args->y);
    float * r = (float *) &o;
    // accumulate result in output address
    *(args->o) += r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
    free(args);
}

void ki_apply(float *K, float *I, float *R, int in_size, int out_size) {
    assert((K != NULL) && (I != NULL) && (R != NULL));

    // K: (in_size * out_size), row major ordered
    // I: (in_size)
    // R: (out_size)

    // number of chunks
    int n_c = ceil(in_size / 8);
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

        // compute dot product between kernel and input
        for (int j=0; j<n_c; j++){
            // allocate an argument holder (will be freed before a thread exits)
            args = malloc(sizeof (struct args));

            // convert subarrays into 256-bit chunks
            n_f = min(in_size - 8 * j, 8);
            args->x = get_chunk(K_o + 8 * j, n_f);
            args->y = get_chunk(I + 8 * j, n_f);
            args->o = R_o;

            // run thread
            pthread_create(tid + (i * n_c + j), NULL, &func, (void *)(args));
        }
    }

    for (int i=0; i<out_size; i++){
        for (int j=0; j<n_c; j++){
            pthread_join(tid[i], NULL);
            printf("thread %d ends\n", i);
        }
    }
}
