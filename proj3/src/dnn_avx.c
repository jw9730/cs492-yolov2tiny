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
    struct args * p = (struct args *) aux;
    int n_f = p->n_f;

    __m256 x = _mm256_setzero_ps();
    __m256 y = _mm256_setzero_ps();
    memcpy(&x, p->x, sizeof (float) * n_f);
    memcpy(&y, p->y, sizeof (float) * n_f);
    __m256 o = _mm256_mul_ps(x, y);
    
    float * r = (float *) &o;
    float acc = 0.0;
    for (int i=0; i<8; i++){
        acc += r[i];
    }
    *(p->o) += acc;
}

void ki_apply(float *K, float *I, float *R, int in_size, int out_size) {
    // K: (in_size * out_size), row major ordered
    // I: (in_size)
    // R: (out_size)
    assert((K != NULL) && (I != NULL) && (R != NULL));
    
    // K_o, R_o: holder for addresses
    // args: holder for args struct
    // n_c: number of chunks
    // n_f: holder for num_elements within a chunk (<= 8)
    void * K_o = NULL;
    void * R_o = NULL;
    struct args args;
    int n_c = ceil((float)in_size / 8.0);
    int n_f;

    int i, j;
    struct args * args_list = malloc((sizeof (struct args)) * n_c);
    pthread_t tid[n_c];

    for (i=0; i<out_size; i++){
        // K_o: kernel vector
        // R_o: output address
        K_o = K + i * in_size;
        R_o = R + i;
        
        // compute dot product between kernel and input
        for (j=0; j<n_c; j++){
            // allocate an argument holder (will be freed before a thread exits)
            // convert subarrays into 256-bit chunks
            args = args_list[j];
            args.x = K_o + 8 * j;
            args.y = I + 8 * j;
            n_f = in_size - 8 * j;
            args.n_f = (n_f > 8) ? 8 : n_f;
            args.o = R_o;
            // run thread
            pthread_create(tid + j, NULL, func, args_list + j);
        }

        // join threads
        for (j=0; j<n_c; j++){
            pthread_join(tid[j], NULL);
        }
    }

    free(args_list);

    return;
}