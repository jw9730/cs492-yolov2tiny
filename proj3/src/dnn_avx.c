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
    __m256 x;
    __m256 y;
    float * o;
};

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
#ifdef DEBUG
    printf("func: acc += %f\n", acc);
#endif
    free(args);
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
            n_f = in_size - 8 * j;
            n_f = (n_f > 8) ? 8 : n_f;

#ifdef DEBUG
            printf("\nki_apply: chunk idx [%d]/[%d], # elements %d, args @ %p\n", j, n_c-1, n_f, args);

            printf("ki_apply: K [");
            for (int i=0; i<n_f; i++) printf("%3.2f ", (float *) (K_o+8*j));
            printf("]\n");
            printf("ki_apply: I [");
            for (int i=0; i<n_f; i++) printf("%3.2f ", (float *) (I+8*j));
            printf("]\n");
#endif
            // allocate an argument holder (will be freed before a thread exits)
            args = malloc(sizeof (struct args));
            memset(args, 0, sizeof (struct args));
            // convert subarrays into 256-bit chunks
            memcpy(&args->x, K_o+8*j, sizeof(float)*n_f);
            memcpy(&args->y, I+8*j, sizeof(float)*n_f);
            args->o = R_o;

#ifdef DEBUG
            printf("ki_apply: x [");
            for (int i=0; i<8; i++) printf("%3.2f ", (float *) &args->x[i]);
            printf("]\n");
            printf("ki_apply: y [");
            for (int i=0; i<8; i++) printf("%3.2f ", (float *) &args->y[i]);
            printf("]\n");
            
            printf("ki_apply: create thread %d\n", i * n_c + j);
#endif
            // run thread
            pthread_create(tid + (i * n_c + j), NULL, func, (void *)(args));
            args = NULL;
        }
    }

    for (int i=0; i<out_size; i++){
        for (int j=0; j<n_c; j++){
            pthread_join(tid[i * n_c + j], NULL);
            printf("thread %d ends\n", i * n_c + j);
        }
    }


#ifdef DEBUG
    printf("ki_apply: output [");
    for (int i=0; i<out_size; i++) printf("%3.2f ", R[i]);
    printf("]\n");
#endif

    return;
}
