#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#define MAX_THREADS 16

/* __m256: 256-bit vector containing 8 floats */

// common arguments
struct global {
    int in_channels;
    int n_outs;
    int n_chunks;
    float * I;
};
// thread-specific arguments
struct args {
    struct global * G;
    int n_o;
    float * K_o;
    float * R_o;
};

void * func(void * aux) {
    // arguments
    struct args * args = (struct args *) aux;
    int in_channels = args->G->in_channels;
    int n_outs = args->G->n_outs;
    int n_chunks = args->G->n_chunks;
    float * I = args->G->I;
    int n_o = args->n_o;
    float * K_o = args->K_o;
    float * R_o = args->R_o;

    // iterate over output channels
    for (int i=0; i<n_o; i++){
        int residue = in_channels;
        float * x = I;
        float * y = K_o + i * in_channels;
        float * o = R_o + i;
        __m256 acc = _mm256_setzero_ps();

        // compute dot product between kernel and input
        for (int j=0; j<n_chunks-1; j++){
            // element-wise product, no aggregation
            __m256 vx = _mm256_loadu_ps(x);
            __m256 vy = _mm256_loadu_ps(y);
            __m256 vo = _mm256_mul_ps(vx, vy);
            acc = _mm256_add_ps(acc, vo);
            // update loop variables
            residue -= 8; x += 8; y += 8;
        }

        // handle last chunk
        __m256 vx = _mm256_setzero_ps();
        __m256 vy = _mm256_setzero_ps();
        memcpy(&vx, x, sizeof(float) * residue);
        memcpy(&vy, y, sizeof(float) * residue);
        __m256 vo = _mm256_mul_ps(vx, vy);
        acc = _mm256_add_ps(acc, vo);

        // accumulate
        float * res = (float *) &acc;
        for (int k=0; k<8; k++) *o += res[k];
    }
}

void ki_apply(float * K, float * I, float * R, int in_channels, int out_channels) {
    // arguments: column major ordered
    // K: (in_channels * out_channels)
    // I: (in_channels)
    // R: (out_channels)
    assert((K != NULL) && (I != NULL) && (R != NULL));

    // threading parameters
    int n_outs = ceil((float) out_channels / (float) MAX_THREADS);
    int n_chunks = ceil((float) in_channels / 8.0);

    // set up global context
    struct global G[1];
    G->I = I;
    G->in_channels = in_channels;
    G->n_chunks = n_chunks;
    G->n_outs = n_outs;

    // set up threads
    pthread_t tid[MAX_THREADS];
    struct args args_list[MAX_THREADS];
    int t = 0;

    // loop variables
    struct args * args = args_list;
    float * K_o = K;
    float * R_o = R;
    int out_residue = out_channels;

    for (t=0; t<MAX_THREADS; t++){
        // set up thread arguments
        args->G = G;
        args->n_o = (out_residue < n_outs) ? out_residue : n_outs;
        args->K_o = K_o;
        args->R_o = R_o;

        // run thread
        pthread_create(tid + t, NULL, func, args);
        
        // processed boundary, exit
        if (out_residue < n_outs) break;
        
        // update loop vars
        args++;
        K_o += in_channels * n_outs;
        R_o += n_outs;
        out_residue -= n_outs;
    }

    for (int i=0; i<=t; i++){
        // join thread
        pthread_join(tid[i], NULL);
    }

    return;
}