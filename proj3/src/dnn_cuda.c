#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#define MAX_THREADS 16

/* __m256: 256-bit vector containing 8 floats */

/* MAT_MUL */
// common arguments
struct mm_global {
    int n_pixels;
    int kernel_in;
    int kernel_out;
    int pix_per_thread;
    int out_per_thread;
    int n_chunks;
};
// thread-specific arguments
struct mm_args {
    struct mm_global * G;
    int n_pix;
    int n_out;
    float * I_o;
    float * K_o;
    float * R_o;
};
void * mm_func(void * aux) {
    struct mm_args * args = (struct mm_args *) aux;
    int n_pixels = args->G->n_pixels;
    int kernel_in = args->G->kernel_in;
    int kernel_out = args->G->kernel_out;
    int pix_per_thread = args->G->pix_per_thread;
    int out_per_thread = args->G->out_per_thread;
    int n_chunks = args->G->n_chunks;
    int n_pix = args->n_pix;
    int n_out = args->n_out;
    float * I_o = args->I_o;
    float * K_o = args->K_o;
    float * R_o = args->R_o;
    for (int i=0; i<n_pix; i++){
        for (int j=0; j<n_out; j++){
            int residue = kernel_in;
            float * x = I_o + i * kernel_in;
            float * y = K_o + j * kernel_in;
            float * o = R_o + i * kernel_out + j;
            __m256 acc = _mm256_setzero_ps();
            for (int k=0; k<n_chunks-1; k++){
                __m256 vx = _mm256_loadu_ps(x);
                __m256 vy = _mm256_loadu_ps(y);
                __m256 vo = _mm256_mul_ps(vx, vy);
                acc = _mm256_add_ps(acc, vo);
                residue -= 8; x += 8; y += 8;
            }
            __m256 vx = _mm256_setzero_ps();
            __m256 vy = _mm256_setzero_ps();
            memcpy(&vx, x, sizeof(float) * residue);
            memcpy(&vy, y, sizeof(float) * residue);
            __m256 vo = _mm256_mul_ps(vx, vy);
            acc = _mm256_add_ps(acc, vo);
            float * res = (float *) &acc;
            for (int k=0; k<8; k++) *o += res[k];
        }
    }
}
void matmul(float * I, float * K, float * R, int n_pixels, int kernel_in, int kernel_out) {
    // I: (n_pixels * kernel_in), row major ordered
    // K: (kernel_in * kernel_out), column major ordered
    // R: (n_pixels * kernel_out), row major ordered
    assert((I != NULL) && (K != NULL) && (R != NULL));
    assert(MAX_THREADS >= 8);
    // dynamic threading
    int MAX_THREADS_PIX = 2;
    int MAX_THREADS_OUT = 8;
    float ratio = n_pixels / kernel_out;
    if (ratio >= 8.0){
        MAX_THREADS_PIX = MAX_THREADS;
        MAX_THREADS_OUT = 1;
    }
    if (8.0 > ratio && ratio >= 1.0){
        MAX_THREADS_PIX = MAX_THREADS/2;
        MAX_THREADS_OUT = 2;
    }
    if (1.0 > ratio && ratio >= 1/8){
        MAX_THREADS_PIX = MAX_THREADS/4;
        MAX_THREADS_OUT = 4;
    }
    if (1/8 > ratio){
        MAX_THREADS_PIX = MAX_THREADS/8;
        MAX_THREADS_OUT = 8;
    }

    int pix_per_thread = ceil((float) n_pixels / (float) MAX_THREADS_PIX);
    int out_per_thread = ceil((float) kernel_out / (float) MAX_THREADS_OUT);
    int n_chunks = ceil((float) kernel_in / 8.0);

    // set up global context
    struct mm_global G[1];
    G->n_pixels = n_pixels;
    G->kernel_in = kernel_in;
    G->kernel_out = kernel_out;
    G->pix_per_thread = pix_per_thread;
    G->out_per_thread = out_per_thread;
    G->n_chunks = n_chunks;

    // set up threads
    pthread_t tid[MAX_THREADS];
    struct mm_args args_list[MAX_THREADS];
    int t_pix = 0, t_out = 0;
    int t_pix_max = MAX_THREADS_PIX, t_out_max = MAX_THREADS_OUT;

    // loop variables
    struct mm_args * args = args_list;
    int in_residue = n_pixels;

    for (t_pix=0; t_pix<MAX_THREADS_PIX; t_pix++){
        // loop variables
        int out_residue = kernel_out;
        float * K_o = K;

        for (t_out=0; t_out<MAX_THREADS_OUT; t_out++){
            int i_ofs = t_pix * pix_per_thread;
            int k_ofs = t_out * out_per_thread;
            // set up thread arguments
            args->G = G;
            args->n_pix = (in_residue < pix_per_thread) ? in_residue : pix_per_thread;
            args->n_out = (out_residue < out_per_thread) ? out_residue : out_per_thread;
            args->I_o = I + i_ofs * kernel_in;
            args->K_o = K + k_ofs * kernel_in;
            args->R_o = R + i_ofs * kernel_out + k_ofs;
            // run thread
            pthread_create(tid + (t_pix * MAX_THREADS_OUT + t_out), NULL, mm_func, args);
            //printf("%d, %d\n", t_pix, t_out);
            args++;
            // processed output boundary, exit
            if (out_residue < out_per_thread){
                t_out_max = t_out + 1;
                break;
            }
            // update loop vars
            out_residue -= out_per_thread;
        }
        // processed input boundary, exit
        if (in_residue < pix_per_thread){
            t_pix_max = t_pix + 1;
            break;
        }
        // update loop vars
        in_residue -= pix_per_thread;
    }
    //printf("<%d, %d>\n", t_pix_max, t_out_max);
    for (int i=0; i<t_pix_max; i++){
        for (int j=0; j<t_out_max; j++){
            // join thread
            //printf("%d, %d\n", i, j);
            pthread_join(tid[i * MAX_THREADS_OUT + j], NULL);
        }
    }
}
