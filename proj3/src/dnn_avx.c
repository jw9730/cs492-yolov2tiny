#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#define MAX_THREADS 8

/* __m256: 256-bit vector containing 8 floats */

/* MAT_MUL */
// common arguments
struct mm_global {
    int kernel_in;
    int kernel_out;
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
    int kernel_in = args->G->kernel_in;
    int kernel_out = args->G->kernel_out;
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
    int MAX_THREADS_PIX = 1;
    int MAX_THREADS_OUT = 8;
    float ratio = n_pixels / kernel_out;
    if (ratio >= 8.0){
        MAX_THREADS_PIX = MAX_THREADS;
        MAX_THREADS_OUT = 1;
    }
    else if (8.0 > ratio && ratio >= 1.0){
        MAX_THREADS_PIX = MAX_THREADS/2;
        MAX_THREADS_OUT = 2;
    }
    else if (1.0 > ratio && ratio >= 1/8){
        MAX_THREADS_PIX = MAX_THREADS/4;
        MAX_THREADS_OUT = 4;
    }
    else if (1/8 > ratio){
        MAX_THREADS_PIX = MAX_THREADS/8;
        MAX_THREADS_OUT = 8;
    }
    int pix_per_thread = ceil((float) n_pixels / (float) MAX_THREADS_PIX);
    int out_per_thread = ceil((float) kernel_out / (float) MAX_THREADS_OUT);
    int n_chunks = ceil((float) kernel_in / 8.0);

    // set up global context
    struct mm_global G[1];
    G->kernel_in = kernel_in;
    G->kernel_out = kernel_out;
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
    for (int i=0; i<t_pix_max; i++){
        for (int j=0; j<t_out_max; j++){
            // join thread
            pthread_join(tid[i * MAX_THREADS_OUT + j], NULL);
        }
    }
}

/* BIAS_ADD */
struct ba_global {
    int n_pixel;
    int n_chunks;
};
struct ba_args {
    struct ba_global * G;
    float * I_o;
    float * B_o;
    float * R_o;
    int n_o;
};
void * ba_func(void * aux) {
    struct ba_args * args = (struct ba_args *) aux;
    int n_pixel = args->G->n_pixel;
    int n_chunks = args->G->n_chunks;
    float * I_o = args->I_o;
    float * B_o = args->B_o;
    float * R_o = args->R_o;
    int n_o = args->n_o;
    // iterate over output channels
    for (int i=0; i<n_o; i++){
        int residue = n_pixel;
        float * x = I_o + i * n_pixel;
        float * y = B_o + i;
        float * o = R_o + i * n_pixel;
        __m256 vy = _mm256_set1_ps(*y);
        // compute elementwise sum
        for (int j=0; j<n_chunks-1; j++){
            __m256 vx = _mm256_loadu_ps(x);
            __m256 vo = _mm256_add_ps(vx, vy);
            memcpy(o, &vo, sizeof(float) * 8);
            // update loop variables
            residue -= 8; x += 8; o += 8;
        }
        // handle last chunk
        assert(residue<=8);
        __m256 vx = _mm256_setzero_ps();
        memcpy(&vx, x, sizeof(float) * residue);
        __m256 vo = _mm256_add_ps(vx, vy);
        memcpy(o, &vo, sizeof(float) * residue);
    }
}
void bias_add(float * I, float * B, float * R, int n_pixel, int n_channel){
    // I: (n_channel, n_pixel), row major ordered
    // B: (n_channel)
    // R: (n_channel, n_pixel), row major ordered

    // threading parameters
    int out_per_thread = ceil((float) n_channel / (float) MAX_THREADS);
    int n_chunks = ceil((float) n_pixel / 8.0);

    // set up threads
    pthread_t tid[MAX_THREADS];
    struct ba_args args_list[MAX_THREADS];
    int t_max = MAX_THREADS;

    // global context
    struct ba_global G[1];
    G->n_pixel = n_pixel;
    G->n_chunks = n_chunks;

    // loop variables
    struct ba_args * args = args_list;
    int out_residue = n_channel;
    float * I_o = I;
    float * B_o = B;
    float * R_o = R;

    for (int t=0; t<MAX_THREADS; t++){
        // set up thread arguments
        args->G = G;
        args->n_o = (out_residue < out_per_thread) ? out_residue : out_per_thread;
        args->I_o = I_o;
        args->B_o = B_o;
        args->R_o = R_o;
        // run thread
        pthread_create(tid + t, NULL, ba_func, args);
        args++;
        // processed boundary, exit
        if (out_residue < out_per_thread){
            t_max = t + 1;
            break;
        }
        // update loop vars
        I_o += n_pixel * out_per_thread;
        B_o += out_per_thread;
        R_o += n_pixel * out_per_thread;
        out_residue -= out_per_thread;
    }
    for (int t=0; t<t_max; t++){
        // join thread
        pthread_join(tid[t], NULL);
    }
}

/* BATCH_NORMALIZATION */
struct bn_global {
    int n_pixel;
    int n_chunks;
    float eps;
};
struct bn_args {
    struct bn_global * G;
    float * I_o;
    float * M_o;
    float * G_o;
    float * V_o;
    float * R_o;
    int n_o;
};
void * bn_func(void * aux) {
    // arguments
    struct bn_args * args = (struct bn_args *) aux;
    int n_pixel = args->G->n_pixel;
    int n_chunks = args->G->n_chunks;
    float eps = args->G->eps;
    float * I_o = args->I_o;
    float * M_o = args->M_o;
    float * G_o = args->G_o;
    float * V_o = args->V_o;
    float * R_o = args->R_o;
    int n_o = args->n_o;
    // iterate over output channels
    for (int i=0; i<n_o; i++){
        int residue = n_pixel;
        float * x = I_o + i * n_pixel;
        float * mu = M_o + i;
        float * gamma = G_o + i;
        float * var = V_o + i;
        float * o = R_o + i * n_pixel;
        __m256 v_mu = _mm256_set1_ps(-*mu);
        __m256 v_factor = _mm256_set1_ps((*gamma)/(sqrt(*var)+eps));
        // compute elementwise sum
        for (int j=0; j<n_chunks-1; j++){
            __m256 vx = _mm256_loadu_ps(x);
            __m256 v1 = _mm256_add_ps(vx, v_mu);
            __m256 vo = _mm256_mul_ps(v1, v_factor);
            memcpy(o, &vo, 8 * sizeof(float));
            // update loop variables
            residue -= 8; x += 8; o += 8;
        }
        // handle last chunk
        __m256 vx = _mm256_setzero_ps();
        memcpy(&vx, x, residue * sizeof(float));
        __m256 v1 = _mm256_add_ps(vx, v_mu);
        __m256 vo = _mm256_mul_ps(v1, v_factor);
        memcpy(o, &vo, residue * sizeof(float));
    }
}
void batch_norm(float * I, float * M, float * G, float * V, float * R, float eps, int n_pixel, int n_channel){
    // I: (n_channel, n_pixel), row major ordered
    // M: (n_channel)
    // G: (n_channel)
    // V: (n_channel)
    // R: (n_channel, n_pixel), row major ordered

    // threading parameters
    int out_per_thread = ceil((float) n_channel / (float) MAX_THREADS);
    int n_chunks = ceil((float) n_pixel / 8.0);

    // set up threads
    pthread_t tid[MAX_THREADS];
    struct bn_args args_list[MAX_THREADS];
    int t_max = MAX_THREADS;

    // global context
    struct bn_global GL[1];
    GL->n_pixel = n_pixel;
    GL->n_chunks = n_chunks;
    GL->eps = eps;

    // loop variables
    struct bn_args * args = args_list;
    int out_residue = n_channel;
    float * I_o = I;
    float * M_o = M;
    float * G_o = G;
    float * V_o = V;
    float * R_o = R;

    for (int t=0; t<MAX_THREADS; t++){
        //printf("%d\n", t);
        // set up thread arguments
        args->G = GL;
        args->n_o = (out_residue < out_per_thread) ? out_residue : out_per_thread;
        args->I_o = I_o;
        args->M_o = M_o;
        args->G_o = G_o;
        args->V_o = V_o;
        args->R_o = R_o;
        // run thread
        pthread_create(tid + t, NULL, bn_func, args);
        args++;
        // processed boundary, exit
        if (out_residue < out_per_thread){
            t_max = t + 1;
            break;
        }
        // update loop vars
        I_o += n_pixel * out_per_thread;
        M_o += out_per_thread;
        G_o += out_per_thread;
        V_o += out_per_thread;
        R_o += n_pixel * out_per_thread;
        out_residue -= out_per_thread;
    }
    //printf("<%d>\n", t_max);
    for (int t=0; t<t_max; t++){
        //printf("%d\n", t);
        // join thread
        pthread_join(tid[t], NULL);
    }
}

/* LEAKY_RELU */
struct lr_args {
    float * I_o;
    float * R_o;
    int n_o;
};
void * lr_func(void * aux) {
    // arguments
    struct lr_args * args = (struct lr_args *) aux;
    float * I_o = args->I_o;
    float * R_o = args->R_o;
    int n_o = args->n_o;
    int n_chunks = ceil((float) n_o / 8.0);

    int residue = n_o;
    float * x = I_o;
    float * o = R_o;
    __m256 leak = _mm256_set1_ps(0.1);
    // compute elementwise recfitication
    for (int j=0; j<n_chunks-1; j++){
        __m256 vx = _mm256_loadu_ps(x);
        __m256 vl = _mm256_mul_ps(vx, leak);
        __m256 vo = _mm256_max_ps(vx, vl);
        memcpy(o, &vo, 8 * sizeof(float));
        // update loop variables
        residue -= 8; x += 8; o += 8;
    }
    // handle last chunk
    __m256 vx = _mm256_setzero_ps();
    memcpy(&vx, x, residue * sizeof(float));
    __m256 vl = _mm256_mul_ps(vx, leak);
    __m256 vo = _mm256_max_ps(vx, vl);
    memcpy(o, &vo, residue * sizeof(float));
}
void leaky_relu(float * I, float * R, int length){
    // threading parameters
    int out_per_thread = ceil((float) length / (float) MAX_THREADS);

    // set up threads
    pthread_t tid[MAX_THREADS];
    struct lr_args args_list[MAX_THREADS];
    int t_max = MAX_THREADS;

    // loop variables
    struct lr_args * args = args_list;
    int out_residue = length;
    float * I_o = I;
    float * R_o = R;

    for (int t=0; t<MAX_THREADS; t++){
        // set up thread arguments
        args->n_o = (out_residue < out_per_thread) ? out_residue : out_per_thread;
        args->I_o = I_o;
        args->R_o = R_o;
        // run thread
        pthread_create(tid + t, NULL, lr_func, args);
        args++;
        // processed boundary, exit
        if (out_residue < out_per_thread){
            t_max = t + 1;
            break;
        }
        // update loop vars
        I_o += out_per_thread;
        R_o += out_per_thread;
        out_residue -= out_per_thread;
    }
    for (int t=0; t<t_max; t++){
        // join thread
        pthread_join(tid[t], NULL);
    }
}

/* MV_MUL */
// common arguments
struct mv_global {
    int in_channels;
    int n_chunks;
    float * I;
};
// thread-specific arguments
struct mv_args {
    struct mv_global * G;
    int n_o;
    float * K_o;
    float * R_o;
};
void * mv_func(void * aux) {
    // arguments
    struct mv_args * args = (struct mv_args *) aux;
    int in_channels = args->G->in_channels;
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
void mvmul(float * K, float * I, float * R, int in_channels, int out_channels) {
    // K: (in_channels, out_channels), column major ordered
    // I: (in_channels)
    // R: (out_channels)
    assert((K != NULL) && (I != NULL) && (R != NULL));

    // threading parameters
    int n_outs = ceil((float) out_channels / (float) MAX_THREADS);
    int n_chunks = ceil((float) in_channels / 8.0);

    // set up global context
    struct mv_global G[1];
    G->I = I;
    G->in_channels = in_channels;
    G->n_chunks = n_chunks;

    // set up threads
    pthread_t tid[MAX_THREADS];
    struct mv_args args_list[MAX_THREADS];
    int t = 0;
    int t_max = MAX_THREADS;

    // loop variables
    struct mv_args * args = args_list;
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
        pthread_create(tid + t, NULL, mv_func, args);
        // processed boundary, exit
        if (out_residue < n_outs){
            t_max = t + 1;
            break;
        }
        // update loop vars
        args++;
        K_o += in_channels * n_outs;
        R_o += n_outs;
        out_residue -= n_outs;
    }
    for (int i=0; i<t_max; i++){
        // join thread
        pthread_join(tid[i], NULL);
    }
}

/* MAX_POOL */
struct mp_global {
    int pool_size;
    int n_chunks;
};
struct mp_args {
    struct mp_global * G;
    float * I_o;
    float * R_o;
    int n_o;
};
void * mp_func(void * aux) {
    struct mp_args * args = (struct mp_args *) aux;
    int pool_size = args->G->pool_size;
    int n_chunks = args->G->n_chunks;
    float * I_o = args->I_o;
    float * R_o = args->R_o;
    int n_o = args->n_o;
    // iterate over pool space
    for (int i=0; i<n_o; i++){
        int residue = pool_size;
        float * x = I_o + i * pool_size;
        float * o = R_o + i;
        // compute max
        __m256 vm = _mm256_set1_ps(-1e20);
        for (int j=0; j<n_chunks-1; j++){
            __m256 vx = _mm256_loadu_ps(x);
            vm = _mm256_max_ps(vm, vx);
            // update loop variables
            residue -= 8; x += 8;
        }
        // handle last chunk
        assert(residue<=8);
        __m256 vx = _mm256_set1_ps(-1e20);
        memcpy(&vx, x, sizeof(float) * residue);
        // reference: https://stackoverflow.com/questions/9795529/how-to-find-the-horizontal-maximum-in-a-256-bit-avx-vector
        __m256 v1 = _mm256_max_ps(vm, vx); // e.g. v1=[1 2 3 4 5 6 7 8]
        __m256 v2 = _mm256_permute_ps(v1, 147); // 147: rotate left by upper 4 elements and lower 4 elements separately, v2=[2 3 4 1 6 7 8 5]
        __m256 v3 = _mm256_max_ps(v1, v2); // v3=[2 3 4 4 6 7 8 8]
        __m256 v4 = _mm256_permute_ps(v3, 147); // v4=[3 4 4 2 7 8 8 6]
        __m256 v5 = _mm256_max_ps(v3, v4); // v5=[3 4 4 4 7 8 8 8]
        __m256 v6 = _mm256_permute_ps(v5, 147); // v6=[4 4 4 3 8 8 8 7]
        vm = _mm256_max_ps(v5, v6); // contains max of upper four elements and lower 4 elements. v7=[4 4 4 4 8 8 8 8]
        float vm1 = ((float *)&vm)[0];
        float vm2 = ((float *)&vm)[4];
        *o = vm1 > vm2 ? vm1 : vm2;
    }
}
void max_pool(float * I, float * R, int out_size, int pool_size){
    // I: (out_size, pool_size), row major ordered
    // R: (out_size)

    // threading parameters
    int out_per_thread = ceil((float) out_size / (float) MAX_THREADS);
    int n_chunks = ceil((float) pool_size / 8.0);

    // set up threads
    pthread_t tid[MAX_THREADS];
    struct mp_args args_list[MAX_THREADS];
    int t_max = MAX_THREADS;

    // global context
    struct mp_global G[1];
    G->pool_size = pool_size;
    G->n_chunks = n_chunks;

    // loop variables
    struct mp_args * args = args_list;
    int out_residue = out_size;
    float * I_o = I;
    float * R_o = R;

    for (int t=0; t<MAX_THREADS; t++){
        // set up thread arguments
        args->G = G;
        args->n_o = (out_residue < out_per_thread) ? out_residue : out_per_thread;
        args->I_o = I_o;
        args->R_o = R_o;
        // run thread
        pthread_create(tid + t, NULL, mp_func, args);
        args++;
        // processed boundary, exit
        if (out_residue < out_per_thread){
            t_max = t + 1;
            break;
        }
        // update loop vars
        I_o += pool_size * out_per_thread;
        R_o += out_per_thread;
        out_residue -= out_per_thread;
    }
    for (int t=0; t<t_max; t++){
        // join thread
        pthread_join(tid[t], NULL);
    }
}