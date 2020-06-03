#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512
#define INDEX_ROW_MAJOR_3(i, j, k, I, J, K) (k + K * (j + J * (i)))
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) (l + L * (k + K * (j + J * (i))))

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void conv_is(float *I, float *K, float *R, int iw, int ih, int ow, int oh, int kw, int kh, int sw, int sh, int ic, int oc){
    // input stationary
    int BLOCKS_PER_PIXEL = ceil(float(oc)/float(THREADS_PER_BLOCK));
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cid = bid % BLOCKS_PER_PIXEL; // channel block index (within pixel)
    int pid = bid / BLOCKS_PER_PIXEL; // pixel index
    // compute output pixel of the block
    int h = pid % oh;
    int w = pid / oh;
    // declare on-chip shared memory
    extern __shared__ float M[];
    // read input data once per block (shared across threads)
    // this process could serve as bottleneck, load distribution is critical
    // distribute indices across threads
    int full_idx = kw * kh * ic;
    int load_per_thread = ceil(float(full_idx)/float(THREADS_PER_BLOCK));
    int lower = load_per_thread * tid;
    int upper = load_per_thread * (tid + 1);
    if (lower < full_idx) {
        upper = (upper < full_idx)? upper : full_idx;
        for (int idx=lower; idx<upper; idx++){
            int k = idx%ic;
            int j = idx/ic%kh;
            int i = idx/ic/kh;
            M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] = I[INDEX_ROW_MAJOR_3(w*sw+i,h*sh+j,k, iw,ih,ic)];
        }
    }
    /*
    if(tid == 0){
        for (int i=0; i<kw; i++){
            for (int j=0; j<kh; j++){
                for (int k=0; k<ic; k++){
                    M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] = I[INDEX_ROW_MAJOR_3(w*sw+i,h*sh+j,k, iw,ih,ic)];
                }
            }
        }
    }
    */
    // compute block index in output channel dimension
    int ofs = cid * THREADS_PER_BLOCK;
    int n_tid = (oc - ofs < THREADS_PER_BLOCK)? (oc - ofs) : THREADS_PER_BLOCK;
    // handle boundary
    if (tid >= n_tid) return;

    // wait until data is ready
    __syncthreads();
    // apply convolution
    float *o = R + INDEX_ROW_MAJOR_3(w,h,ofs+tid, ow,oh,oc);
    for (int i=0; i<kw; i++){
        for (int j=0; j<kh; j++){
            for (int k=0; k<ic; k++){
                atomicAdd(o, M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] * K[INDEX_ROW_MAJOR_4(i,j,k,ofs+tid, kw,kh,ic,oc)]);
            }
        }
    }
}
__global__ void conv_ws(float *I, float *K, float *R, int iw, int ih, int ow, int oh, int kw, int kh, int sw, int sh, int ic, int oc){
    // weight stationary
    int BLOCKS_PER_CHANNEL = ceil(float(ow * oh)/float(THREADS_PER_BLOCK));
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int pid = bid % BLOCKS_PER_CHANNEL; // pixel block index (within channel)
    int cid = bid / BLOCKS_PER_CHANNEL; // output channel index
    // declare on-chip shared memory
    extern __shared__ float M[];
    // read input data once per block (shared across threads)
    // this process could serve as bottleneck, load distribution is critical
    // distribute indices across threads
    int full_idx = kw * kh * ic;
    int load_per_thread = ceil(float(full_idx)/float(THREADS_PER_BLOCK));
    int lower = load_per_thread * tid;
    int upper = load_per_thread * (tid + 1);
    if (lower < full_idx) {
        upper = (upper < full_idx)? upper : full_idx;
        for (int idx=lower; idx<upper; idx++){
            int k = idx%ic;
            int j = idx/ic%kh;
            int i = idx/ic/kh;
            M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] = K[INDEX_ROW_MAJOR_4(i,j,k,cid, kw,kh,ic,oc)];
        }
    }
    // compute block index in output pixel dimension
    int ofs = pid * THREADS_PER_BLOCK;
    int n_tid = (ow * oh - ofs < THREADS_PER_BLOCK)? (ow * oh - ofs) : THREADS_PER_BLOCK;
    // handle boundary
    if (tid >= n_tid) return;
    /*
    if(tid == 0){
        for (int i=0; i<kw; i++){
            for (int j=0; j<kh; j++){
                for (int k=0; k<ic; k++){
                    M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)] = K[INDEX_ROW_MAJOR_4(i,j,k,cid, kw,kh,ic,oc)];
                }
            }
        }
    }
    */
    // retrieve output pixel
    int pos = ofs + tid;
    int w = pos/oh;
    int h = pos%oh;
    float *o = R + INDEX_ROW_MAJOR_3(w,h,cid, ow,oh,oc);
    
    // wait until data is ready
    __syncthreads();
    // apply convolution
    for (int i=0; i<kw; i++){
        for (int j=0; j<kh; j++){
            for (int k=0; k<ic; k++){
                atomicAdd(o, I[INDEX_ROW_MAJOR_3(w*sw+i,h*sh+j,k, iw,ih,ic)] * M[INDEX_ROW_MAJOR_3(i,j,k, kw,kh,ic)]);
            }
        }
    }
}
extern "C"
void conv2d(float * I, float * K, float * R, int iw, int ih, int ow, int oh, int kw, int kh, int sw, int sh, int ic, int oc) {
    float *dev_I, *dev_K, *dev_R;
    // I: (iw * ih * ic), row major ordered
    // K: (kw * kh * ic * oc), row major ordered
    // R: (ow * oh * oc), row major ordered
    // todo: 2d convolution between I and K
    // loop over outer dimensions, and compute dot product in chunks of size 512
    // kernel function: convolution for a single sliding window
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, iw * ih * ic * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_K, kw * kh * ic * oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, ow * oh * oc * sizeof(float) ) );
    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, iw * ih * ic * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_K, K, kw * kh * ic * oc * sizeof(float), cudaMemcpyHostToDevice ) );

    // how to organize blocks?
    // maximizing data reuse and parallelism within a block
    // dynamic on-chip memory allocation
    int BLOCK_MEMSIZE = kw * kh * ic * sizeof(float);
    if (ow*oh > THREADS_PER_BLOCK){
        // weight stationary
        // within a block, hold kernel and thread over output pixels
        int BLOCKS_PER_CHANNEL = ceil(float(ow * oh)/float(THREADS_PER_BLOCK));
        int BLOCKS = oc * BLOCKS_PER_CHANNEL;
        conv_ws<<<BLOCKS,THREADS_PER_BLOCK,BLOCK_MEMSIZE>>>(dev_I, dev_K, dev_R, iw, ih, ow, oh, kw, kh, sw, sh, ic, oc);
    }else{
        // input stationary
        // within a block, hold input and thread over output channels
        int BLOCKS_PER_PIXEL = ceil(float(oc)/float(THREADS_PER_BLOCK));
        int BLOCKS = ow * oh * BLOCKS_PER_PIXEL;
        conv_is<<<BLOCKS,THREADS_PER_BLOCK,BLOCK_MEMSIZE>>>(dev_I, dev_K, dev_R, iw, ih, ow, oh, kw, kh, sw, sh, ic, oc);
    }
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, ow * oh * oc * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_K); cudaFree(dev_R);
}





__global__ void badd(float *I, float *B, float *R, int ow, int oh, int oc){
    int BLOCKS_PER_CHANNEL = ceil(float(ow * oh)/float(THREADS_PER_BLOCK));
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int pid = bid % BLOCKS_PER_CHANNEL; // pixel block index (within channel)
    int cid = bid / BLOCKS_PER_CHANNEL; // channel index
    // declare on-chip shared memory
    __shared__ float M[1];
    if(tid == 0) M[0] = B[cid];
    // compute block index in output pixel dimension
    int ofs = pid * THREADS_PER_BLOCK;
    // handle boundary
    if (tid >= (ow * oh - ofs < THREADS_PER_BLOCK)? (ow * oh - ofs) : THREADS_PER_BLOCK) return;
    // retrieve output pixel
    int pos = ofs + tid;
    int w = pos/oh;
    int h = pos%oh;
    ofs = INDEX_ROW_MAJOR_3(w,h,cid, ow,oh,oc);
    // wait until data is ready
    __syncthreads();
    // add
    atomicAdd(R + ofs, I[ofs] + M[0]);
}
extern "C"
void bias_add(float * I, float * B, float * R, int ow, int oh, int oc) {
    float *dev_I, *dev_B, *dev_R;
    // I: (ow * oh * oc), row major ordered
    // B: (oc)
    // R: (ow * oh * oc), row major ordered
    // todo: element-wise addition
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, ow * oh * oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_B, oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, ow * oh * oc * sizeof(float) ) );
    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, ow * oh * oc * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_B, B, oc * sizeof(float), cudaMemcpyHostToDevice ) );

    // block = channel, thread over pixels
    int BLOCKS_PER_CHANNEL = ceil(float(ow*oh)/float(THREADS_PER_BLOCK));
    int BLOCKS = oc * BLOCKS_PER_CHANNEL;
    badd<<<BLOCKS,THREADS_PER_BLOCK>>>(dev_I, dev_B, dev_R, ow, oh, oc);
    
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, ow * oh * oc * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_B); cudaFree(dev_R);
}






__global__ void lr(float *I, float *R, int ow, int oh, int oc){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // handle boundary
    int ofs = ow*oh*oc - bid*THREADS_PER_BLOCK;
    if (tid >= (ofs < THREADS_PER_BLOCK? ofs : THREADS_PER_BLOCK)) return;
    // add
    ofs = bid*THREADS_PER_BLOCK+tid;
    atomicAdd(R + ofs, I[ofs] * (I[ofs]>0? 1 : 0.1));
}
extern "C"
void leaky_relu(float * I, float * R, int ow, int oh, int oc) {
    float *dev_I, *dev_R;
    // I: (ow * oh * oc), row major ordered
    // R: (ow * oh * oc), row major ordered
    // todo: element-wise rectification
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, ow * oh * oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, ow * oh * oc * sizeof(float) ) );
    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, ow * oh * oc * sizeof(float), cudaMemcpyHostToDevice ) );
    // block = channel, thread over pixels
    int BLOCKS = ceil(float(ow*oh*oc)/float(THREADS_PER_BLOCK));
    lr<<<BLOCKS,THREADS_PER_BLOCK>>>(dev_I, dev_R, ow, oh, oc);
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, ow * oh * oc * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_R);
}





__global__ void bn(float *I, float *M, float *G, float *V, float *R, float eps, int ow, int oh, int oc){
    int BLOCKS_PER_CHANNEL = ceil(float(ow * oh)/float(THREADS_PER_BLOCK));
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int pid = bid % BLOCKS_PER_CHANNEL; // pixel block index (within channel)
    int cid = bid / BLOCKS_PER_CHANNEL; // channel index
    // declare on-chip shared memory
    __shared__ float Mem[4];
    if(tid == 0){
        Mem[0] = M[cid];
        Mem[1] = G[cid];
        Mem[2] = V[cid];
        Mem[3] = eps;
    }
    // compute block index in output pixel dimension
    int ofs = pid * THREADS_PER_BLOCK;
    // handle boundary
    if (tid >= (ow * oh - ofs < THREADS_PER_BLOCK)? (ow * oh - ofs) : THREADS_PER_BLOCK) return;
    // retrieve output pixel
    int pos = ofs + tid;
    int w = pos/oh;
    int h = pos%oh;
    ofs = INDEX_ROW_MAJOR_3(w,h,cid, ow,oh,oc);
    // wait until data is ready
    __syncthreads();
    // normalize
    atomicAdd(R + ofs, Mem[1] * (I[ofs] - Mem[0]) / (sqrt(Mem[2]) + Mem[3]));
}
extern "C"
void batch_norm(float * I, float * M, float * G, float * V, float * R, float eps, int ow, int oh, int oc) {
    float *dev_I, *dev_M, *dev_G, *dev_V, *dev_R;
    // I: (ow * oh * oc), row major ordered
    // M, G, V, R: (oc)
    // R: (ow * oh * oc), row major ordered
    // todo: element-wise normalization
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, ow * oh * oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_M, oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_G, oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_V, oc * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_R, ow * oh * oc * sizeof(float) ) );
    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, ow * oh * oc * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_M, M, oc * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_G, G, oc * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_V, V, oc * sizeof(float), cudaMemcpyHostToDevice ) );
    // block = channel, thread over pixels
    int BLOCKS_PER_CHANNEL = ceil(float(ow*oh)/float(THREADS_PER_BLOCK));
    int BLOCKS = oc * BLOCKS_PER_CHANNEL;
    bn<<<BLOCKS,THREADS_PER_BLOCK>>>(dev_I, dev_M, dev_G, dev_V, dev_R, eps, ow, oh, oc);
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( R, dev_R, ow * oh * oc * sizeof(float), cudaMemcpyDeviceToHost ) );
    // cleanup
    cudaFree(dev_I); cudaFree(dev_M); cudaFree(dev_G); cudaFree(dev_V); cudaFree(dev_R);
}