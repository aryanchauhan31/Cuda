#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

inline int cdiv(int a, int b){ return (a + b - 1)/b; }

__global__ void reduction_kernel(const float* input, float* output, int N){
    __shared__ float smem[THREADS_PER_BLOCK];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if(i<N) sum = input[i];
    if(i+ blockDim.x < N) sum += input[i + blockDim.x];
    smem[tid] = sum;
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
    if(tid < s){
        smem[tid] += smem[tid +s];
    }
    __syncthreads();
    }
    if(tid==0){
    output[blockIdx.x] = smem[0];
    }
}

// input, output are device pointers; output must have at least "blocks" elements
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threads(THREADS_PER_BLOCK);
    // each thread can load up to 2 elements
    dim3 blocks((N + threads.x * 2 - 1) / (threads.x * 2));
    reduction_kernel<<<blocks, threads>>>(input, output, N);
}
