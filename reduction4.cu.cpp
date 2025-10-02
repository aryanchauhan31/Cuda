#include<cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void reduction4(const float* input, float* output, int N){
  __shared__ float smem[THREADS_PER_BLOCK];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  smem[tid] = (tid < N ) ? (input[tid] + input[tid + blockDim.x]) : 0.0f;
  __syncThreads();

  for(unsigned int s = blockDim.x / 2; s>0; s>>=1){
    int index  = 2 * s * tid;
    if(tid < N){
      smem[tid] += smem[tid + index];
    }
    __syncThreads();
  }
  if(tid==0){
    output[blockIdx.x] = smem[0];
  }
}
