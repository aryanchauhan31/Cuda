#include<cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void reduction3(const float* input, float* output){
  __shared__ float smem[THREADS_PER_BLOCK};

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  smem[tid] = (tid < N) ? input[tid] : 0.0f;
  __syncThreads();
  
  for(unsigned int s = blockDim.x/2; s>0; s>>=1){
    int index = s * 2 * tid;
    if(tid < s){
      smem[tid] += smem[tid + index];
    }
    __syncThreads();
  }
  if(tid==0){
    output[blockIdx.x] = smem[0];
  }
}
