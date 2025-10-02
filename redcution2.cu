#include<cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void reduction2(const float* input, float* output){
  __shared__ float smem[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  smem[tid] = input[i];
  __syncThreads();

  for(int s=1; s<blockDim.x; s*=2){
    int index = 2 * s * tid;
    if(index < blockDim.x){
      sdata[index] += sdata[index + s];
    }
    __syncThreads();
  }
  if(tid==0){
    output[blockIdx.x] = smem[0];
  }
}
