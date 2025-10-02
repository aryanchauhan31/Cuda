#include<cuda_runtime.h>
#define THREADS_PER_BLCOK 256
__global__ void reduce1(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];
  
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  
  for (unsigned int s=1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
  }
  __syncthreads();
  }
  
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
