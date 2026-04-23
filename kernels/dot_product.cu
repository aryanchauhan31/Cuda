#include <cuda_runtime.h>

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK/WARP_SIZE)
#define FULL_MASK 0xffffffffu
#define MAX_BLOCKS 4096

__device__ __forceinline__ float warp_reduce_sum(float v){
    #pragma unroll
    for( int offset = WARP_SIZE /2 ; offset>0;  offset>>=1){
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v){
    __shared__ float warp_scratch[WARPS_PER_BLOCK];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum(v); 
    if (lane == 0) warp_scratch[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = (lane < WARPS_PER_BLOCK) ? warp_scratch[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}

__global__ void dot_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ result,
                           int N){
    float partial = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        partial += A[i] * B[i];
    }

    float block_sum = block_reduce_sum(partial);

    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    cudaMemset(result, 0, sizeof(float));

    if (N <= 0) return;

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;

    dot_kernel<<<blocks, THREADS_PER_BLOCK>>>(A, B, result, N);
}

// Pytorch Wrapper Function
#include<torch/extension.h>

torch::Tensor dot_product_forward(torch::Tensor a, torch::Tensor b){
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be on CUDA DEVICE");
    TORCH_CHECK(a.dtype() == torch.kFloat32, "inputs must be Float32");
    TORCH_CHECK(a.size(0) == b.size(0), "inputs must be of same size");

    int N = a.size(0);

    auto result = torch::zeros({1}, a.options());

    int blocks = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    if (blocks>MAX_BLOCKS) blocks = MAX_BLOCKS;

    dot_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        N
    );

    return result;
}
