#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256  
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__device__ float block_max(float val) {
    __shared__ float max_smem[WARPS_PER_BLOCK];
    __shared__ float shared_max;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;

    val = warp_max(val);
    if (lane == 0) max_smem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (lane < WARPS_PER_BLOCK) ? max_smem[lane] : -INFINITY;
        val = warp_max(val);
    }
    if (threadIdx.x == 0) shared_max = val;
    __syncthreads();
    return shared_max;
}

__device__ float block_sum(float val) {
    __shared__ float sum_smem[WARPS_PER_BLOCK];
    __shared__ float shared_sum;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;

    val = warp_sum(val);
    if (lane == 0) sum_smem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (lane < WARPS_PER_BLOCK) ? sum_smem[lane] : 0.0f;
        val = warp_sum(val);
    }
    if (threadIdx.x == 0) shared_sum = val;
    __syncthreads();
    return shared_sum;
}

__global__ void cross_entropy_kernel(const float* logits, const int* true_labels,
                                      double* loss_acc, int N, int C) {
    int j   = blockIdx.x;
    int tid = threadIdx.x;

    if (j >= N) return;

    const float* row = logits + j * C;

    float row_max = -INFINITY;
    for (int c = tid; c < C; c += blockDim.x)
        row_max = fmaxf(row_max, row[c]);
    row_max = block_max(row_max);

    float exp_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x)
        exp_sum += expf(row[c] - row_max);
    exp_sum = block_sum(exp_sum);

    if (tid == 0) {
        int true_class    = true_labels[j];
        float log_sum_exp = logf(exp_sum) + row_max;
        float loss_j      = log_sum_exp - row[true_class];
        atomicAdd(loss_acc, (double)loss_j);  
    }
}

extern "C" void solve(const float* logits, const int* true_labels,
                      float* loss, int N, int C) {
    double* d_acc;
    cudaMalloc(&d_acc, sizeof(double));
    cudaMemset(d_acc, 0, sizeof(double));

    int threads = min(C, THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    cross_entropy_kernel<<<N, threads>>>(logits, true_labels, d_acc, N, C);
    cudaDeviceSynchronize();

    double h_acc;
    cudaMemcpy(&h_acc, d_acc, sizeof(double), cudaMemcpyDeviceToHost);
    float result = (float)(h_acc / N);
    cudaMemcpy(loss, &result, sizeof(float), cudaMemcpyHostToDevice);

    cudaFree(d_acc);
}

// Pytorch Wrapper Function
#include <torch/extension.h>

torch::Tensor cross_entropy_forward(torch::Tensor logits, torch::Tensor labels) {
    TORCH_CHECK(logits.is_cuda() && labels.is_cuda(), "inputs must be on CUDA DEVICE");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be Float32");
    TORCH_CHECK(labels.dtype() == torch::kInt32, "labels must be Int32");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [N x C]");

    int N = logits.size(0);
    int C = logits.size(1);

    auto acc = torch::zeros({1}, logits.options().dtype(torch::kFloat64));

    int threads = min(C, THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    cross_entropy_kernel<<<N, threads>>>(
        logits.data_ptr<float>(),
        labels.data_ptr<int>(),
        acc.data_ptr<double>(),
        N, C
    );

    return (acc / N).to(torch::kFloat32);
}
