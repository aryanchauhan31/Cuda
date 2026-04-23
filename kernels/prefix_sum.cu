#include<cuda_runtime.h>

__global__ void block_scan_kernel(const float* input,float* output, float* block_sums, int N){
  extern __shared__ double shared[];
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;

  shared[tid] = (gid < N) ? (double)input[tid] : 0.0;
  __syncthreads();

  for(int stride = 1; stride < blockDim.x; stride <<=1){
    double temp = (tid >= stride) ? shared[tid - stride] : 0.0;
    __syncthreads();
    shared[tid] += temp;
    __syncthreads();
  }
  if (gid < N) output[gid] = (float)shared[tid];
  if (block_sums != nullptr && tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = (float)shared[tid];
  }
}

__global__ void add_block_offsets_kernel(float* output, const float* block_sums, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < N) {
        output[gid] += block_sums[blockIdx.x - 1];
    }
}

void scan(const float* input, float* output, int N) {
    if (N <= 1024) {
        int threads = 1;
        while (threads < N) threads <<= 1;
        threads = min(threads, 1024);
        // double uses 8 bytes instead of 4
        block_scan_kernel<<<1, threads, threads * sizeof(double)>>>(input, output, nullptr, N);
        return;
    }

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    float* block_sums;
    float* scanned_block_sums;
    cudaMalloc(&block_sums, blocks * sizeof(float));
    cudaMalloc(&scanned_block_sums, blocks * sizeof(float));

    block_scan_kernel<<<blocks, threads, threads * sizeof(double)>>>(input, output, block_sums, N);
    scan(block_sums, scanned_block_sums, blocks);
    add_block_offsets_kernel<<<blocks, threads>>>(output, scanned_block_sums, N);

    cudaFree(block_sums);
    cudaFree(scanned_block_sums);
}

#ifndef TORCH_EXTENSION_NAME
extern "C" void solve(const float* input, float* output, int N) {
    scan(input, output, N);
    cudaDeviceSynchronize();
}
#endif

// PyTorch Wrapper Function 

#include<torch/extension.h>

torch::Tensor prefix_sum_forward(torch::Tensor input){
    TORCH_CHECK(input.is_cuda(), "input is on cuda device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input should be Float32");
    TORCH_CHECK(input.dim() == 1, "input should be 1D");

    int N = input.size(0);
    auto output = torch::empty_like(input);

    scan(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    cudaDeviceSynchronize();

    return output;
}
