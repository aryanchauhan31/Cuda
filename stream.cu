#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#define THREADS_PER_BLOCK 256
inline int cdiv(int a, int b){ return (a + b - 1) / b; }

__global__ void reduction_kernel_int(const int* __restrict__ input,
                                     int* __restrict__ output, int N) {
    __shared__ int smem[THREADS_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    int sum = 0;
    if (i < (unsigned)N) sum = input[i];
    if (i + blockDim.x < (unsigned)N) sum += input[i + blockDim.x];
    smem[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = smem[0];
}

int main() {
    const int N = 10000;
    const int n0 = N/2, n1 = N - n0;

    // pinned host input and host partials (non-const types)
    int* h_in = nullptr;
    int* h_part0 = nullptr;
    int* h_part1 = nullptr;
    cudaMallocHost(&h_in,    N   * sizeof(int));                 // pinned [web:38]
    cudaMallocHost(&h_part0, std::max(1, cdiv(n0,2*THREADS_PER_BLOCK)) * sizeof(int)); // pinned [web:38]
    cudaMallocHost(&h_part1, std::max(1, cdiv(n1,2*THREADS_PER_BLOCK)) * sizeof(int)); // pinned [web:38]
    for (int i = 0; i < N; ++i) h_in[i] = i;

    // device buffers per stream
    int *d_in0=nullptr, *d_out0=nullptr, *d_in1=nullptr, *d_out1=nullptr;
    const int blocks0 = cdiv(n0, 2*THREADS_PER_BLOCK);           // two-load pattern [web:6]
    const int blocks1 = cdiv(n1, 2*THREADS_PER_BLOCK);           // two-load pattern [web:6]
    cudaMalloc(&d_in0,  std::max(1,n0)     * sizeof(int));
    cudaMalloc(&d_out0, std::max(1,blocks0)* sizeof(int));
    cudaMalloc(&d_in1,  std::max(1,n1)     * sizeof(int));
    cudaMalloc(&d_out1, std::max(1,blocks1)* sizeof(int));

    // two non-default streams
    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    // stream 0: first half
    if (n0 > 0) {
        cudaMemcpyAsync(d_in0, h_in, n0*sizeof(int),
                        cudaMemcpyHostToDevice, s0);             // dst void*, src const void* [web:30][web:38]
        reduction_kernel_int<<<blocks0, THREADS_PER_BLOCK, 0, s0>>>(d_in0, d_out0, n0); // compute [web:31]
        cudaMemcpyAsync(h_part0, d_out0, blocks0*sizeof(int),
                        cudaMemcpyDeviceToHost, s0);             // dst non-const host [web:30][web:38]
    }

    // stream 1: second half
    if (n1 > 0) {
        cudaMemcpyAsync(d_in1, h_in + n0, n1*sizeof(int),
                        cudaMemcpyHostToDevice, s1);             // async H2D [web:30][web:38]
        reduction_kernel_int<<<blocks1, THREADS_PER_BLOCK, 0, s1>>>(d_in1, d_out1, n1); // compute [web:31]
        cudaMemcpyAsync(h_part1, d_out1, blocks1*sizeof(int),
                        cudaMemcpyDeviceToHost, s1);             // async D2H [web:30][web:38]
    }

    // synchronize both streams
    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);

    long long total = 0;
    for (int b = 0; b < blocks0; ++b) total += h_part0[b];
    for (int b = 0; b < blocks1; ++b) total += h_part1[b];
    std::cout << "Sum = " << total << std::endl;

    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaFree(d_in0); cudaFree(d_out0); cudaFree(d_in1); cudaFree(d_out1);
    cudaFreeHost(h_in); cudaFreeHost(h_part0); cudaFreeHost(h_part1);  // ptr type is void* compatible [web:38]
    return 0;
}
