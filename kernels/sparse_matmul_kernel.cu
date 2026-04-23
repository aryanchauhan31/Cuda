#include <cuda_runtime.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)   // 128

__global__ void matmul_kernel_v1(const float* A, const float* x, float* y,
                                 int M, int N, int nnz) {
    int lane          = threadIdx.x & (WARP_SIZE - 1);
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int row           = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    __shared__ float scratch[WARPS_PER_BLOCK][WARP_SIZE];

    float sum = 0.0f;
    if (row < M) {
        for (int j = lane; j < N; j += WARP_SIZE) {
            sum += A[row * N + j] * x[j];
        }
    }

    scratch[warp_in_block][lane] = sum;
    __syncwarp();

    if (lane < 16) scratch[warp_in_block][lane] += scratch[warp_in_block][lane + 16];
    __syncwarp();
    if (lane <  8) scratch[warp_in_block][lane] += scratch[warp_in_block][lane +  8];
    __syncwarp();
    if (lane <  4) scratch[warp_in_block][lane] += scratch[warp_in_block][lane +  4];
    __syncwarp();
    if (lane <  2) scratch[warp_in_block][lane] += scratch[warp_in_block][lane +  2];
    __syncwarp();
    if (lane <  1) scratch[warp_in_block][lane] += scratch[warp_in_block][lane +  1];
    __syncwarp();

    if (lane == 0 && row < M) {
        y[row] = scratch[warp_in_block][0];
    }
}

extern "C" void solve(const float* A, const float* x, float* y,
                      int M, int N, int nnz) {
    int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    matmul_kernel_v1<<<blocks, THREADS_PER_BLOCK>>>(A, x, y, M, N, nnz);
}
