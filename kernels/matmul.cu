#include <cuda_runtime.h>

#define TILE 16   

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    
    int tx = threadIdx.x;    
    int ty = threadIdx.y;    
    int col = blockIdx.x * TILE + tx;   
    int row = blockIdx.y * TILE + ty;   

    
    __shared__ float As[TILE][TILE];    
    __shared__ float Bs[TILE][TILE];    

    float acc = 0.0f;

    
    int num_tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        
        int a_col = t * TILE + tx;
        As[ty][tx] = (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;

        
        int b_row = t * TILE + ty;
        Bs[ty][tx] = (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;
        __syncthreads();   

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();   
    }

    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((K + TILE - 1) / TILE,
                       (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}
