#include <cuda_runtime.h>

#define TILE 32   

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

#ifndef TORCH_EXTENSION_NAME
extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((K + TILE - 1) / TILE,
                       (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}
#endif

// PyTorch Wrapper Function
#include<torch/extension.h>

torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B){
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs should be on cuda device");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "inputs should be float32");
    TORCH_CHECK(A.dim()==2 && B.dim()==2, "inputs must be 2d");
    TORCH_CHECK(A.size(1)==B.size(0), "A cols must match B rows");

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    auto C = torch::zeros({M, K}, A.options());

    dim3 threads(TILE, TILE);
    dim3 blocks((K + TILE - 1) / TILE,
                (M + TILE - 1) / TILE);

    matmul_tiled_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

