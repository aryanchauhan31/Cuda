#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_sum(int *d1, int *d2, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)
        d1[i] += d2[i];
}

int main(){
    // host memory
    int N=10;
    int *h1 = new int[N];
    for(int i=0; i<N; i++) h1[i] = i;
    
    int *h2 = new int[N];
    for(int i=0; i<N; i++) h2[i] = N-i;

    // device memory
    int *d1 = nullptr;
    int *d2 = nullptr;
    cudaMalloc((void**)&d1, N*sizeof(int));
    cudaMalloc((void**)&d2, N*sizeof(int));

    // H2D
    cudaMemcpy(d1, h1, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d2, h2, N*sizeof(int), cudaMemcpyHostToDevice);

    // launch
    dim3 block_size(N);
    dim3 grid_size(1);
    vector_sum<<<grid_size, block_size>>>(d1,d2, N);
    // D2H
    cudaMemcpy(h1, d1, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2, d2, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i=0; i<N; i++){
        printf("d  = %d\n", h1[i]);
    }
    cudaFree(d1);
    cudaFree(d2);
    return 0;
}
