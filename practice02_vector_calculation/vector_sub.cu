#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorSub(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] - B[i];
    }
}

int main() {

    // --- example of 1D vector sub

    const int N = 1024;
    const int size = N * sizeof(float);

    // allocate host memory
    float *h_A = new float[N]; 
    float *h_B = new float[N]; 
    float *h_C = new float[N]; 

    // initialize host memory
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 2.0f;
        h_B[i] = i * 1.0f;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}