#include <iostream>
#include <vector>

__global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = i / cols;
    int col = i % cols;

    if (row < rows && col < cols) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int rows = 5;
    const int cols = 5;
    int N = rows * cols;

    // allocate host memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_A, h_A.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // trigger kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    // wait kernel done and check error
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel execution: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // copy result from host to device
    cudaMemcpy(h_C.data(), d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // release device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\nResult Matrix C (A + B) from GPU:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_C[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}