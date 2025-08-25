#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include "utils.hpp"

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main () {

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    } else {
        std::cout << "Number of CUDA devices: " << device_count << std::endl;
    }

    const int N = 1 << 20;
    const int total_bytes = N * sizeof(float);

    // -- allocate unified memory by "cudaMallocaManaged"
    float *A, *B, *C;
    CUDA_CHECK(cudaMallocManaged((void**)&A, total_bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&B, total_bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&C, total_bytes));

    // initial memory data
    std::iota(A, A + N, 2.0f);
    std::iota(B, B + N, 4.0f);

    for (int i = 0; i < 10; ++i) {
        
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    std::cout << "Using device " << device_id << std::endl;
    CUDA_CHECK(cudaMemPrefetchAsync(A, total_bytes, device_id, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(B, total_bytes, device_id, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(C, total_bytes, device_id, 0));

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\n--- First 20 results ---" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << "A[" << i << "]: " << A[i] << " + B[" << i << "]: " << B[i] << " = C[" << i << "]: " << C[i] << std::endl;
    }

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return 0;
}
