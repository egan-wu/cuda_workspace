#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define TILE_SIZE 256

__global__ void reductionKernel(const float* data, float* sum, int n) {
    // allocate shared memory
    __shared__ float sdata[TILE_SIZE];

    // global thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // local thread id
    int local_tid = threadIdx.x;

    // load data from global memory into shared memory
    sdata[local_tid] = (tid < n) ? data[tid] : 0.0f;

    // locally reduction by each block into sdata[0]
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            sdata[local_tid] += sdata[local_tid + s];
        }
        __syncthreads(); // synchronize all threads in block
    }

    if (local_tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

int main() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    float* h_data = new float[N];
    float* d_data;
    float* d_sum;
    float h_sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;
    }

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    reductionKernel<<<gridSize, blockSize>>>(d_data, d_sum, N);

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "The sum of the array is: " << h_sum << std::endl;

    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_sum);

    return 0;
}