#include <cuda_runtime.h>
#include <iostream>
#include <float.h>
#include <random>

#define N 1024
#define TILE_SIZE 16

__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old; // old = "int-format value stored in address"
        float f_assumed = __int_as_float(assumed); // cast assumed into float value
        float f_min = fminf(f_assumed, val); // compare address-float with shared-memory value
        old = atomicCAS(address_as_int, assumed, __float_as_int(f_min));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void minKernel(const float* data, float* min_value, int n) {
    // allocate shared memory
    __shared__ float sdata[TILE_SIZE];

    // global thread index / local thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // load data from global memory into shared memory
    sdata[local_tid] = (local_tid < n) ? data[tid] : FLT_MAX;

    for (unsigned int s = blockDim.x / 2; s > 0;  s >>= 1) {
        if (local_tid < s) {
            sdata[local_tid] = min(sdata[local_tid], sdata[local_tid + s]);
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        atomicMinFloat(min_value, sdata[0]);
    }
}

int main() {
    float* h_data = new float[N];
    float* d_data;
    float* d_min;
    float h_min = 0.0f;

    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_min, sizeof(float));

    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    float init = FLT_MAX;
    cudaMemcpy(d_min, &init, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    minKernel<<<gridSize, blockSize>>>(d_data, d_min, N);

    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "The min of the array is: " << h_min << std::endl;

    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_min);

    return 0;
}