#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 22; // vector number
    const int chunk_size = N / 2;
    const int size = chunk_size * sizeof(float); // total size of vectors
    const int total_byte = N * sizeof(float);

    // allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // initialize host data
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, total_byte);
    cudaMalloc(&d_B, total_byte);
    cudaMalloc(&d_C, total_byte);

    // create cuda-stream
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // define kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;

    // copy data from host to device
    // -- stream1 memcpy
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream1);
    
    // -- stream2 memcpy
    cudaMemcpyAsync(d_A + chunk_size, h_A + chunk_size, size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B + chunk_size, h_B + chunk_size, size, cudaMemcpyHostToDevice, stream2);

    // launch kernel
    // -- stream1 kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, chunk_size);
    
    // -- stream2 kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A + chunk_size, d_B + chunk_size, d_C + chunk_size, chunk_size);

    // copy result from device to host
    // -- stream1 memcpy
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);
    
    // -- stream2 memcpy
    cudaMemcpyAsync(h_C + chunk_size, d_C + chunk_size, size, cudaMemcpyDeviceToHost, stream2);

    // stream synchronize
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

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
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}