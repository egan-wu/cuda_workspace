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

    // create CUDA Events
    cudaEvent_t start_event, stop_event;
    cudaEvent_t copy_h2d_start_event, copy_h2d_stop_event;
    cudaEvent_t kernel_start_event, kernel_stop_event;
    cudaEvent_t copy_d2h_start_event, copy_d2h_stop_event;

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventCreate(&copy_h2d_start_event);
    cudaEventCreate(&copy_h2d_stop_event);
    cudaEventCreate(&kernel_start_event);
    cudaEventCreate(&kernel_stop_event);
    cudaEventCreate(&copy_d2h_start_event);
    cudaEventCreate(&copy_d2h_stop_event);

    float total_time = 0.0f;
    float h2d_time = 0.0f;
    float kernel_time = 0.0f;
    float d2h_time = 0.0f;

    // define kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start_event, 0);
    {
        cudaEventRecord(copy_h2d_start_event, 0);
            {
                // copy data from host to device
                // -- stream1 memcpy
                cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
                cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream1);

                // -- stream2 memcpy
                cudaMemcpyAsync(d_A + chunk_size, h_A + chunk_size, size, cudaMemcpyHostToDevice, stream2);
                cudaMemcpyAsync(d_B + chunk_size, h_B + chunk_size, size, cudaMemcpyHostToDevice, stream2);
            }
            cudaEventRecord(copy_h2d_stop_event, 0);
            cudaEventSynchronize(copy_h2d_stop_event);

            cudaEventRecord(kernel_start_event, 0);
            {
                // launch kernel
                // -- stream1 kernel
                vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, chunk_size);
                
                // -- stream2 kernel
                vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A + chunk_size, d_B + chunk_size, d_C + chunk_size, chunk_size);
            }
            cudaEventRecord(kernel_stop_event, 0);
            cudaEventSynchronize(kernel_stop_event);

            cudaEventRecord(copy_d2h_start_event, 0);
            {
                // copy result from device to host
                // -- stream1 memcpy
                cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);
                
                // -- stream2 memcpy
                cudaMemcpyAsync(h_C + chunk_size, d_C + chunk_size, size, cudaMemcpyDeviceToHost, stream2);
            }
            cudaEventRecord(copy_d2h_stop_event, 0);
            cudaEventSynchronize(copy_d2h_stop_event);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    // stream synchronize
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaEventElapsedTime(&total_time, start_event, stop_event);
    cudaEventElapsedTime(&h2d_time, copy_h2d_start_event, copy_h2d_stop_event);
    cudaEventElapsedTime(&kernel_time, kernel_start_event, kernel_stop_event);
    cudaEventElapsedTime(&d2h_time, copy_d2h_start_event, copy_d2h_stop_event);

    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "chunk size: " << chunk_size << std::endl;
    std::cout << "blocks per grid: " << blocksPerGrid << std::endl;
    std::cout << "thread per block: " << threadsPerBlock << std::endl;
    std::cout << "\n--- Performance Report ---" << std::endl;
    std::cout << "Total execution time: " << total_time << " ms" << std::endl;
    std::cout << "Host to Device copy time: " << h2d_time << " ms" << std::endl;
    std::cout << "Kernel execution time: " << kernel_time << " ms" << std::endl;
    std::cout << "Device to Host copy time: " << d2h_time << " ms" << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(copy_h2d_start_event);
    cudaEventDestroy(copy_h2d_stop_event);
    cudaEventDestroy(kernel_start_event);
    cudaEventDestroy(kernel_stop_event);
    cudaEventDestroy(copy_d2h_start_event);
    cudaEventDestroy(copy_d2h_stop_event);

    return 0;
}