#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip> // For std::setw and std::setprecision
#include <cmath>   // For std::abs

// -- practice target
// 1. using shared memory
// 2. using dim3 as block
// 3. synchronize point
// 4. coalesced memory access

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ":" << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// define tile size being used
// meaning crop data into 16x16 each tile
const int TILE_SIZE = 4;

__global__ void matrixMultShared(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE]; 
    __shared__ float sB[TILE_SIZE][TILE_SIZE]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float Cvalue = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {

        // load A value into shared memory
        if (row < M && (tile * TILE_SIZE + tx) < K) {
            sA[ty][tx] = A[row * K + (tile * TILE_SIZE + tx)];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // load B value into shared memory
        if ((tile * TILE_SIZE + ty) < K && col < N) {
            sB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        // synchronize load-A & load-B
        __syncthreads();

        // 
        for (int i = 0; i < TILE_SIZE; ++i) {
            Cvalue += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

int main() {

    // aligned tile size for simplisity 
    int M = 1 * TILE_SIZE;
    int K = 2 * TILE_SIZE;
    int N = 3 * TILE_SIZE;

    int sizeA = M * K;
    int sizeB = K * N;
    int sizeC = M * N;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC);
    std::vector<float> h_C_cpu(sizeC);

    // initialize host memory
    for (int i = 0; i < sizeA; ++i) {
        h_A[i] = static_cast<float>(i % 10 + 1); 
    }
    for (int i = 0; i < sizeB; ++i) {
        h_B[i] = static_cast<float>((i % 5) + 0.5f);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid( (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y  );


    matrixMultShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << "\nResult Matrix C = (A @ B) from GPU:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}