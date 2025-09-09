#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <unordered_map>

#define BM  200
#define BN  300
#define BK  150
#define RNUP(x, y) (((x) + (y) - 1) / (y))

extern "C"
__global__ void gemm_naive
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, // A(M, K)
    int N, // B(K, N)
    int K
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = acc;
    }
}

#define GEMM_TILE_SIZE  16
extern "C"
__global__ void gemm_tiled
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, // A(M, K)
    int N, // B(K, N)
    int K
)
{
    __shared__ float sA[GEMM_TILE_SIZE][GEMM_TILE_SIZE];
    __shared__ float sB[GEMM_TILE_SIZE][GEMM_TILE_SIZE];

    int row = blockIdx.y * GEMM_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * GEMM_TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int tidx = 0; tidx < (K + GEMM_TILE_SIZE - 1) / GEMM_TILE_SIZE; tidx++) {
        // load data from A
        if (row < M && tidx * GEMM_TILE_SIZE + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + tidx * GEMM_TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load data from B
        if (col < N && tidx * GEMM_TILE_SIZE + threadIdx.y < K) {
            sB[threadIdx.y][threadIdx.x] = B[(tidx * GEMM_TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < GEMM_TILE_SIZE; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// *** General function ***
const std::string RESET = "\033[0m";
const std::string GREEN = "\033[32m";
const std::string RED = "\033[31m";
const std::string BOLD = "\033[1m";

struct Record {
    std::string name;
    float exec_time;
};

void init_random(float* arr, int size, float low = -1.0f, float high = 1.0f, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);
    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }
}

bool isclose(const float* a, const float* b, int size, float rtol = 1e-3f, float atol = 1e-5f) {
    for (int i = 0; i < size; i++) {
        float diff = std::fabs(a[i] - b[i]);
        float tol  = atol + rtol * std::fabs(b[i]);
        if (diff > tol) {
            std::cerr << RED << "fail" << RESET << " "
                      << "Mismatch at index " << i 
                      << " |a=" << a[i] 
                      << " b=" << b[i] 
                      << " diff=" << diff 
                      << " tol=" << tol << std::endl;
            return false;
        }
    }

    std::cout << GREEN << "pass" << RESET << std::endl;
    return true;
}

void cpu_gemm
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

int main() {

    std::unordered_map<std::string, Record> records;
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    float *h_mat1 = new float[BM * BK];
    float *h_mat2 = new float[BK * BN];
    float *h_mat3 = new float[BM * BN];
    int size_of_mat1 = BM * BK;
    int size_of_mat2 = BK * BN;
    int size_of_mat3 = BM * BN;
    init_random(h_mat1, BM * BK, -1.0f, 1.0f);
    init_random(h_mat2, BK * BN, -1.0f, 1.0f);
    std::fill(h_mat3, h_mat3 + size_of_mat3, 0.0f);

    float *d_mat1, *d_mat2, *d_mat3;
    float *d_mat3_rst_naive = new float[BM * BN];
    float *d_mat3_rst_tiled = new float[BM * BN];
    CUDA_CHECK(cudaMalloc(&d_mat1, sizeof(float) * size_of_mat1));
    CUDA_CHECK(cudaMalloc(&d_mat2, sizeof(float) * size_of_mat2));
    CUDA_CHECK(cudaMalloc(&d_mat3, sizeof(float) * size_of_mat3));
    CUDA_CHECK(cudaMemcpy(d_mat1, h_mat1, sizeof(float) * size_of_mat1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat2, h_mat2, sizeof(float) * size_of_mat2, cudaMemcpyHostToDevice));

    // ** naive GEMM **
    dim3 threadPerBlock_naive(16, 16);
    dim3 blockPerGrid_naive(RNUP(BN, threadPerBlock_naive.x), RNUP(BM, threadPerBlock_naive.y)); // (x, y)
    float exec_time_naive = 0;
    CUDA_CHECK(cudaEventRecord(start_time));
    gemm_naive<<<blockPerGrid_naive, threadPerBlock_naive>>>(d_mat1, d_mat2, d_mat3, BM, BN, BK);
    CUDA_CHECK(cudaEventRecord(end_time));
    CUDA_CHECK(cudaEventSynchronize(end_time));
    CUDA_CHECK(cudaMemcpy(d_mat3_rst_naive, d_mat3, sizeof(float) * size_of_mat3, cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&exec_time_naive, start_time, end_time);    

    // ** tiled GEMM **
    Record record_naive = {"Tiled GEMM", 0.0f};
    dim3 threadPerBlock_tiled(16, 16);
    dim3 blockPerGrid_tiled(RNUP(BN, threadPerBlock_tiled.x), RNUP(BM, threadPerBlock_tiled.y)); // (x, y)
    float exec_time_tiled = 0;
    CUDA_CHECK(cudaEventRecord(start_time));
    gemm_tiled<<<blockPerGrid_tiled, threadPerBlock_tiled>>>(d_mat1, d_mat2, d_mat3, BM, BN, BK);
    CUDA_CHECK(cudaEventRecord(end_time));
    CUDA_CHECK(cudaEventSynchronize(end_time));
    CUDA_CHECK(cudaMemcpy(d_mat3_rst_tiled, d_mat3, sizeof(float) * size_of_mat3, cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&exec_time_tiled, start_time, end_time);

    // Verify the result
    cpu_gemm(h_mat1, h_mat2, h_mat3, BM, BN, BK);
    std::cout << BOLD << "Verification" << RESET << std::endl;
    std::cout << BOLD << "-------------------------------------" << RESET << std::endl;
    std::cout << std::left << std::setw(12) << "Naive GEMM" << " | " << std::right << std::setw(8) << exec_time_naive << " ms | ";
    isclose(d_mat3_rst_naive, h_mat3, size_of_mat3);
    std::cout << std::left << std::setw(12) << "Tiled GEMM" << " | " << std::right << std::setw(8) << exec_time_tiled << " ms | ";
    isclose(d_mat3_rst_tiled, h_mat3, size_of_mat3);

    delete[] h_mat1;
    delete[] h_mat2;
    delete[] h_mat3;
    delete[] d_mat3_rst_naive;
    delete[] d_mat3_rst_tiled;
    CUDA_CHECK(cudaFree(d_mat1));
    CUDA_CHECK(cudaFree(d_mat2));
    CUDA_CHECK(cudaFree(d_mat3));
    return 0;
}