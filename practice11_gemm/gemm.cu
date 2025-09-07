#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

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
    CUDA_CHECK(cudaMalloc(&d_mat1, sizeof(float) * size_of_mat1));
    CUDA_CHECK(cudaMalloc(&d_mat2, sizeof(float) * size_of_mat2));
    CUDA_CHECK(cudaMalloc(&d_mat3, sizeof(float) * size_of_mat3));
    CUDA_CHECK(cudaMemcpy(d_mat1, h_mat1, sizeof(float) * size_of_mat1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat2, h_mat2, sizeof(float) * size_of_mat2, cudaMemcpyHostToDevice));

    // ** naive GEMM **
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid(RNUP(BN, threadPerBlock.x), RNUP(BM, threadPerBlock.y)); // (x, y)
    float exec_time_naive = 0;
    CUDA_CHECK(cudaEventRecord(start_time));
    gemm_naive<<<blockPerGrid, threadPerBlock>>>(d_mat1, d_mat2, d_mat3, BM, BN, BK);
    CUDA_CHECK(cudaEventRecord(end_time));
    CUDA_CHECK(cudaEventSynchronize(end_time));
    cudaEventElapsedTime(&exec_time_naive, start_time, end_time);
    CUDA_CHECK(cudaMemcpy(d_mat3_rst_naive, d_mat3, sizeof(float) * size_of_mat3, cudaMemcpyDeviceToHost));


    // Verify the result
    cpu_gemm(h_mat1, h_mat2, h_mat3, BM, BN, BK);
    std::cout << BOLD << "Verification" << RESET << std::endl;
    std::cout << BOLD << "-------------------------------------" << RESET << std::endl;
    std::cout << std::left << std::setw(12) << "Naive GEMM" << " | " << std::right << std::setw(8) << exec_time_naive << " ms | ";
    isclose(d_mat3_rst_naive, h_mat3, size_of_mat3);

    delete[] h_mat1;
    delete[] h_mat2;
    delete[] h_mat3;
    delete[] d_mat3_rst_naive;
    CUDA_CHECK(cudaFree(d_mat1));
    CUDA_CHECK(cudaFree(d_mat2));
    CUDA_CHECK(cudaFree(d_mat3));
    return 0;
}