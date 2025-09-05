
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

#define SEQ_LEN 512
#define D_MODEL 4096
#define BM      64
#define BN      64
#define BK      16
#define TILE    64

#define PRINT_EN   0
#define ALIGN_SIZE 12

#define RNUP(x, y) ((x + y - 1)/y)

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

template<typename Func>
double measure_cpu_time(Func f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count(); // return ms
}

// *** GPU related calculation ***
#define GEMM_TILE 16
#define NAIVE_MODE_EN      (1)
#define SHARED_MEM_MODE_EN (1)
extern "C"
__global__ void gemm_kernel
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    float bias
)
{

#if (NAIVE_MODE_EN == 1)
    /* Most simple and naive gemm */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            val += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = val + bias;
    }

#else // if (SHARED_MEM_MODE_EN == 1)
    __shared__ float smemA[GEMM_TILE][GEMM_TILE];
    __shared__ float smemB[GEMM_TILE][GEMM_TILE];

    int row = blockIdx.y * GEMM_TILE + threadIdx.y;
    int col = blockIdx.x * GEMM_TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < RNUP(K, GEMM_TILE); t++) {
        int k;
        
        // load data A into shared memory
        k = t * GEMM_TILE + threadIdx.x;
        if (row < M && k < K) {
            smemA[threadIdx.y][threadIdx.x] = A[row * K + k];
        } else {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load data B into shared memory
        k = t * GEMM_TILE + threadIdx.y;
        if (col < N && k < K) {
            smemB[threadIdx.y][threadIdx.x] = B[k * N + col];
        } else {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // make sure data are all moved into shared memory

        for (k = 0; k < GEMM_TILE; k++) {
            acc += smemA[threadIdx.y][k] * smemB[k][threadIdx.x];
        }

        __syncthreads(); // make sure all threads finish calculating
    }

    if (row < M && col < N) {
        C[row * N + col] = acc + bias;
    }
#endif
}

extern "C"
__global__ void qk_matmul 
(
    const float* __restrict__ Q, // Query matrix, shape=(N, d)
    const float* __restrict__ K, // Key matrix, shape=(N, d)
    float* __restrict__ S, // Score matrix, shape=(N, N)
    int N, // element count
    int d, // head dimension = hidden_dim / num_head
    float scale // scale value to Score
) 
{
    int block_i = blockIdx.y;
    int block_j = blockIdx.x;
    int i_base = block_i * BM;
    int j_base = block_j * BN;

    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int local_i = ty;
    int local_j = tx;

    int thread_per_row = 16;
    int thread_per_col = 16;

    extern __shared__ float smem[];     // dynamic shared memory
    float* smem_Q = smem;               // shared memory arrange for Q 
    float* smem_K = smem_Q + (BM * BK); // shared memory arrange for K

    for (int ii = local_i; ii < BM; ii+=thread_per_row) {
        for (int jj = local_j; jj < BN; jj+=thread_per_col) {
            // calculate global coordinated
            int global_i = i_base + ii;
            int global_j = j_base + jj;
            float acc = 0.0f;

            // load Q, K along d
            for (int k0 = 0; k0 < d; k0+=BK) {
                int k_size = min(BK, d - k0); // over boundary of x-dim (d)

                // load Q tile
                for (int kk = 0; kk < k_size; kk++) {
                    int sQ_idx = ii * BK + kk;
                    int q_row = global_i;
                    if (q_row < N) {
                        smem_Q[sQ_idx] = Q[(size_t)q_row * d + (k0 + kk)];
                    } else {
                        // over boundary of y-dim (N)
                        smem_Q[sQ_idx] = 0.0f;
                    }
                }

                // load K tile
                for (int kk = 0; kk < k_size; kk++) {
                    int sK_idx = jj * BK + kk;
                    int k_row = global_j;
                    if (k_row < N) {
                        smem_K[sK_idx] = K[(size_t)k_row * d + (k0 + kk)];
                    } else {
                        // over boundary of y-dim (N)
                        smem_K[sK_idx] = 0.0f;
                    }
                }

                __syncthreads(); // all threads finish loading

                for (int kk = 0; kk < k_size; kk++) {
                    float a = smem_Q[ii * BK + kk];
                    float b = smem_K[jj * BK + kk];
                    acc += a * b;
                }

                __syncthreads(); // all thread finish calculating
            } // finish calculating one S tile in shared memory

            // load S tile from shared memory into global memory
            if (global_i < N && global_j < N) {
                S[(size_t)global_i * N + global_j] = acc * scale;
            }

        }
    }
}

extern "C"
__global__ void softmax_row_max_kernel
(
    const float* __restrict__ score, 
    float* __restrict__ row_max, 
    int N
)
{
    // Each thread find max value of a row
    int ridx = blockIdx.x;
    if (ridx >= N) return;

    int tidx = threadIdx.x;
    float m = FLT_MIN;
    for (int j = tidx; j < N; j+=blockDim.x) {
        float v = score[(size_t)ridx * N + j];
        if (v > m) { m = v;}
    }

    __shared__ float smax;
    if (tidx == 0) {
        smax = FLT_MIN;
    }
    __syncthreads();

    atomicMax((int*)&smax, __float_as_int(m));
    __syncthreads();
    if (tidx == 0) {
        row_max[ridx] = smax;
    }
}

extern "C"
__global__ void softmax_row_norm_kernel
(
    float* __restrict__ score, 
    const float* __restrict__ row_max, 
    int N
)
{
    int ridx = blockIdx.x;
    if (ridx >= N) { return; }
    int tidx = threadIdx.x;
    float maxVal = row_max[ridx];
    float sumVal = 0.0f;
    for (int j = tidx; j < N; j+=blockDim.x) {
        float exp = expf(score[(size_t)ridx * N + j] - maxVal);
        score[(size_t)ridx * N + j] = exp;
        sumVal += exp;
    }

    __shared__ float smem_sum;
    if (tidx == 0) {
        smem_sum = 0.0f;
    }
    __syncthreads();

    atomicAdd(&smem_sum, sumVal);
    __syncthreads();

    float divVal = smem_sum;
    for (int j = tidx; j < N; j+=blockDim.x) {
        score[(size_t)ridx * N + j] = score[(size_t)ridx * N + j] / divVal;
    }
}

// *** CPU related calculation ***

void matmul
(
    const float* __restrict__ mat1,
    const float* __restrict__ mat2,
    float* __restrict__ mat,
    int i,
    int j,
    int k
)
{
    for (int _i = 0; _i < i; _i++) {
        for (int _k = 0; _k < k; _k++) {
            for (int _j = 0; _j < j; _j++) {
                mat[_i * j + _j] += mat1[_i * k + _k] * mat2[_k * j + _j];
            }
        }
    }
}

void cpu_qk_matmul
(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    float* __restrict__ S, 
    int N, 
    int d,
    float scale
) 
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++) {
                acc += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = acc * scale;
        }
    }
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

void cpu_softmax
(
    const float* __restrict__ S,
    float* __restrict__ S_weight,
    int N, 
    int d    
)
{
    for (int i = 0; i < N; i++) {
        // safe softmax, avoid overflow
        float max_val = S[i * d];
        for (int j = 1; j < d; j++) {
            max_val = std::max(max_val, S[i * d + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            S_weight[i * d + j] = std::exp(S[i * d + j] - max_val);
            sum += S_weight[i * d + j];
        }

        for (int j = 0; j < d; j++) {
            S_weight[i * d + j] /= sum;
        }

    }
}

int main() {

    // *** configuration
    std::cout << BOLD << "Configuration" << RESET << std::endl;
    std::cout << "Q:    (" << SEQ_LEN << ", " << D_MODEL << ")" << std::endl;
    std::cout << "K:    (" << SEQ_LEN << ", " << D_MODEL << ")" << std::endl;
    std::cout << "V:    (" << SEQ_LEN << ", " << D_MODEL << ")" << std::endl;
    std::cout << "S:    (" << SEQ_LEN << ", " << SEQ_LEN << ")" << std::endl;
    std::cout << "Out:  (" << SEQ_LEN << ", " << D_MODEL << ")" << std::endl;
    std::cout << "Tile: (" << BM << ", " << BN << ")" << std::endl;
    std::cout << std::endl;

    // *** init device data
    float* host_Q = new float[SEQ_LEN * D_MODEL];
    float* host_K = new float[SEQ_LEN * D_MODEL];
    float* host_V = new float[SEQ_LEN * D_MODEL];
    float* host_Out = new float[SEQ_LEN * D_MODEL];
    float* host_cal_S = new float[SEQ_LEN * SEQ_LEN];
    float* host_cal_softmax = new float[SEQ_LEN * SEQ_LEN];
    float* host_cal_weight_score = new float[SEQ_LEN * D_MODEL];
    float scale = 1.0f / std::sqrt((float)D_MODEL);
    int Q_element_size = SEQ_LEN * D_MODEL;
    int K_element_size = SEQ_LEN * D_MODEL;
    int V_element_size = SEQ_LEN * D_MODEL;
    int S_element_size = SEQ_LEN * SEQ_LEN;

    init_random(host_Q, Q_element_size, -1.0f, 1.0f);
    init_random(host_K, K_element_size, -1.0f, 1.0f);
    init_random(host_V, V_element_size, -1.0f, 1.0f);

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            host_cal_S[i * SEQ_LEN + j] = 0.0f;
        }
    }

    // *** CPU calculate
    std::cout << BOLD << "CPU starts calculating attention layer..." << RESET << std::endl;
    double cpu_exe_time = measure_cpu_time([&]() {
        cpu_qk_matmul(host_Q, host_K, host_cal_S, SEQ_LEN, D_MODEL, scale);
    });
    std::cout << std::left << std::setw(ALIGN_SIZE) << "QK matmul " << "| " << GREEN << cpu_exe_time << RESET << " ms\n";

    cpu_exe_time = measure_cpu_time([&]() {
        cpu_softmax(host_cal_S, host_cal_softmax, SEQ_LEN, SEQ_LEN);
    });
    std::cout << std::left << std::setw(ALIGN_SIZE) << "Softmax " << "| " << GREEN << cpu_exe_time << RESET << " ms\n";

    cpu_exe_time = measure_cpu_time([&]() {
        cpu_gemm(host_cal_softmax, host_V, host_cal_weight_score, SEQ_LEN, D_MODEL, SEQ_LEN);
    });
    std::cout << std::left << std::setw(ALIGN_SIZE) << "SV matmul " << "| " << GREEN << cpu_exe_time << RESET << " ms\n";

#if (PRINT_EN == 1)
    for (int i = 0; i < 10; i++) {
        std::cout << "[Row" << i << "]: ";
        for (int j = 0; j < 10; j++) {
            std::cout << host_cal_softmax[i * SEQ_LEN + j] << " ";
        }
        std::cout << "..." << std::endl;
    }
#endif

    std::cout << std::endl;

    // *** GPU calculate
    std::cout << BOLD << "GPU starts calculating attention layer..." << RESET << std::endl;

    float* dev_cal_S = new float[SEQ_LEN * SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            dev_cal_S[i * SEQ_LEN + j] = 0.0f;
        }
    }
    float* dev_cal_softmax = new float[SEQ_LEN * SEQ_LEN];
    float* dev_cal_out = new float[SEQ_LEN * D_MODEL];

    // --- cuda event
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    // --- move host data to device
    float* dev_Q;
    float* dev_K;
    float* dev_V;
    float* dev_S;
    float* dev_Out;
    float* dev_Smax;
    int Q_total_bytes = SEQ_LEN * D_MODEL * sizeof(float);
    int K_total_bytes = SEQ_LEN * D_MODEL * sizeof(float);
    int V_total_bytes = SEQ_LEN * D_MODEL * sizeof(float);
    int Out_total_bytes = SEQ_LEN * D_MODEL * sizeof(float);
    int S_total_bytes = SEQ_LEN * SEQ_LEN * sizeof(float);
    int Smax_total_bytes = SEQ_LEN * sizeof(float);
    cudaMalloc(&dev_Q, Q_total_bytes);
    cudaMalloc(&dev_K, K_total_bytes);
    cudaMalloc(&dev_V, V_total_bytes);
    cudaMalloc(&dev_S, S_total_bytes);
    cudaMalloc(&dev_Out, Out_total_bytes);
    cudaMalloc(&dev_Smax, Smax_total_bytes);
    cudaMemcpy(dev_Q, host_Q, Q_total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_K, host_K, K_total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V, host_V, V_total_bytes, cudaMemcpyHostToDevice);

    dim3 kernelBlock(256); // 16 * 16
    dim3 kernelGrid((SEQ_LEN + BN - 1) / BN, (SEQ_LEN + BM - 1) / BM);
    size_t smem_byte_size = (BM * BK + BK * BN) * sizeof(float);

    // *************
    // * QK Matmul *
    // *************
    cudaEventRecord(start_time);
    qk_matmul<<<kernelGrid, kernelBlock, smem_byte_size>>>(dev_Q, dev_K, dev_S, SEQ_LEN, D_MODEL, scale);
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    float gpu_exec_time = 0;
    cudaEventElapsedTime(&gpu_exec_time, start_time, end_time);
    std::cout << std::left << std::setw(ALIGN_SIZE) << "QK matmul " << "| " << GREEN << gpu_exec_time << RESET << " ms\n";
    cudaMemcpy(dev_cal_S, dev_S, S_total_bytes, cudaMemcpyDeviceToHost);

    // ***********
    // * Softmax *
    // ***********
    cudaEventRecord(start_time);
    softmax_row_max_kernel<<<SEQ_LEN, 256>>>(dev_S, dev_Smax, SEQ_LEN);
    softmax_row_norm_kernel<<<SEQ_LEN, 256>>>(dev_S, dev_Smax, SEQ_LEN);
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    gpu_exec_time = 0;
    cudaEventElapsedTime(&gpu_exec_time, start_time, end_time);
    std::cout << std::left << std::setw(ALIGN_SIZE) << "Softmax " << "| " << GREEN << gpu_exec_time << RESET << " ms\n";
    cudaMemcpy(dev_cal_softmax, dev_S, S_total_bytes, cudaMemcpyDeviceToHost);


    // *************
    // * SV matmul *
    // *************
    dim3 threadsPerBlock_gemm(GEMM_TILE, GEMM_TILE);
    dim3 blocksPerGrid_gemm(RNUP(D_MODEL, GEMM_TILE), RNUP(SEQ_LEN, GEMM_TILE));
    cudaEventRecord(start_time);
    gemm_kernel<<<blocksPerGrid_gemm, threadsPerBlock_gemm>>>(dev_S, dev_V, dev_Out, SEQ_LEN, D_MODEL, SEQ_LEN, 0.0f);
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    gpu_exec_time = 0;
    cudaEventElapsedTime(&gpu_exec_time, start_time, end_time);
    std::cout << std::left << std::setw(ALIGN_SIZE) << "SV matmul " << "| " << GREEN << gpu_exec_time << RESET << " ms\n";
    cudaMemcpy(dev_cal_out, dev_Out, Out_total_bytes, cudaMemcpyDeviceToHost);
    
#if (PRINT_EN == 1)
    for (int i = 0; i < 10; i++) {
        std::cout << "[Row" << i << "]: ";
        for (int j = 0; j < 10; j++) {
            std::cout << dev_cal_S[i * SEQ_LEN + j] << " ";
        }
        std::cout << "..." << std::endl;
    }
#endif

    std::cout << std::endl;
    std::cout << BOLD << "Comparing CPU result & GPU result..." << RESET << std::endl;
    std::cout << std::left << std::setw(ALIGN_SIZE) << "QK matmul" << "| ";
    isclose(host_cal_S, dev_cal_S, S_element_size);
    std::cout << std::left << std::setw(ALIGN_SIZE) << "Softmax" << "| ";
    isclose(host_cal_softmax, dev_cal_softmax, S_element_size);
    std::cout << std::left << std::setw(ALIGN_SIZE) << "SV matmul" << "| ";
    isclose(host_cal_weight_score, dev_cal_out, V_element_size);

    cudaFree(dev_Q);
    cudaFree(dev_K);
    cudaFree(dev_S);
    free(host_Q);
    free(host_K);
    free(dev_cal_S);
    free(host_cal_S);

    return 0;
}


