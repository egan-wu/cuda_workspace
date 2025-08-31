
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

#define SEQ_LEN 512
#define D_MODEL 4096
#define BM  64
#define BN  64
#define BK  16

#define PRINT_EN 0

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
    std::cout << BOLD << "Comparing CPU result & GPU result..." << RESET << std::endl;
    bool fail = false;
    for (int i = 0; i < size; i++) {
        float diff = std::fabs(a[i] - b[i]);
        float tol  = atol + rtol * std::fabs(b[i]);
        if (diff > tol) {
            std::cerr << "Mismatch at index " << i 
                      << " |a=" << a[i] 
                      << " b=" << b[i] 
                      << " diff=" << diff 
                      << " tol=" << tol << std::endl;
            fail = true;
            break;
        }
    }

    std::cout << "is_close: ";
    if (fail) {
        std::cout << RED << "false" << RESET << std::endl;
    }
    else {
        std::cout << GREEN << "true" << RESET << std::endl;
    }
    
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
    int d
) 
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++) {
                acc += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = acc;
        }
    }
}

int main() {

    // *** configuration
    std::cout << BOLD << "Configuration" << RESET << std::endl;
    std::cout << "Q: (" << SEQ_LEN << ", " << D_MODEL << ")" << std::endl;
    std::cout << "K: (" << SEQ_LEN << ", " << D_MODEL << ")" << std::endl;
    std::cout << "S: (" << SEQ_LEN << ", " << SEQ_LEN << ")" << std::endl;
    std::cout << "Tile: (" << BM << ", " << BN << ")" << std::endl;
    std::cout << std::endl;

    // *** init device data
    float* host_Q = new float[SEQ_LEN * D_MODEL];
    float* host_K = new float[SEQ_LEN * D_MODEL];
    float* host_cal_S = new float[SEQ_LEN * SEQ_LEN];
    int Q_element_size = SEQ_LEN * D_MODEL;
    int K_element_size = SEQ_LEN * D_MODEL;
    int S_element_size = SEQ_LEN * SEQ_LEN;

    init_random(host_Q, Q_element_size, -1.0f, 1.0f);
    init_random(host_K, K_element_size, -1.0f, 1.0f);

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            host_cal_S[i * SEQ_LEN + j] = 0.0f;
        }
    }

    // *** CPU calculate
    std::cout << BOLD << "CPU starts calculating attention layer..." << RESET << std::endl;
    double cpu_exe_time = measure_cpu_time([&]() {
        cpu_qk_matmul(host_Q, host_K, host_cal_S, SEQ_LEN, D_MODEL);
    });
    std::cout << "CPU finished in " << GREEN << cpu_exe_time << RESET << " ms\n";

#if (PRINT_EN == 1)
    for (int i = 0; i < 10; i++) {
        std::cout << "[Row" << i << "]: ";
        for (int j = 0; j < 10; j++) {
            std::cout << host_cal_S[i * SEQ_LEN + j] << " ";
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

    // --- cuda event
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    // --- move host data to device
    float* dev_Q;
    float* dev_K;
    float* dev_S;
    int Q_total_bytes = SEQ_LEN * D_MODEL * sizeof(float);
    int K_total_bytes = SEQ_LEN * D_MODEL * sizeof(float);
    int S_total_bytes = SEQ_LEN * SEQ_LEN * sizeof(float);
    cudaMalloc(&dev_Q, Q_total_bytes);
    cudaMalloc(&dev_K, K_total_bytes);
    cudaMalloc(&dev_S, S_total_bytes);
    cudaMemcpy(dev_Q, host_Q, Q_total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_K, host_K, K_total_bytes, cudaMemcpyHostToDevice);

    dim3 kernelBlock(256); // 16 * 16
    dim3 kernelGrid((SEQ_LEN + BN - 1) / BN, (SEQ_LEN + BM - 1) / BM);
    size_t smem_byte_size = (BM * BK + BK * BN) * sizeof(float);

    cudaEventRecord(start_time);
    qk_matmul<<<kernelGrid, kernelBlock, smem_byte_size>>>(dev_Q, dev_K, dev_S, SEQ_LEN, D_MODEL, 1.0f);
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    float gpu_exec_time = 0;
    cudaEventElapsedTime(&gpu_exec_time, start_time, end_time);
    std::cout << "GPU kernel finished in " << GREEN << gpu_exec_time << RESET << " ms\n";

    cudaMemcpy(dev_cal_S, dev_S, S_total_bytes, cudaMemcpyDeviceToHost);

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
    isclose(host_cal_S, dev_cal_S, S_element_size);

    cudaFree(dev_Q);
    cudaFree(dev_K);
    cudaFree(dev_S);
    free(host_Q);
    free(host_K);
    free(dev_cal_S);
    free(host_cal_S);

    return 0;
}


