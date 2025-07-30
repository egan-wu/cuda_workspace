#include <stdio.h>

__global__ void hello() {
    printf("Hello from thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 8>>>();  // 啟動 1 個 block，每個 block 8 個 thread
    cudaDeviceSynchronize();  // 等待所有 thread 執行完
    return 0;
}