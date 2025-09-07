This is a practice of implementing GEMM kernel (General Matrix Multiply) in different Cuda skills .
GEMM is one of the most common used calculation in AI model, which shows the importance of increasing efficiency of calculating GEMM.
By leveraging different Cuda skills, let's see how much improvment could that be.

## Implementation
1. [CPU] row-major calculation (baseline of CPU calculation)
2. [GPU] naive gemm (baseline of GPU calculation)
3. [GPU] shared memory gemm

Compile & Run
`
1. nvcc gemm.cu -o gemm
2. gemm
`