This is a matrix multiply kernel

vector_add_2d : C[i] = A[i] @ B[i]
- input
1. start pointer of A data
2. start pointer of B data
3. start pointer of C data
4. ROW of A = M
5. COL of A = K
6. COL of B = N


Compile & Run
`
1. nvcc matrix_multiply.cu -o matrix_multiply
2. matrix_multiply
`