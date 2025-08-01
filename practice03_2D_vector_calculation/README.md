This is a matrix add kernel

vector_add_2d : C[i] = A[i] + B[i]
- input
1. start pointer of A data
2. start pointer of B data
3. start pointer of C data
4. dimension-row
5. dimension-column


Compile & Run
`
1. nvcc vector_add_2d.cu -o vector_add_2d
2. vector_add_2d
`