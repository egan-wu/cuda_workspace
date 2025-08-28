This is a reductionAdd Kernel considering bank conflict issue

reductionAdd : C = A[0] + A[1] + A[2] + ... + A[n]
- input
1. start pointer of A data
3. pointer of C data
4. element count of A

Compile & Run
`
1. nvcc reduceAdd_avoid_bank_conflict.cu -o reduceAdd
2. reduceAdd
`