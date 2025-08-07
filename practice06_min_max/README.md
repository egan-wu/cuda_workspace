This is a min/max Kernel

reductionMin/Max : C = min/max(A[0], A[1], A[2], ... , A[n]) 
- input
1. start pointer of A data
3. pointer of C data
4. element count of A

Compile & Run
`
1. nvcc reductionMinMax.cu -o reductionMinMax
2. reductionAdd
`

Learning Target
1. Device function : comparing "__device__" and "__global__"
2. How to use "spin-lock" to write customized atomic function
3. Positive floating-point number remains the same sequence when converting into bit-format  : using "__int_as_float"
4. min / max locally reduction 