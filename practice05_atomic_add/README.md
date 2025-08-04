This is a reductionAdd Kernel

reductionAdd : C = A[0] + A[1] + A[2] + ... + A[n]
- input
1. start pointer of A data
3. pointer of C data
4. element count of A

Compile & Run
`
1. nvcc reductionAdd.cu -o reductionAdd
2. reductionAdd
`

Atomic Operations

- Arithmetic Operations
  - atomicAdd()
  - atomicSub()
  - atomicInc()
  - atomicDec()

- Bitwise Operations
  - atomicAnd()
  - atomicOr()
  - atomicXor()

- Min/Max Operations
  - atomicMin()小值。
  - atomicMax()大值。

- Exchange and Compare Operations
  - atomicExch()：exchange value with other
  - atomicCAS() (Compare-And-Swap)