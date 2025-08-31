# CUDA Programming Practice
## Projects
### Project01: **LLM Attention Layer**
   * Implement attention layer with different structure and algorithm
   * Learn LLM model structure and mechanism
   * Learn how to leverage Cuda parallelism to construct a modern AI model
   * Learn how to improve LLM issue (ex. flash attention(memory-bound))

## Practice
### Practice01: **Hello Cuda**
   * Print a hello world string to showcase CUDA environment is all set
     
### Practice02: **Vector Add (Basic)**
   * Implement a vectorAdd-kernel to perform vector addition
   * Learn GPU architecture: "threads, blocks"
   * Learn how to write a kernel: "\_\_global\_\_"
   * Learn how to allocate memory on device: "cudaMalloc(...)"
   * Learn how to copy data from host to device: "cudaMemcpy(...)"
### Practice03: **Vector Add (2D)**
   * Implement a 2D-vectorAdd-kernel perform 2D vector addition
   * Learn how to flatten multi-dimension vector into 1-dimension to perform calculation
### Practice04: **Matrix Multipy**
   * Implement a Matrix-Multiply-kernel to perform matrix multiplication
   * Learn how to crop data into tile
   * Learn how to allocate shared memory: "\_\_shared\_\_"
   * Learn how to leverage 3D-dim indexing: "threadIdx.x, threadIdx.y, ...", "blockIdx.x, blockIdx.y, ..."
### Practice05: **Atomic Operation**
   * Implement a sum-kernel to perform adding all elements in vector
   * Learn how to write local reduction algorithm: "for(uint32_t s = blockDim.x / 2; s > 0; s >>= 1)"
   * Learn how to use atomic operation: "atomicAdd(*addr, val)"
### Practice06: **Atomic CAS**
   * Implement a min-kernel to perform finding minimum among all elements in vector (positive numbers)
   * Learn how to write device-only function: "\_\_deviec\_\_"
   * Learn how to leverage bit-format: "__int_as_float(val)"
   * Learn how to write a self spinning function (soft-lock) with atomic function: "atomicCAS(*addr, compare, val)"
### Practice07: **Vector Add (Stream)**
   * Implement a vectorAdd-kernel to perform vector addition, by using multiple stream for leveraging pipeline mechanism
   * Learn how to create and use stream: "cudaStreamCreate(...), cudaStreamDestroy(...)"
   * Learn more types of async function: "cudaMemcpyAsync(...)"
### Practice08: **Profiling**
   * Implement a vectorAdd-kernel and leverage cudaEvent to measure kernel execution time
   * Learn how to create and use cudaEvent: "cudaEvent_t, cudaEventCreate(...), cudaEventRecord(...)"
   * Learn event synchronization: "cudaEventSynchronize(...)"
   * Learn other elapse time calculating function: "cudaEventElapsedTime(...)"
### Practice09: **Unified Memory**
   * Implement a vectoAdd-kernel and leverage unified memory for calculation
   * Learn how to create a unified memory in Cuda: "cudaMallocManaged(...)"
   * Learn how to determin target device for unified memory before use: "cudaMemPrefetchAsync(...)"
### Practice10: **Bank Conflict**
   * Implement a vectoAdd-kernel discover how bank conflict may affect performance, then resolve the problem by introducing padding mechanism
   * Learn how to introduce padding mechanism to mitigate bank conflict affection: "share_memory[size + size / wrap_size]"
