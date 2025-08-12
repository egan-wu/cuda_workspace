This code is based on practice07, with adding profiling code

Record time elapse by CudaEvent
1. total execution time
2. Host to Device memory copy time
3. Kernel execution time
4. Device to Host memory copy time

Compile & Run
`
1. nvcc vector_add_stream.cu -o vectorAddStream
2. vectorAddStream
`