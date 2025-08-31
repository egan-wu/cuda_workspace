## Attention Layer in LLM

### Main Target
Attempt to learn detail architecture and algorithm in LLM (large language model).
Starts from naive build-up to advanced optimizing technique (ex: flash attention, online-softmax, ...).
By comparing metrics changes between different algorithm to enhance understanding of LLM and Cuda implementation skills.

### Implemented Content
* Currently, implementation only handle forward-propagartion state
1. Attention Layer
   - QK matmul
     - `Q(seq_len, d_model)` @ `K_transpose(d_model, seq_len)`) = `S(seq_len, seq_len)`
     - using block-level shared memory

### Compile & Run
```
1. nvcc attentionLayerKernel.cu -o attentionLayer
2. attentionLayer.exe
```
