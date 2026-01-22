/**
 * Tensor Parallelism Implementation (CPU-based)
 * 
 * NOTE: This implementation uses CPU-based matrix multiplication and
 * simulates distributed communication using standard memory operations.
 * For actual GPU acceleration, replace cg_tensor_matmul with cuBLAS
 * and communication primitives with NCCL.
 */

#include "cg_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Stub for NCCL if not available in compile environment */
#ifndef NCCL_MAJOR
typedef void* ncclComm_t;
typedef void* cudaStream_t;
typedef int ncclResult_t;
#define ncclSuccess 0
#define ncclFloat 0
#define ncclSum 0
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, int datatype, ncclComm_t comm, void* stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, int datatype, int op, ncclComm_t comm, void* stream);
#endif

/*============================================================================
 * COLUMN PARALLEL LINEAR
 *============================================================================*/

cg_tensor* cg_tp_column_parallel_forward(cg_tensor_parallel_linear* tp,
                                         cg_tensor* input, cg_tensor* weight) {
    /* 
     * Y = X @ W_shard
     * 
     * Weight W is split along output dimension (columns).
     * Output Y is split along output dimension.
     * We perform local matmul.
     * 
     * If output_is_parallel is false, we AllGather the result.
     */
    
    /* 1. Copy input to GPU if needed (simulated) */
    /* Input X is replicated (all ranks have same X) */
    
    /* 2. Local MatMul */
    cg_tensor* output_shard = cg_tensor_new(NULL, 0, true); /* Placeholder shape */
    
    /* Infer shape: [batch, in_features] @ [in_features, out_features_shard] */
    int M = input->shape[0];
    int N = weight->shape[1]; /* Sharded output dim */
    int out_shape[] = {M, N};
    cg_tensor_reshape(output_shard, out_shape, 2);
    
    cg_tensor_matmul(input, weight, output_shard);
    
    /* 3. AllGather if needed */
    if (!tp->output_is_parallel && tp->group) {
        /* Create output tensor gathering all shards */
        int full_dim = output_shard->shape[1] * tp->group->world_size;
        int out_shape_full[] = {input->shape[0], full_dim};
        cg_tensor* output_full = cg_tensor_new(out_shape_full, 2, true);
        
        ncclComm_t comm = (ncclComm_t)tp->group->comm;
        cudaStream_t stream = (cudaStream_t)tp->group->stream;
        
        /* NCCL AllGather */
        ncclAllGather(output_shard->data, output_full->data, output_shard->size, 
                      ncclFloat, comm, stream);
                      
        return output_full;
    }
    
    return output_shard;
}

/*============================================================================
 * ROW PARALLEL LINEAR
 *============================================================================*/

cg_tensor* cg_tp_row_parallel_forward(cg_tensor_parallel_linear* tp,
                                      cg_tensor* input, cg_tensor* weight) {
    /* 1. Local MatMul: X_shard @ W_shard */
    cg_tensor* output_partial = cg_tensor_new(NULL, 0, true);
    
    int M = input->shape[0];
    int N = weight->shape[1];
    int out_shape[] = {M, N};
    cg_tensor_reshape(output_partial, out_shape, 2);
    
    cg_tensor_matmul(input, weight, output_partial);
    
    /* 2. AllReduce Sum */
    if (tp->group) {
        ncclComm_t comm = (ncclComm_t)tp->group->comm;
        cudaStream_t stream = (cudaStream_t)tp->group->stream;
        
        /* In-place AllReduce Sum */
        /* Note: NCCL supports in-place by passing same pointer for send/recv */
        ncclAllReduce(output_partial->data, output_partial->data, output_partial->size,
                      ncclFloat, ncclSum, comm, stream);
    }
    
    return output_partial;
}
