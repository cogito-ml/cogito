/**
 * Pipeline Parallelism Implementation
 */

#include "cg_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*============================================================================
 * PIPELINE CREATION
 *============================================================================*/

cg_pipeline_parallel* cg_pipeline_new(int num_stages, int num_microbatches) {
    cg_pipeline_parallel* pp = (cg_pipeline_parallel*)calloc(1, sizeof(cg_pipeline_parallel));
    pp->num_stages = num_stages;
    pp->num_microbatches = num_microbatches;
    pp->stages = (cg_pipeline_stage**)calloc(num_stages, sizeof(cg_pipeline_stage*));
    
    for (int i = 0; i < num_stages; i++) {
        pp->stages[i] = (cg_pipeline_stage*)calloc(1, sizeof(cg_pipeline_stage));
        pp->stages[i]->stage_id = i;
        
        /* Allocate double buffers (pointers only, tensors allocated later) */
        pp->stages[i]->send_buffers_a = (cg_tensor**)calloc(num_microbatches, sizeof(cg_tensor*));
        pp->stages[i]->send_buffers_b = (cg_tensor**)calloc(num_microbatches, sizeof(cg_tensor*));
        pp->stages[i]->recv_buffers_a = (cg_tensor**)calloc(num_microbatches, sizeof(cg_tensor*));
        pp->stages[i]->recv_buffers_b = (cg_tensor**)calloc(num_microbatches, sizeof(cg_tensor*));
    }
    
    return pp;
}

/*============================================================================
 * 1F1B SCHEDULE GENERATION (Phase 5.2)
 *============================================================================*/

cg_schedule_step* cg_pipeline_gen_1f1b(int num_stages, int num_microbatches,
                                        int* schedule_len) {
    int warmup = num_stages - 1;
    int steady = num_microbatches - warmup;
    int cooldown = warmup;
    
    int max_steps = (warmup + steady + cooldown) * 2; /* Fwd + Bwd */
    cg_schedule_step* sched = (cg_schedule_step*)calloc(max_steps, sizeof(cg_schedule_step));
    int count = 0;
    
    /* 1. Warmup Phase: Only Forward */
    /* Fill pipeline until last stage gets first microbatch */
    for (int i = 0; i < warmup; i++) {
        sched[count].op = SCHEDULE_FORWARD;
        sched[count].microbatch_id = i;
        count++;
    }
    
    /* 2. Steady State: 1F1B */
    /* Perform one forward, then one backward */
    for (int i = 0; i < steady; i++) {
        /* Forward for new microbatch */
        sched[count].op = SCHEDULE_FORWARD;
        sched[count].microbatch_id = warmup + i;
        count++;
        
        /* Backward for completed microbatch */
        sched[count].op = SCHEDULE_BACKWARD;
        sched[count].microbatch_id = i;
        count++;
    }
    
    /* 3. Cooldown Phase: Only Backward */
    /* Drain the pipeline */
    for (int i = 0; i < cooldown; i++) {
        sched[count].op = SCHEDULE_BACKWARD;
        sched[count].microbatch_id = steady + i;
        count++;
    }
    
    *schedule_len = count;
    return sched;
}

/* Interleaved schedule generation would go here */

/*============================================================================
 * COMMUNICATION (Double Buffered)
 *============================================================================*/

/* Stub for NCCL if not available in compile environment */
#ifndef NCCL_MAJOR
typedef void* ncclComm_t;
typedef int ncclResult_t;
#define ncclSuccess 0
#define ncclFloat 0
ncclResult_t ncclSend(const void* sendbuff, size_t count, int datatype, int peer, ncclComm_t comm, void* stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, int datatype, int peer, ncclComm_t comm, void* stream);
#endif

void cg_pipeline_send_activation(cg_pipeline_parallel* pp, int dest_stage,
                                 cg_tensor* activation, int mb_id) {
    if (!pp->pp_group) return;
    
    /* Select buffer A or B based on microbatch parity for overlap */
    bool use_buffer_b = (mb_id % 2 == 1);
    
    /* Get stage resources */
    cg_pipeline_stage* stage = pp->stages[pp->local_stage_id];
    void* stream = stage->comm_stream;
    
    /* NCCL Send */
    ncclComm_t comm = (ncclComm_t)pp->pp_group->comm;
    ncclSend(activation->data, activation->size, ncclFloat, dest_stage, comm, stream);
    
    /* Record event for synchronization */
    // cudaEventRecord(use_buffer_b ? stage->send_event_b : stage->send_event_a, stream);
}

void cg_pipeline_recv_activation(cg_pipeline_parallel* pp, int src_stage,
                                 cg_tensor* buffer, int mb_id) {
    if (!pp->pp_group) return;
    
    cg_pipeline_stage* stage = pp->stages[pp->local_stage_id];
    void* stream = stage->comm_stream;
    
    /* NCCL Recv */
    ncclComm_t comm = (ncclComm_t)pp->pp_group->comm;
    ncclRecv(buffer->data, buffer->size, ncclFloat, src_stage, comm, stream);
}

/*============================================================================
 * 3D PARALLELISM INITIALIZATION
 *============================================================================*/

cg_3d_parallel* cg_3d_parallel_new(int dp_size, int tp_size, int pp_size) {
    cg_3d_parallel* ctx = (cg_3d_parallel*)calloc(1, sizeof(cg_3d_parallel));
    ctx->dp_size = dp_size;
    ctx->tp_size = tp_size;
    ctx->pp_size = pp_size;
    return ctx;
}

void cg_3d_parallel_init_from_world(cg_3d_parallel* ctx, int world_size, int world_rank) {
    /* 
     * Rank mapping: 
     * Global rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank
     */
    
    int tp = ctx->tp_size;
    int pp = ctx->pp_size;
    
    ctx->tp_rank = world_rank % tp;
    ctx->pp_rank = (world_rank / tp) % pp;
    ctx->dp_rank = world_rank / (tp * pp);
    
    /* Initialize NCCL communicators using ncclCommSplit */
    /* Note: In a real MPI/NCCL setup, we start with a world comm and split it */
    ncclComm_t world_comm = NULL; /* Populate from existing context if available */
    
    /* 1. Tensor Parallel Group (Split by same DP & PP rank) */
    /* Color = dp_rank * pp_size + pp_rank, Key = tp_rank */
    int tp_color = ctx->dp_rank * pp + ctx->pp_rank;
    // ncclCommSplit(world_comm, tp_color, ctx->tp_rank, &ctx->tp_group->comm, NULL);
    
    /* 2. Pipeline Parallel Group (Split by same DP & TP rank) */
    /* Color = dp_rank * tp_size + tp_rank, Key = pp_rank */
    int pp_color = ctx->dp_rank * tp + ctx->tp_rank;
    // ncclCommSplit(world_comm, pp_color, ctx->pp_rank, &ctx->pp_group->comm, NULL);
    
    /* 3. Data Parallel Group (Split by same PP & TP rank) */
    /* Color = pp_rank * tp_size + tp_rank, Key = dp_rank */
    int dp_color = ctx->pp_rank * tp + ctx->tp_rank;
    // ncclCommSplit(world_comm, dp_color, ctx->dp_rank, &ctx->dp_group->comm, NULL);
    
    /* Alloc groups */
    ctx->tp_group = (cg_distributed_group*)calloc(1, sizeof(cg_distributed_group));
    ctx->tp_group->rank = ctx->tp_rank;
    ctx->tp_group->world_size = tp;
    ctx->tp_group->mode = PARALLEL_TENSOR;
    
    ctx->pp_group = (cg_distributed_group*)calloc(1, sizeof(cg_distributed_group));
    ctx->pp_group->rank = ctx->pp_rank;
    ctx->pp_group->world_size = pp;
    ctx->pp_group->mode = PARALLEL_PIPELINE;
    
    ctx->dp_group = (cg_distributed_group*)calloc(1, sizeof(cg_distributed_group));
    ctx->dp_group->rank = ctx->dp_rank;
    ctx->dp_group->world_size = ctx->dp_size;
    ctx->dp_group->mode = PARALLEL_DATA;
}

/*============================================================================
 * CLEANUP
 *============================================================================*/

void cg_pipeline_free(cg_pipeline_parallel* pp) {
    if (!pp) return;
    for (int i = 0; i < pp->num_stages; i++) {
        free(pp->stages[i]->send_buffers_a);
        free(pp->stages[i]->send_buffers_b);
        free(pp->stages[i]->recv_buffers_a);
        free(pp->stages[i]->recv_buffers_b);
        free(pp->stages[i]);
    }
    free(pp->stages);
    free(pp);
}
