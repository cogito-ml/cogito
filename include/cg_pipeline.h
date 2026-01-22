/**
 * Pipeline Parallelism - 1F1B Schedule
 */

#ifndef CG_PIPELINE_H
#define CG_PIPELINE_H

#include "cg_tensor.h"
#include <stdbool.h>

/*============================================================================
 * DISTRIBUTED GROUP
 *============================================================================*/

typedef enum {
    PARALLEL_DATA,              /* Data parallelism */
    PARALLEL_TENSOR,            /* Tensor/model parallelism */
    PARALLEL_PIPELINE           /* Pipeline parallelism */
} cg_parallel_mode;

typedef struct {
    cg_parallel_mode mode;
    int world_size;             /* Total GPUs in group */
    int rank;                   /* This GPU's rank */
    int* ranks;                 /* All ranks in group */
    void* comm;                 /* NCCL communicator */
    void* stream;               /* CUDA stream */
} cg_distributed_group;

/*============================================================================
 * PIPELINE STAGE
 *============================================================================*/

typedef struct {
    int stage_id;
    void* layers;               /* Layers in this stage */
    int num_layers;
    
    /* Communication buffers (Double Buffered: A/B) */
    cg_tensor** send_buffers_a;   /* Activations to send forward (Set A) */
    cg_tensor** send_buffers_b;   /* Activations to send forward (Set B) */
    
    cg_tensor** recv_buffers_a;   /* Activations received (Set A) */
    cg_tensor** recv_buffers_b;   /* Activations received (Set B) */
    
    cg_tensor** grad_send_a;      /* Gradients to send backward (Set A) */
    cg_tensor** grad_send_b;      /* Gradients to send backward (Set B) */
    
    cg_tensor** grad_recv_a;      /* Gradients received (Set A) */
    cg_tensor** grad_recv_b;      /* Gradients received (Set B) */
    
    /* Events for synchronization */
    void* compute_event;
    void* comm_stream;            /* Dedicated stream for NCCL */
    void* send_event_a;           /* Event for buffer A send completion */
    void* send_event_b;           /* Event for buffer B send completion */
    void* recv_event_a;
    void* recv_event_b;
} cg_pipeline_stage;

/*============================================================================
 * PIPELINE PARALLEL CONTEXT
 *============================================================================*/

typedef struct {
    int num_stages;             /* Number of pipeline stages */
    int micro_batch_size;       /* Per-GPU micro-batch size */
    int num_microbatches;       /* Pipeline depth */
    int global_batch_size;      /* Total batch size */
    
    /* Parallel groups */
    cg_distributed_group* dp_group;  /* Data parallel group */
    cg_distributed_group* tp_group;  /* Tensor parallel group */
    cg_distributed_group* pp_group;  /* Pipeline parallel group */
    
    /* Stages */
    cg_pipeline_stage** stages;
    int local_stage_id;          /* Which stage this process owns */
    
    /* Activation checkpointing */
    cg_tensor**** activation_cache;  /* [stage][microbatch][layer] */
    bool use_recompute;              /* Activation checkpointing */
    
    /* Schedule */
    int** schedule;                  /* Pre-computed 1F1B schedule */
    int schedule_length;
} cg_pipeline_parallel;

/*============================================================================
 * PIPELINE API
 *============================================================================*/

/**
 * Initialize pipeline parallel context.
 */
cg_pipeline_parallel* cg_pipeline_new(int num_stages, int num_microbatches);

/**
 * Assign layers to stages.
 */
void cg_pipeline_set_stage(cg_pipeline_parallel* pp, int stage_id, 
                           void* layers, int num_layers);

/**
 * Run 1F1B schedule.
 */
void cg_pipeline_forward_backward(cg_pipeline_parallel* pp,
                                  cg_tensor* input, cg_tensor* target);

/**
 * All-reduce gradients within data parallel group.
 */
void cg_pipeline_allreduce_grads(cg_pipeline_parallel* pp);

void cg_pipeline_free(cg_pipeline_parallel* pp);

/*============================================================================
 * SCHEDULE PRIMITIVES
 *============================================================================*/

typedef enum {
    SCHEDULE_FORWARD,
    SCHEDULE_BACKWARD,
    SCHEDULE_SEND_FWD,
    SCHEDULE_RECV_FWD,
    SCHEDULE_SEND_BWD,
    SCHEDULE_RECV_BWD,
    SCHEDULE_WAIT,
    SCHEDULE_SYNC
} cg_schedule_op;

typedef struct {
    cg_schedule_op op;
    int microbatch_id;
    int stage_id;
} cg_schedule_step;

/**
 * Generate 1F1B schedule.
 */
cg_schedule_step* cg_pipeline_gen_1f1b(int num_stages, int num_microbatches,
                                        int* schedule_len);

/**
 * Generate interleaved schedule (for virtual stages).
 */
cg_schedule_step* cg_pipeline_gen_interleaved(int num_stages, int num_virtual,
                                               int num_microbatches, int* schedule_len);

/*============================================================================
 * COMMUNICATION
 *============================================================================*/

void cg_pipeline_send_activation(cg_pipeline_parallel* pp, int dest_stage,
                                 cg_tensor* activation, int microbatch_id);

void cg_pipeline_recv_activation(cg_pipeline_parallel* pp, int src_stage,
                                 cg_tensor* buffer, int microbatch_id);

void cg_pipeline_send_gradient(cg_pipeline_parallel* pp, int dest_stage,
                               cg_tensor* gradient, int microbatch_id);

void cg_pipeline_recv_gradient(cg_pipeline_parallel* pp, int src_stage,
                               cg_tensor* buffer, int microbatch_id);

/*============================================================================
 * TENSOR PARALLELISM (Column/Row Parallel)
 *============================================================================*/

typedef struct {
    cg_distributed_group* group;
    int shard_dim;              /* Dimension to shard (0=row, 1=col) */
    int input_is_parallel;      /* Input already sharded? */
    int output_is_parallel;     /* Keep output sharded? */
} cg_tensor_parallel_linear;

cg_tensor* cg_tp_column_parallel_forward(cg_tensor_parallel_linear* tp,
                                         cg_tensor* input, cg_tensor* weight);

cg_tensor* cg_tp_row_parallel_forward(cg_tensor_parallel_linear* tp,
                                      cg_tensor* input, cg_tensor* weight);

/*============================================================================
 * 3D PARALLELISM (DP + TP + PP)
 *============================================================================*/

typedef struct {
    int dp_size;                /* Data parallel size */
    int tp_size;                /* Tensor parallel size */
    int pp_size;                /* Pipeline parallel size */
    
    int dp_rank;
    int tp_rank;
    int pp_rank;
    
    cg_distributed_group* dp_group;
    cg_distributed_group* tp_group;
    cg_distributed_group* pp_group;
    
    cg_pipeline_parallel* pipeline;
} cg_3d_parallel;

cg_3d_parallel* cg_3d_parallel_new(int dp_size, int tp_size, int pp_size);
void cg_3d_parallel_init_from_world(cg_3d_parallel* ctx, int world_size, int world_rank);

#endif /* CG_PIPELINE_H */
