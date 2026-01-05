/**
 * Automatic Kernel Fusion Engine
 * 
 * Fuses compatible operations into single kernels to reduce memory traffic.
 */

#ifndef CG_FUSION_H
#define CG_FUSION_H

#include "cg_jit.h"
#include <stdbool.h>

/*============================================================================
 * FUSION PATTERNS
 *============================================================================*/

typedef enum {
    FUSION_NONE = 0,
    FUSION_ELEMENTWISE,         /* a + b * c -> fused_abc */
    FUSION_REDUCTION,           /* sum(relu(x)) */
    FUSION_EPILOGUE,            /* matmul + bias + relu */
    FUSION_ATTENTION,           /* Q @ K.T -> softmax -> @ V */
    FUSION_LAYERNORM,           /* norm + scale + shift */
    FUSION_CUSTOM
} cg_fusion_type;

/*============================================================================
 * FUSION GROUP
 *============================================================================*/

typedef struct cg_fusion_group {
    cg_ir_node** nodes;         /* Nodes in this fusion group */
    int num_nodes;
    int capacity;
    
    cg_fusion_type fusion_type;
    char* kernel_name;          /* Generated kernel name */
    char* kernel_code;          /* Generated kernel source */
    
    /* Shape info */
    int* output_shape;
    int output_ndim;
    
    /* Memory analysis */
    size_t memory_saved;        /* Bytes saved by fusion */
    size_t flops;               /* FLOPs in fused kernel */
    float arithmetic_intensity; /* FLOPs / bytes */
    
    /* Scheduling */
    int schedule_priority;
    bool is_valid;
} cg_fusion_group;

/*============================================================================
 * FUSION PASS
 *============================================================================*/

typedef struct {
    cg_ir_graph* graph;
    cg_fusion_group** groups;
    int num_groups;
    int max_groups;
    
    /* Configuration */
    int max_fusion_depth;       /* Max ops per fusion */
    bool enable_epilogue;       /* Fuse matmul epilogues */
    bool enable_attention;      /* Fuse attention patterns */
    bool aggressive;            /* Fuse across control flow */
} cg_fusion_pass;

/*============================================================================
 * API
 *============================================================================*/

/**
 * Create fusion pass.
 */
cg_fusion_pass* cg_fusion_pass_new(cg_ir_graph* graph);

/**
 * Run fusion analysis on graph.
 */
void cg_fusion_analyze(cg_fusion_pass* pass);

/**
 * Find fusible patterns in graph.
 */
cg_fusion_group** cg_fusion_find_patterns(cg_ir_graph* graph, int* num_groups);

/**
 * Check if two ops can be fused.
 */
bool cg_fusion_can_fuse(cg_ir_node* a, cg_ir_node* b);

/**
 * Create fusion group from nodes.
 */
cg_fusion_group* cg_fusion_group_new(cg_ir_node** nodes, int count);

/**
 * Add node to fusion group.
 */
void cg_fusion_group_add(cg_fusion_group* group, cg_ir_node* node);

/**
 * Generate fused kernel code.
 */
char* cg_fusion_generate_kernel(cg_fusion_group* group);

/**
 * Apply fusion to graph (replace subgraph with fused op).
 */
void cg_fusion_apply(cg_fusion_pass* pass);

/**
 * Cleanup.
 */
void cg_fusion_group_free(cg_fusion_group* group);
void cg_fusion_pass_free(cg_fusion_pass* pass);

/*============================================================================
 * PATTERN MATCHERS
 *============================================================================*/

/**
 * Match: MATMUL -> (BIAS ->) ACTIVATION
 */
bool cg_fusion_match_matmul_epilogue(cg_ir_graph* graph, int start_idx,
                                     cg_fusion_group** out_group);

/**
 * Match: ReLU -> Dropout -> Add (common in transformers)
 */
bool cg_fusion_match_relu_dropout_add(cg_ir_graph* graph, int start_idx,
                                      cg_fusion_group** out_group);

/**
 * Match: LayerNorm pattern (mean -> var -> normalize -> scale -> shift)
 */
bool cg_fusion_match_layernorm(cg_ir_graph* graph, int start_idx,
                               cg_fusion_group** out_group);

/**
 * Match: Softmax(QK^T) @ V
 */
bool cg_fusion_match_attention(cg_ir_graph* graph, int start_idx,
                               cg_fusion_group** out_group);

#endif /* CG_FUSION_H */
