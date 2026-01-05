/**
 * JIT Compiler - Graph Capture to PTX Generation
 * 
 * Traces computation graphs and compiles to optimized kernels.
 */

#ifndef CG_JIT_H
#define CG_JIT_H

#include "cg_tensor.h"
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/*============================================================================
 * IR OPCODES
 *============================================================================*/

typedef enum {
    OP_NOOP = 0,
    /* Elementwise */
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_NEG,
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_RSQRT,
    /* Activation */
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_GELU,
    OP_SOFTMAX,
    /* Linear algebra */
    OP_MATMUL,
    OP_TRANSPOSE,
    /* Convolution */
    OP_CONV2D,
    OP_MAXPOOL2D,
    /* Reduction */
    OP_SUM,
    OP_MEAN,
    OP_MAX,
    OP_MIN,
    /* Memory */
    OP_COPY,
    OP_RESHAPE,
    OP_BROADCAST,
    /* Custom */
    OP_FUSED_ATTENTION,
    OP_FUSED_LAYERNORM,
    OP_CUSTOM
} cg_ir_opcode;

/*============================================================================
 * IR NODE
 *============================================================================*/

typedef struct cg_ir_node {
    cg_ir_opcode opcode;
    int id;                      /* Unique node ID */
    
    /* Inputs */
    int* input_ids;              /* IDs of input tensors */
    int num_inputs;
    
    /* Output */
    int output_id;               /* ID of output tensor */
    int* output_shape;           /* Output shape (may be symbolic) */
    int output_ndim;
    
    /* Attributes */
    void* attributes;            /* Op-specific params */
    size_t attr_size;
    
    /* Scheduling */
    int schedule_order;          /* Topological order */
    bool can_fuse;               /* Can be fused with successor */
    struct cg_ir_node* fused_with;
} cg_ir_node;

/*============================================================================
 * IR GRAPH
 *============================================================================*/

typedef struct {
    cg_ir_node** nodes;
    int num_nodes;
    int capacity;
    
    /* Tensor registry */
    cg_tensor** tensors;
    int num_tensors;
    int tensor_capacity;
    
    /* Input/Output nodes */
    int* input_node_ids;
    int num_inputs;
    int* output_node_ids;
    int num_outputs;
    
    /* Optimization level */
    int opt_level;               /* 0=none, 1=fuse, 2=reorder, 3=aggressive */
} cg_ir_graph;

/*============================================================================
 * JIT COMPILER STATE
 *============================================================================*/

typedef enum {
    JIT_BACKEND_CPU,
    JIT_BACKEND_CUDA,
    JIT_BACKEND_METAL,           /* Apple Silicon */
    JIT_BACKEND_VULKAN           /* Cross-platform compute */
} cg_jit_backend;

typedef struct {
    cg_ir_graph* graph;
    cg_jit_backend backend;
    
    /* Generated code */
    char* source_code;           /* Generated C/CUDA/Metal code */
    size_t source_len;
    
    /* Compiled artifacts */
    void* compiled_module;       /* CUmodule / MTLLibrary / etc. */
    void* compiled_kernel;       /* CUfunction / MTLComputePipeline */
    
    /* Runtime info */
    bool is_compiled;
    int shared_memory_bytes;
    int registers_per_thread;
    int occupancy;
    
    /* Cache key */
    uint64_t shape_hash;         /* For shape-specialized recompilation */
} cg_jit_compiler;

/*============================================================================
 * TRACING API
 *============================================================================*/

/**
 * Begin tracing mode - all operations recorded to IR graph.
 */
void cg_jit_begin_trace(void);

/**
 * End tracing mode and return captured graph.
 */
cg_ir_graph* cg_jit_end_trace(void);

/**
 * Check if currently tracing.
 */
bool cg_jit_is_tracing(void);

/**
 * Record an operation to IR (called by tensor ops when tracing).
 */
cg_tensor* cg_jit_record_op(cg_ir_opcode op, cg_tensor** inputs, int num_inputs,
                            void* attributes, size_t attr_size);

/*============================================================================
 * GRAPH OPTIMIZATION
 *============================================================================*/

/**
 * Optimize IR graph.
 */
void cg_ir_optimize(cg_ir_graph* graph, int opt_level);

/**
 * Fuse compatible adjacent operations.
 */
void cg_ir_fuse_ops(cg_ir_graph* graph);

/**
 * Eliminate dead code.
 */
void cg_ir_dce(cg_ir_graph* graph);

/**
 * Common subexpression elimination.
 */
void cg_ir_cse(cg_ir_graph* graph);

/*============================================================================
 * COMPILATION
 *============================================================================*/

/**
 * Create JIT compiler for a graph.
 */
cg_jit_compiler* cg_jit_compiler_new(cg_ir_graph* graph, cg_jit_backend backend);

/**
 * Compile graph to target backend.
 */
bool cg_jit_compile(cg_jit_compiler* compiler);

/**
 * Execute compiled kernel.
 */
void cg_jit_execute(cg_jit_compiler* compiler, cg_tensor** inputs, int num_inputs,
                    cg_tensor** outputs, int num_outputs);

/**
 * Free compiler resources.
 */
void cg_jit_compiler_free(cg_jit_compiler* compiler);

/*============================================================================
 * CODE GENERATION
 *============================================================================*/

/**
 * Generate C code (for CPU backend).
 */
char* cg_jit_codegen_c(cg_ir_graph* graph);

/**
 * Generate CUDA code.
 */
char* cg_jit_codegen_cuda(cg_ir_graph* graph);

/**
 * Print IR graph for debugging.
 */
void cg_ir_graph_print(cg_ir_graph* graph);

/*============================================================================
 * GRAPH LIFECYCLE
 *============================================================================*/

cg_ir_graph* cg_ir_graph_new(void);
void cg_ir_graph_free(cg_ir_graph* graph);
cg_ir_node* cg_ir_node_new(cg_ir_opcode op);
void cg_ir_node_free(cg_ir_node* node);

#endif /* CG_JIT_H */
