/**
 * JIT Compiler - Graph Capture and Code Generation
 */

#include "cg_jit.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/*============================================================================
 * THREAD-LOCAL STATE
 *============================================================================*/

#ifdef _WIN32
static __declspec(thread) bool JIT_TRACING = false;
static __declspec(thread) cg_ir_graph* ACTIVE_GRAPH = NULL;
static __declspec(thread) int NEXT_TENSOR_ID = 0;
#else
static __thread bool JIT_TRACING = false;
static __thread cg_ir_graph* ACTIVE_GRAPH = NULL;
static __thread int NEXT_TENSOR_ID = 0;
#endif

/*============================================================================
 * IR NODE
 *============================================================================*/

cg_ir_node* cg_ir_node_new(cg_ir_opcode op) {
    cg_ir_node* node = (cg_ir_node*)calloc(1, sizeof(cg_ir_node));
    node->opcode = op;
    node->id = -1;
    return node;
}

void cg_ir_node_free(cg_ir_node* node) {
    if (!node) return;
    free(node->input_ids);
    free(node->output_shape);
    free(node->attributes);
    free(node);
}

static const char* opcode_names[] = {
    "NOOP", "ADD", "SUB", "MUL", "DIV", "NEG", "EXP", "LOG", "SQRT", "RSQRT",
    "RELU", "SIGMOID", "TANH", "GELU", "SOFTMAX",
    "MATMUL", "TRANSPOSE",
    "CONV2D", "MAXPOOL2D",
    "SUM", "MEAN", "MAX", "MIN",
    "COPY", "RESHAPE", "BROADCAST",
    "FUSED_ATTENTION", "FUSED_LAYERNORM", "CUSTOM"
};

const char* cg_opcode_name(cg_ir_opcode op) {
    if (op >= 0 && op <= OP_CUSTOM) {
        return opcode_names[op];
    }
    return "UNKNOWN";
}

/*============================================================================
 * IR GRAPH
 *============================================================================*/

cg_ir_graph* cg_ir_graph_new(void) {
    cg_ir_graph* graph = (cg_ir_graph*)calloc(1, sizeof(cg_ir_graph));
    graph->capacity = 128;
    graph->nodes = (cg_ir_node**)calloc(128, sizeof(cg_ir_node*));
    graph->tensor_capacity = 128;
    graph->tensors = (cg_tensor**)calloc(128, sizeof(cg_tensor*));
    graph->opt_level = 1;
    graph->shape_hash = 0;
    return graph;
}

void cg_ir_graph_free(cg_ir_graph* graph) {
    if (!graph) return;
    
    for (int i = 0; i < graph->num_nodes; i++) {
        cg_ir_node_free(graph->nodes[i]);
    }
    free(graph->nodes);
    free(graph->tensors);
    free(graph->input_node_ids);
    free(graph->output_node_ids);
    free(graph);
}

static void ir_graph_add_node(cg_ir_graph* graph, cg_ir_node* node) {
    if (graph->num_nodes >= graph->capacity) {
        graph->capacity *= 2;
        graph->nodes = (cg_ir_node**)realloc(graph->nodes, 
                                              graph->capacity * sizeof(cg_ir_node*));
    }
    node->id = graph->num_nodes;
    graph->nodes[graph->num_nodes++] = node;
}

/*============================================================================
 * TRACING API
 *============================================================================*/

void cg_jit_begin_trace(void) {
    if (JIT_TRACING) {
        cg_ir_graph_free(ACTIVE_GRAPH);
    }
    JIT_TRACING = true;
    ACTIVE_GRAPH = cg_ir_graph_new();
    NEXT_TENSOR_ID = 0;
}

cg_ir_graph* cg_jit_end_trace(void) {
    JIT_TRACING = false;
    cg_ir_graph* graph = ACTIVE_GRAPH;
    ACTIVE_GRAPH = NULL;
    return graph;
}

bool cg_jit_is_tracing(void) {
    return JIT_TRACING;
}

static int allocate_tensor_id(void) {
    return NEXT_TENSOR_ID++;
}

cg_tensor* cg_jit_record_op(cg_ir_opcode op, cg_tensor** inputs, int num_inputs,
                            void* attributes, size_t attr_size) {
    if (!JIT_TRACING || !ACTIVE_GRAPH) {
        return NULL;  /* Should call normal execution path */
    }
    
    cg_ir_node* node = cg_ir_node_new(op);
    node->num_inputs = num_inputs;
    
    if (num_inputs > 0) {
        node->input_ids = (int*)malloc(num_inputs * sizeof(int));
        for (int i = 0; i < num_inputs; i++) {
            node->input_ids[i] = inputs[i]->num_parents;  /* Use as tensor ID */
        }
    }
    
    /* Copy attributes */
    if (attributes && attr_size > 0) {
        node->attributes = malloc(attr_size);
        memcpy(node->attributes, attributes, attr_size);
        node->attr_size = attr_size;
    }
    
    /* Infer output shape (simplified - use first input's shape for most ops) */
    cg_tensor* first = inputs[0];
    node->output_ndim = first->ndim;
    node->output_shape = (int*)malloc(first->ndim * sizeof(int));
    memcpy(node->output_shape, first->shape, first->ndim * sizeof(int));
    
    /* Special cases for shape inference */
    if (op == OP_MATMUL && num_inputs == 2) {
        node->output_shape[node->output_ndim - 1] = inputs[1]->shape[inputs[1]->ndim - 1];
    }
    
    /* Create symbolic output tensor */
    node->output_id = allocate_tensor_id();
    cg_tensor* output = cg_tensor_new(node->output_shape, node->output_ndim, true);
    output->num_parents = node->output_id;  /* Store tensor ID */
    
    ir_graph_add_node(ACTIVE_GRAPH, node);
    
    /* Update shape hash */
    uint64_t step_hash = node->opcode;
    for (int i = 0; i < node->output_ndim; i++) {
        step_hash = (step_hash * 31) + node->output_shape[i];
    }
    /* Simple XOR-rotate mix */
    if (ACTIVE_GRAPH->num_nodes == 1) ACTIVE_GRAPH->shape_hash = 0;
    ACTIVE_GRAPH->shape_hash ^= (step_hash << (node->id % 64));
    
    return output;
}

/*============================================================================
 * GRAPH OPTIMIZATION
 *============================================================================*/

void cg_ir_optimize(cg_ir_graph* graph, int opt_level) {
    if (opt_level >= 1) {
        cg_ir_dce(graph);
    }
    if (opt_level >= 2) {
        cg_ir_cse(graph);
        cg_ir_fuse_ops(graph);
    }
}

static bool is_fusable_elementwise(cg_ir_opcode op) {
    return (op >= OP_ADD && op <= OP_RSQRT) || 
           (op >= OP_RELU && op <= OP_GELU);
}

void cg_ir_fuse_ops(cg_ir_graph* graph) {
    for (int i = 0; i < graph->num_nodes - 1; i++) {
        cg_ir_node* node = graph->nodes[i];
        cg_ir_node* next = graph->nodes[i + 1];
        
        /* Check if both are elementwise and can be fused */
        if (is_fusable_elementwise(node->opcode) && 
            is_fusable_elementwise(next->opcode)) {
            /* Check if next uses only this node's output */
            if (next->num_inputs == 1 && 
                next->input_ids[0] == node->output_id) {
                node->can_fuse = true;
                node->fusion_type = FUSION_VERTICAL;
                node->fused_with = next;
            }
        }
        
        /* HORIZONTAL FUSION */
        /* Check if node and next are independent and compatible */
        if (node->opcode == next->opcode && 
            node->output_ndim == next->output_ndim &&
            !node->can_fuse) {
            
            bool independent = true;
            /* Check if next depends on node */
            for (int k = 0; k < next->num_inputs; k++) {
                if (next->input_ids[k] == node->output_id) independent = false;
            }
            
            /* Check shapes match */
            bool shapes_match = true;
            for (int k = 0; k < node->output_ndim; k++) {
                if (node->output_shape[k] != next->output_shape[k]) shapes_match = false;
            }
            
            if (independent && shapes_match) {
                /* Fuse horizontally: Execute in same loop */
                node->can_fuse = true;
                node->fusion_type = FUSION_HORIZONTAL;
                node->fused_with = next;
                /* Note: In real implementation, we'd mark this as horizontal fusion type */
            }
        }
    }
}

void cg_ir_dce(cg_ir_graph* graph) {
    /* Mark used nodes */
    bool* used = (bool*)calloc(graph->num_nodes, sizeof(bool));
    
    /* Mark outputs as used */
    for (int i = 0; i < graph->num_outputs; i++) {
        used[graph->output_node_ids[i]] = true;
    }
    
    /* Propagate backwards */
    for (int i = graph->num_nodes - 1; i >= 0; i--) {
        if (used[i]) {
            cg_ir_node* node = graph->nodes[i];
            for (int j = 0; j < node->num_inputs; j++) {
                /* Mark input nodes as used */
                for (int k = 0; k < graph->num_nodes; k++) {
                    if (graph->nodes[k]->output_id == node->input_ids[j]) {
                        used[k] = true;
                    }
                }
            }
        }
    }
    
    /* Remove unused nodes */
    int write_idx = 0;
    for (int i = 0; i < graph->num_nodes; i++) {
        if (used[i]) {
            graph->nodes[write_idx++] = graph->nodes[i];
        } else {
            cg_ir_node_free(graph->nodes[i]);
        }
    }
    graph->num_nodes = write_idx;
    
    free(used);
}

void cg_ir_cse(cg_ir_graph* graph) {
    /* Simple CSE: find duplicate expressions */
    for (int i = 0; i < graph->num_nodes; i++) {
        cg_ir_node* node_i = graph->nodes[i];
        
        for (int j = i + 1; j < graph->num_nodes; j++) {
            cg_ir_node* node_j = graph->nodes[j];
            
            /* Check if same operation with same inputs */
            if (node_i->opcode == node_j->opcode &&
                node_i->num_inputs == node_j->num_inputs) {
                bool same = true;
                for (int k = 0; k < node_i->num_inputs && same; k++) {
                    if (node_i->input_ids[k] != node_j->input_ids[k]) {
                        same = false;
                    }
                }
                
                if (same) {
                    /* Replace uses of node_j with node_i */
                    for (int k = j + 1; k < graph->num_nodes; k++) {
                        cg_ir_node* node_k = graph->nodes[k];
                        for (int l = 0; l < node_k->num_inputs; l++) {
                            if (node_k->input_ids[l] == node_j->output_id) {
                                node_k->input_ids[l] = node_i->output_id;
                            }
                        }
                    }
                }
            }
        }
    }
}

/*============================================================================
 * VIRTUAL REGISTER ALLOCATION
 *============================================================================*/

void cg_ir_allocate_registers(cg_ir_graph* graph, int max_regs) {
    /* 1. Liveness Analysis */
    int* last_use = (int*)calloc(graph->num_nodes, sizeof(int));
    for (int i = 0; i < graph->num_nodes; i++) {
        last_use[i] = i; /* Default last use is definition */
        cg_ir_node* node = graph->nodes[i];
        for (int j = 0; j < node->num_inputs; j++) {
            /* Find producer of input (simplified mapping) */
            int producer_idx = node->input_ids[j]; 
            if (producer_idx < i) {
                last_use[producer_idx] = i;
            }
        }
    }
    
    /* 2. Linear Scan Allocation */
    int* reg_map = (int*)malloc(graph->num_nodes * sizeof(int));
    bool* reg_busy = (bool*)calloc(max_regs, sizeof(bool));
    int* reg_owner = (int*)malloc(max_regs * sizeof(int)); // Node index owning reg
    
    memset(reg_map, -1, graph->num_nodes * sizeof(int));
    
    for (int i = 0; i < graph->num_nodes; i++) {
        /* Expire old intervals */
        for (int r = 0; r < max_regs; r++) {
            if (reg_busy[r]) {
                int owner = reg_owner[r];
                if (last_use[owner] < i) {
                    reg_busy[r] = false; /* Free register */
                }
            }
        }
        
        /* Allocate new register */
        int best_reg = -1;
        for (int r = 0; r < max_regs; r++) {
            if (!reg_busy[r]) {
                best_reg = r;
                break;
            }
        }
        
        if (best_reg >= 0) {
            reg_busy[best_reg] = true;
            reg_owner[best_reg] = i;
            reg_map[i] = best_reg;
        } else {
            /* Spill to local memory (assign virtual ID > max) */
            reg_map[i] = i + max_regs; 
        }
    }
    
    free(last_use);
    free(reg_map);
    free(reg_busy);
    free(reg_owner);
}

/*============================================================================
 * C CODE GENERATION (CPU Backend)
 *============================================================================*/

static void codegen_c_header(FILE* f) {
    fprintf(f, "/* Auto-generated by Cogito JIT */\n");
    fprintf(f, "#include <math.h>\n");
    fprintf(f, "#include <string.h>\n\n");
}

static void codegen_c_op(FILE* f, cg_ir_node* node, int indent) {
    const char* pad = "    ";
    
    switch (node->opcode) {
        case OP_ADD:
            fprintf(f, "%sfor (int i = 0; i < size_%d; i++) {\n", pad, node->output_id);
            fprintf(f, "%s    t%d[i] = t%d[i] + t%d[i];\n", pad, 
                    node->output_id, node->input_ids[0], node->input_ids[1]);
            fprintf(f, "%s}\n", pad);
            break;
            
        case OP_MUL:
            fprintf(f, "%sfor (int i = 0; i < size_%d; i++) {\n", pad, node->output_id);
            fprintf(f, "%s    t%d[i] = t%d[i] * t%d[i];\n", pad,
                    node->output_id, node->input_ids[0], node->input_ids[1]);
            fprintf(f, "%s}\n", pad);
            break;
            
        case OP_RELU:
            fprintf(f, "%sfor (int i = 0; i < size_%d; i++) {\n", pad, node->output_id);
            fprintf(f, "%s    t%d[i] = t%d[i] > 0 ? t%d[i] : 0;\n", pad,
                    node->output_id, node->input_ids[0], node->input_ids[0]);
            fprintf(f, "%s}\n", pad);
            break;
            
        case OP_EXP:
            fprintf(f, "%sfor (int i = 0; i < size_%d; i++) {\n", pad, node->output_id);
            fprintf(f, "%s    t%d[i] = expf(t%d[i]);\n", pad,
                    node->output_id, node->input_ids[0]);
            fprintf(f, "%s}\n", pad);
            break;
            
        case OP_MATMUL:
            fprintf(f, "%s/* MATMUL: t%d = t%d @ t%d */\n", pad,
                    node->output_id, node->input_ids[0], node->input_ids[1]);
            fprintf(f, "%smemset(t%d, 0, size_%d * sizeof(float));\n", pad,
                    node->output_id, node->output_id);
            fprintf(f, "%sfor (int i = 0; i < M_%d; i++) {\n", pad, node->output_id);
            fprintf(f, "%s    for (int j = 0; j < N_%d; j++) {\n", pad, node->output_id);
            fprintf(f, "%s        for (int k = 0; k < K_%d; k++) {\n", pad, node->output_id);
            fprintf(f, "%s            t%d[i*N_%d+j] += t%d[i*K_%d+k] * t%d[k*N_%d+j];\n", 
                    pad, node->output_id, node->output_id,
                    node->input_ids[0], node->output_id, 
                    node->input_ids[1], node->output_id);
            fprintf(f, "%s        }\n", pad);
            fprintf(f, "%s    }\n", pad);
            fprintf(f, "%s}\n", pad);
            break;
            
        default:
            fprintf(f, "%s/* TODO: codegen for %s */\n", pad, cg_opcode_name(node->opcode));
            break;
    }
}

char* cg_jit_codegen_c(cg_ir_graph* graph) {
    FILE* f = tmpfile();
    if (!f) return NULL;
    
    codegen_c_header(f);
    
    /* Function signature */
    fprintf(f, "void jit_kernel(");
    for (int i = 0; i < graph->num_inputs; i++) {
        fprintf(f, "float* t%d, int size_%d", 
                graph->input_node_ids[i], graph->input_node_ids[i]);
        if (i < graph->num_inputs - 1) fprintf(f, ", ");
    }
    for (int i = 0; i < graph->num_outputs; i++) {
        fprintf(f, ", float* t%d, int size_%d",
                graph->output_node_ids[i], graph->output_node_ids[i]);
    }
    fprintf(f, ") {\n");
    
    /* Allocate temporaries */
    for (int i = 0; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        bool is_input = false;
        for (int j = 0; j < graph->num_inputs && !is_input; j++) {
            if (graph->input_node_ids[j] == node->output_id) is_input = true;
        }
        bool is_output = false;
        for (int j = 0; j < graph->num_outputs && !is_output; j++) {
            if (graph->output_node_ids[j] == node->output_id) is_output = true;
        }
        
        if (!is_input && !is_output) {
            int size = 1;
            for (int d = 0; d < node->output_ndim; d++) {
                size *= node->output_shape[d];
            }
            fprintf(f, "    float t%d[%d];\n", node->output_id, size);
            fprintf(f, "    int size_%d = %d;\n", node->output_id, size);
        }
    }
    fprintf(f, "\n");
    
    /* Generate operations */
    for (int i = 0; i < graph->num_nodes; i++) {
        codegen_c_op(f, graph->nodes[i], 1);
    }
    
    fprintf(f, "}\n");
    
    /* Read back string */
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* code = (char*)malloc(len + 1);
    fread(code, 1, len, f);
    code[len] = '\0';
    
    fclose(f);
    return code;
}

/*============================================================================
 * GRAPH PRINTING
 *============================================================================*/

void cg_ir_graph_print(cg_ir_graph* graph) {
    printf("=== IR Graph (%d nodes) ===\n", graph->num_nodes);
    
    for (int i = 0; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        printf("  %3d: %-12s ", node->id, cg_opcode_name(node->opcode));
        
        printf("inputs=[");
        for (int j = 0; j < node->num_inputs; j++) {
            printf("%d", node->input_ids[j]);
            if (j < node->num_inputs - 1) printf(",");
        }
        printf("] -> t%d shape=[", node->output_id);
        
        for (int j = 0; j < node->output_ndim; j++) {
            printf("%d", node->output_shape[j]);
            if (j < node->output_ndim - 1) printf(",");
        }
        printf("]");
        
        if (node->can_fuse) printf(" [FUSED]");
        printf("\n");
    }
    printf("===========================\n");
}

/*============================================================================
 * JIT COMPILER
 *============================================================================*/

cg_jit_compiler* cg_jit_compiler_new(cg_ir_graph* graph, cg_jit_backend backend) {
    cg_jit_compiler* compiler = (cg_jit_compiler*)calloc(1, sizeof(cg_jit_compiler));
    compiler->graph = graph;
    compiler->backend = backend;
    compiler->is_compiled = false;
    return compiler;
}

bool cg_jit_compile(cg_jit_compiler* compiler) {
    if (!compiler || !compiler->graph) return false;
    
    /* Optimize graph */
    cg_ir_optimize(compiler->graph, compiler->graph->opt_level);
    
    /* Generate code for backend */
    switch (compiler->backend) {
        case JIT_BACKEND_CPU:
            compiler->source_code = cg_jit_codegen_c(compiler->graph);
            break;
            
        case JIT_BACKEND_CUDA:
            compiler->source_code = cg_jit_codegen_cuda(compiler->graph);
            break;
            
        default:
            fprintf(stderr, "JIT: Unsupported backend\n");
            return false;
    }
    
    compiler->is_compiled = (compiler->source_code != NULL);
    return compiler->is_compiled;
}

void cg_jit_compiler_free(cg_jit_compiler* compiler) {
    if (!compiler) return;
    free(compiler->source_code);
    /* Note: graph ownership is external */
    free(compiler);
}

/*============================================================================
 * CUDA CODE GENERATION (Stub - requires NVRTC)
 *============================================================================*/

char* cg_jit_codegen_cuda(cg_ir_graph* graph) {
    /* Generate CUDA kernel code */
    size_t buf_size = 8192;
    char* code = (char*)malloc(buf_size);
    int offset = 0;
    
    offset += snprintf(code + offset, buf_size - offset,
        "/* Auto-generated CUDA kernel */\n"
        "extern \"C\" __global__ void jit_kernel(\n");
    
    /* Arguments */
    for (int i = 0; i < graph->num_inputs; i++) {
        offset += snprintf(code + offset, buf_size - offset,
            "    float* __restrict__ t%d, int size_%d",
            graph->input_node_ids[i], graph->input_node_ids[i]);
        if (i < graph->num_inputs - 1 || graph->num_outputs > 0) {
            offset += snprintf(code + offset, buf_size - offset, ",\n");
        }
    }
    for (int i = 0; i < graph->num_outputs; i++) {
        offset += snprintf(code + offset, buf_size - offset,
            "    float* __restrict__ t%d, int size_%d",
            graph->output_node_ids[i], graph->output_node_ids[i]);
        if (i < graph->num_outputs - 1) {
            offset += snprintf(code + offset, buf_size - offset, ",\n");
        }
    }
    
    offset += snprintf(code + offset, buf_size - offset,
        ") {\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\n");
    
    /* Generate fused operations */
    for (int i = 0; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        
        switch (node->opcode) {
            case OP_ADD:
                offset += snprintf(code + offset, buf_size - offset,
                    "    if (idx < size_%d) t%d[idx] = t%d[idx] + t%d[idx];\n",
                    node->output_id, node->output_id, 
                    node->input_ids[0], node->input_ids[1]);
                break;
                
            case OP_MUL:
                offset += snprintf(code + offset, buf_size - offset,
                    "    if (idx < size_%d) t%d[idx] = t%d[idx] * t%d[idx];\n",
                    node->output_id, node->output_id,
                    node->input_ids[0], node->input_ids[1]);
                break;
                
            case OP_RELU:
                offset += snprintf(code + offset, buf_size - offset,
                    "    if (idx < size_%d) t%d[idx] = fmaxf(t%d[idx], 0.0f);\n",
                    node->output_id, node->output_id, node->input_ids[0]);
                break;
                
            case OP_GELU:
                offset += snprintf(code + offset, buf_size - offset,
                    "    if (idx < size_%d) {\n"
                    "        float x = t%d[idx];\n"
                    "        t%d[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));\n"
                    "    }\n",
                    node->output_id, node->input_ids[0], node->output_id);
                break;
                
            default:
                offset += snprintf(code + offset, buf_size - offset,
                    "    /* TODO: %s */\n", cg_opcode_name(node->opcode));
                break;
        }
    }
    
    offset += snprintf(code + offset, buf_size - offset, "}\n");
    
    return code;
}
