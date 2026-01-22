/**
 * Kernel Fusion Engine
 */

#include "cg_fusion.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*============================================================================
 * FUSION GROUP
 *============================================================================*/

cg_fusion_group* cg_fusion_group_new(cg_ir_node** nodes, int count) {
    cg_fusion_group* group = (cg_fusion_group*)calloc(1, sizeof(cg_fusion_group));
    group->capacity = (count > 8) ? count * 2 : 16;
    group->nodes = (cg_ir_node**)malloc(group->capacity * sizeof(cg_ir_node*));
    group->num_nodes = 0;
    
    for (int i = 0; i < count; i++) {
        cg_fusion_group_add(group, nodes[i]);
    }
    
    return group;
}

void cg_fusion_group_add(cg_fusion_group* group, cg_ir_node* node) {
    if (group->num_nodes >= group->capacity) {
        group->capacity *= 2;
        group->nodes = (cg_ir_node**)realloc(group->nodes, 
                                              group->capacity * sizeof(cg_ir_node*));
    }
    group->nodes[group->num_nodes++] = node;
}

void cg_fusion_group_free(cg_fusion_group* group) {
    if (!group) return;
    free(group->nodes);
    free(group->kernel_name);
    free(group->kernel_code);
    free(group->output_shape);
    free(group);
}

/*============================================================================
 * FUSION PASS
 *============================================================================*/

cg_fusion_pass* cg_fusion_pass_new(cg_ir_graph* graph) {
    cg_fusion_pass* pass = (cg_fusion_pass*)calloc(1, sizeof(cg_fusion_pass));
    pass->graph = graph;
    pass->max_groups = 64;
    pass->groups = (cg_fusion_group**)calloc(pass->max_groups, sizeof(cg_fusion_group*));
    pass->max_fusion_depth = 8;
    pass->enable_epilogue = true;
    pass->enable_attention = true;
    return pass;
}

void cg_fusion_pass_free(cg_fusion_pass* pass) {
    if (!pass) return;
    for (int i = 0; i < pass->num_groups; i++) {
        cg_fusion_group_free(pass->groups[i]);
    }
    free(pass->groups);
    free(pass);
}

/*============================================================================
 * FUSION PREDICATES
 *============================================================================*/

static bool is_elementwise(cg_ir_opcode op) {
    switch (op) {
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_NEG: case OP_EXP: case OP_LOG: case OP_SQRT: case OP_RSQRT:
        case OP_RELU: case OP_SIGMOID: case OP_TANH: case OP_GELU:
            return true;
        default:
            return false;
    }
}

static bool is_reduction(cg_ir_opcode op) {
    switch (op) {
        case OP_SUM: case OP_MEAN: case OP_MAX: case OP_MIN:
            return true;
        default:
            return false;
    }
}

bool cg_fusion_can_fuse(cg_ir_node* a, cg_ir_node* b) {
    /* Elementwise + Elementwise: always fusable */
    if (is_elementwise(a->opcode) && is_elementwise(b->opcode)) {
        return true;
    }
    
    /* Elementwise -> Reduction: fusable */
    if (is_elementwise(a->opcode) && is_reduction(b->opcode)) {
        return true;
    }
    
    /* MATMUL -> (BIAS ->) ACTIVATION: epilogue fusion */
    if (a->opcode == OP_MATMUL && 
        (b->opcode == OP_ADD || is_elementwise(b->opcode))) {
        return true;
    }
    
    return false;
}

/*============================================================================
 * ADVANCED PATTERN MATCHERS
 *============================================================================*/

static bool match_layernorm_gelu_linear(cg_ir_graph* graph, int node_idx, int* matched_nodes) {
    if (node_idx + 2 >= graph->num_nodes) return false;

    /* Check pattern: LN -> GeLU -> Linear */
    cg_ir_node* ln = graph->nodes[node_idx];
    if (ln->opcode != OP_FUSED_LAYERNORM) return false;
    
    /* Find GeLU user of LN */
    cg_ir_node* gelu = NULL;
    for (int i = node_idx + 1; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        if (node->opcode == OP_GELU && node->num_inputs > 0 && node->input_ids[0] == ln->output_id) {
            gelu = node;
            break;
        }
    }
    if (!gelu) return false;
    
    /* Find Linear (Matmul) user of GeLU */
    cg_ir_node* linear = NULL;
    for (int i = gelu->id + 1; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        if (node->opcode == OP_MATMUL && node->num_inputs > 0 && node->input_ids[0] == gelu->output_id) {
            linear = node;
            break;
        }
    }
    if (!linear) return false;
    
    matched_nodes[0] = ln->id;
    matched_nodes[1] = gelu->id;
    matched_nodes[2] = linear->id;
    return true;
}

static bool match_softmax_dropout_residual(cg_ir_graph* graph, int node_idx, int* matched_nodes) {
    if (node_idx + 2 >= graph->num_nodes) return false;
    
    /* Check pattern: Softmax -> Dropout (Mul) -> Residual (Add) */
    cg_ir_node* sm = graph->nodes[node_idx];
    if (sm->opcode != OP_SOFTMAX) return false;
    
    /* Find Dropout (Mul) user of Softmax */
    cg_ir_node* drop = NULL;
    for (int i = node_idx + 1; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        if (node->opcode == OP_MUL && node->num_inputs > 0 && node->input_ids[0] == sm->output_id) {
            drop = node;
            break;
        }
    }
    if (!drop) return false;
    
    /* Find Residual (Add) user of Dropout */
    cg_ir_node* res = NULL;
    for (int i = drop->id + 1; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        if (node->opcode == OP_ADD && node->num_inputs > 0) {
            for (int k = 0; k < node->num_inputs; k++) {
                if (node->input_ids[k] == drop->output_id) {
                    res = node;
                    break;
                }
            }
        }
        if (res) break;
    }
    if (!res) return false;
    
    matched_nodes[0] = sm->id;
    matched_nodes[1] = drop->id;
    matched_nodes[2] = res->id;
    return true;
}

static bool match_qkv_rope(cg_ir_graph* graph, int node_idx, int* matched_nodes) {
    /* Pattern: Matmul (QKV proj) -> Reshape -> RoPE */
    cg_ir_node* proj = graph->nodes[node_idx];
    if (proj->opcode != OP_MATMUL) return false;
    
    /* Find Reshape */
    cg_ir_node* reshape = NULL;
    for (int i = node_idx + 1; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        if (node->opcode == OP_RESHAPE && node->num_inputs > 0 && node->input_ids[0] == proj->output_id) {
            reshape = node;
            break;
        }
    }
    if (!reshape) return false;
    
    /* Find RoPE (simulated as Custom op logic) */
    cg_ir_node* rope = NULL;
    for (int i = reshape->id + 1; i < graph->num_nodes; i++) {
        cg_ir_node* node = graph->nodes[i];
        /* Often OP_CUSTOM for RoPE */
        if (node->opcode == OP_CUSTOM && node->num_inputs > 0 && node->input_ids[0] == reshape->output_id) {
            rope = node;
            break;
        }
    }
    if (!rope) return false;
    
    matched_nodes[0] = proj->id;
    matched_nodes[1] = reshape->id;
    matched_nodes[2] = rope->id;
    return true;
}

/*============================================================================
 * PATTERN MATCHING
 *============================================================================*/

bool cg_fusion_match_matmul_epilogue(cg_ir_graph* graph, int start_idx,
                                     cg_fusion_group** out_group) {
    if (start_idx >= graph->num_nodes) return false;
    
    cg_ir_node* matmul = graph->nodes[start_idx];
    if (matmul->opcode != OP_MATMUL) return false;
    
    /* Look for bias add */
    cg_ir_node* bias = NULL;
    cg_ir_node* activation = NULL;
    
    for (int i = start_idx + 1; i < graph->num_nodes && i <= start_idx + 3; i++) {
        cg_ir_node* node = graph->nodes[i];
        
        /* Check if this node uses matmul output */
        bool uses_prev = false;
        for (int j = 0; j < node->num_inputs; j++) {
            if (bias && node->input_ids[j] == bias->output_id) uses_prev = true;
            if (!bias && node->input_ids[j] == matmul->output_id) uses_prev = true;
        }
        if (!uses_prev) continue;
        
        if (node->opcode == OP_ADD && !bias) {
            bias = node;
        } else if (is_elementwise(node->opcode)) {
            activation = node;
            break;
        }
    }
    
    if (!bias && !activation) return false;
    
    /* Create fusion group */
    cg_fusion_group* group = (cg_fusion_group*)calloc(1, sizeof(cg_fusion_group));
    group->fusion_type = FUSION_EPILOGUE;
    group->nodes = (cg_ir_node**)malloc(3 * sizeof(cg_ir_node*));
    group->capacity = 3;
    
    group->nodes[group->num_nodes++] = matmul;
    if (bias) group->nodes[group->num_nodes++] = bias;
    if (activation) group->nodes[group->num_nodes++] = activation;
    
    group->kernel_name = strdup("fused_matmul_epilogue");
    group->is_valid = true;
    
    *out_group = group;
    return true;
}

bool cg_fusion_match_relu_dropout_add(cg_ir_graph* graph, int start_idx,
                                      cg_fusion_group** out_group) {
    if (start_idx + 2 >= graph->num_nodes) return false;
    
    cg_ir_node* relu = graph->nodes[start_idx];
    if (relu->opcode != OP_RELU) return false;
    
    /* Note: We don't have OP_DROPOUT yet, so check for pattern with what we have */
    cg_ir_node* next1 = graph->nodes[start_idx + 1];
    cg_ir_node* next2 = graph->nodes[start_idx + 2];
    
    /* Look for relu -> elementwise -> add pattern */
    if (is_elementwise(next1->opcode) && next2->opcode == OP_ADD) {
        bool chain = false;
        for (int j = 0; j < next1->num_inputs; j++) {
            if (next1->input_ids[j] == relu->output_id) chain = true;
        }
        if (!chain) return false;
        
        for (int j = 0; j < next2->num_inputs; j++) {
            if (next2->input_ids[j] == next1->output_id) chain = chain && true;
        }
        if (!chain) return false;
        
        cg_fusion_group* group = (cg_fusion_group*)calloc(1, sizeof(cg_fusion_group));
        group->fusion_type = FUSION_ELEMENTWISE;
        group->nodes = (cg_ir_node**)malloc(3 * sizeof(cg_ir_node*));
        group->capacity = 3;
        group->nodes[group->num_nodes++] = relu;
        group->nodes[group->num_nodes++] = next1;
        group->nodes[group->num_nodes++] = next2;
        group->kernel_name = strdup("fused_relu_chain");
        group->is_valid = true;
        
        *out_group = group;
        return true;
    }
    
    return false;
}

/*============================================================================
 * FUSION ANALYSIS
 *============================================================================*/

cg_fusion_group** cg_fusion_find_patterns(cg_ir_graph* graph, int* num_groups) {
    cg_fusion_group** groups = (cg_fusion_group**)malloc(64 * sizeof(cg_fusion_group*));
    int count = 0;
    
    bool* fused = (bool*)calloc(graph->num_nodes, sizeof(bool));
    
    for (int i = 0; i < graph->num_nodes; i++) {
        if (fused[i]) continue;
        
        cg_fusion_group* group = NULL;
        int matched_nodes[8]; /* Temporary storage for node IDs */
        
        /* 1. Try complex Transformer patterns first */
        if (match_layernorm_gelu_linear(graph, i, matched_nodes)) {
            cg_ir_node* nodes[3];
            for (int k = 0; k < 3; k++) nodes[k] = graph->nodes[matched_nodes[k]];
            
            groups[count++] = cg_fusion_group_new(nodes, 3);
            groups[count-1]->fusion_type = FUSION_CUSTOM;
            groups[count-1]->kernel_name = strdup("fused_ln_gelu_linear");
            
            for (int k = 0; k < 3; k++) fused[matched_nodes[k]] = true;
            continue;
        }
        
        if (match_softmax_dropout_residual(graph, i, matched_nodes)) {
            cg_ir_node* nodes[3];
            for (int k = 0; k < 3; k++) nodes[k] = graph->nodes[matched_nodes[k]];
            
            groups[count++] = cg_fusion_group_new(nodes, 3);
            groups[count-1]->fusion_type = FUSION_CUSTOM;
            groups[count-1]->kernel_name = strdup("fused_softmax_dropout_res");
            
            for (int k = 0; k < 3; k++) fused[matched_nodes[k]] = true;
            continue;
        }
        
        if (match_qkv_rope(graph, i, matched_nodes)) {
            cg_ir_node* nodes[3];
            for (int k = 0; k < 3; k++) nodes[k] = graph->nodes[matched_nodes[k]];
            
            groups[count++] = cg_fusion_group_new(nodes, 3);
            groups[count-1]->fusion_type = FUSION_CUSTOM;
            groups[count-1]->kernel_name = strdup("fused_qkv_rope");
            
            for (int k = 0; k < 3; k++) fused[matched_nodes[k]] = true;
            continue;
        }

        /* 2. Try matmul epilogue fusion */
        if (cg_fusion_match_matmul_epilogue(graph, i, &group)) {
            groups[count++] = group;
            for (int j = 0; j < group->num_nodes; j++) {
                fused[group->nodes[j]->id] = true;
            }
            continue;
        }
        
        /* Try elementwise chain fusion */
        if (cg_fusion_match_relu_dropout_add(graph, i, &group)) {
            groups[count++] = group;
            for (int j = 0; j < group->num_nodes; j++) {
                fused[group->nodes[j]->id] = true;
            }
            continue;
        }
        
        /* Simple pairwise elementwise fusion */
        if (is_elementwise(graph->nodes[i]->opcode) && i + 1 < graph->num_nodes) {
            cg_ir_node* next = graph->nodes[i + 1];
            if (is_elementwise(next->opcode)) {
                bool connected = false;
                for (int j = 0; j < next->num_inputs; j++) {
                    if (next->input_ids[j] == graph->nodes[i]->output_id) {
                        connected = true;
                    }
                }
                
                if (connected && !fused[i + 1]) {
                    cg_ir_node* pair[2] = {graph->nodes[i], next};
                    groups[count++] = cg_fusion_group_new(pair, 2);
                    groups[count-1]->fusion_type = FUSION_ELEMENTWISE;
                    groups[count-1]->kernel_name = strdup("fused_elementwise");
                    fused[i] = true;
                    fused[i + 1] = true;
                }
            }
        }
    }
    
    free(fused);
    *num_groups = count;
    return groups;
}

void cg_fusion_analyze(cg_fusion_pass* pass) {
    int count;
    pass->groups = cg_fusion_find_patterns(pass->graph, &count);
    pass->num_groups = count;
    
    /* Compute memory savings */
    for (int i = 0; i < pass->num_groups; i++) {
        cg_fusion_group* group = pass->groups[i];
        size_t reads = 0, writes = 0;
        
        for (int j = 0; j < group->num_nodes; j++) {
            cg_ir_node* node = group->nodes[j];
            int size = 1;
            for (int d = 0; d < node->output_ndim; d++) {
                size *= node->output_shape[d];
            }
            writes += size * sizeof(float);
            reads += node->num_inputs * size * sizeof(float);
        }
        
        /* Fused kernel: only one read and one write */
        cg_ir_node* first = group->nodes[0];
        cg_ir_node* last = group->nodes[group->num_nodes - 1];
        
        int fused_size = 1;
        for (int d = 0; d < last->output_ndim; d++) {
            fused_size *= last->output_shape[d];
        }
        
        size_t fused_io = first->num_inputs * fused_size * sizeof(float) + 
                          fused_size * sizeof(float);
        
        group->memory_saved = (reads + writes) - fused_io;
    }
}

/*============================================================================
 * KERNEL GENERATION
 *============================================================================*/

static const char* op_to_expr(cg_ir_opcode op, const char* input) {
    static char buf[256];
    switch (op) {
        case OP_RELU:
            snprintf(buf, sizeof(buf), "fmaxf(0.0f, %s)", input);
            break;
        case OP_SIGMOID:
            snprintf(buf, sizeof(buf), "(1.0f / (1.0f + expf(-%s)))", input);
            break;
        case OP_TANH:
            snprintf(buf, sizeof(buf), "tanhf(%s)", input);
            break;
        case OP_GELU:
            snprintf(buf, sizeof(buf), 
                "(0.5f * %s * (1.0f + tanhf(0.7978845608f * (%s + 0.044715f * %s * %s * %s))))",
                input, input, input, input, input);
            break;
        case OP_EXP:
            snprintf(buf, sizeof(buf), "expf(%s)", input);
            break;
        case OP_LOG:
            snprintf(buf, sizeof(buf), "logf(%s)", input);
            break;
        case OP_SQRT:
            snprintf(buf, sizeof(buf), "sqrtf(%s)", input);
            break;
        case OP_RSQRT:
            snprintf(buf, sizeof(buf), "rsqrtf(%s)", input);
            break;
        default:
            snprintf(buf, sizeof(buf), "%s", input);
            break;
    }
    return buf;
}

char* cg_fusion_generate_kernel(cg_fusion_group* group) {
    char* code = (char*)malloc(4096);
    int offset = 0;
    
    /* Kernel header */
    offset += snprintf(code + offset, 4096 - offset,
        "/* Fused kernel: %s */\n"
        "__global__ void %s(float* input, float* output, int size) {\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (idx >= size) return;\n\n"
        "    float x = input[idx];\n",
        group->kernel_name, group->kernel_name);
    
    /* Generate fused operations */
    for (int i = 0; i < group->num_nodes; i++) {
        cg_ir_node* node = group->nodes[i];
        
        if (is_elementwise(node->opcode)) {
            const char* expr = op_to_expr(node->opcode, "x");
            offset += snprintf(code + offset, 4096 - offset,
                "    x = %s;\n", expr);
        } else if (node->opcode == OP_ADD) {
            offset += snprintf(code + offset, 4096 - offset,
                "    x = x + bias[idx %% bias_size];\n");
        }
    }
    
    /* Store result */
    offset += snprintf(code + offset, 4096 - offset,
        "\n    output[idx] = x;\n"
        "}\n");
    
    group->kernel_code = code;
    return code;
}

/*============================================================================
 * FUSION APPLICATION
 *============================================================================*/

void cg_fusion_apply(cg_fusion_pass* pass) {
    /* For each fusion group, replace the subgraph with a fused op */
    for (int g = 0; g < pass->num_groups; g++) {
        cg_fusion_group* group = pass->groups[g];
        
        if (!group->is_valid || group->num_nodes < 2) continue;
        
        /* Generate kernel code */
        cg_fusion_generate_kernel(group);
        
        /* Create fused IR node */
        cg_ir_node* fused = cg_ir_node_new(OP_CUSTOM);
        fused->id = group->nodes[0]->id;
        
        /* Copy first node's inputs */
        cg_ir_node* first = group->nodes[0];
        fused->num_inputs = first->num_inputs;
        fused->input_ids = (int*)malloc(first->num_inputs * sizeof(int));
        memcpy(fused->input_ids, first->input_ids, first->num_inputs * sizeof(int));
        
        /* Use last node's output */
        cg_ir_node* last = group->nodes[group->num_nodes - 1];
        fused->output_id = last->output_id;
        fused->output_ndim = last->output_ndim;
        fused->output_shape = (int*)malloc(last->output_ndim * sizeof(int));
        memcpy(fused->output_shape, last->output_shape, last->output_ndim * sizeof(int));
        
        /* Store kernel code as attribute */
        fused->attributes = strdup(group->kernel_code);
        fused->attr_size = strlen(group->kernel_code) + 1;
        
        /* Replace first node in graph */
        pass->graph->nodes[group->nodes[0]->id] = fused;
        
        /* Mark other nodes as dead */
        for (int i = 1; i < group->num_nodes; i++) {
            pass->graph->nodes[group->nodes[i]->id]->opcode = OP_NOOP;
        }
    }
    
    /* Run DCE to remove dead nodes */
    cg_ir_dce(pass->graph);
}
