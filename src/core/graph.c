/**
 * Computational Graph - Explicit graph-based autograd with Kahn's topological sort
 */

#include "cg_layers.h"
#include <stdlib.h>
#include <string.h>

/*============================================================================
 * GRAPH NODE AND GRAPH STRUCTURES
 *============================================================================*/

typedef struct cg_graph_node {
    cg_tensor* tensor;
    cg_tensor** parents;
    int num_parents;
    int ref_count;
    bool visited;
} cg_graph_node;

typedef struct cg_graph {
    cg_graph_node** nodes;
    int num_nodes;
    int capacity;
} cg_graph;

/* Global graph instance */
static cg_graph* CG_ACTIVE_GRAPH = NULL;

/*============================================================================
 * GRAPH LIFECYCLE
 *============================================================================*/

void cg_graph_init(void) {
    if (CG_ACTIVE_GRAPH) cg_graph_reset();
    CG_ACTIVE_GRAPH = (cg_graph*)calloc(1, sizeof(cg_graph));
    CG_ACTIVE_GRAPH->capacity = 128;
    CG_ACTIVE_GRAPH->nodes = (cg_graph_node**)calloc(128, sizeof(cg_graph_node*));
}

void cg_graph_reset(void) {
    if (!CG_ACTIVE_GRAPH) return;
    
    for (int i = 0; i < CG_ACTIVE_GRAPH->num_nodes; i++) {
        cg_graph_node* node = CG_ACTIVE_GRAPH->nodes[i];
        if (node) {
            free(node->parents);
            free(node);
        }
    }
    
    free(CG_ACTIVE_GRAPH->nodes);
    free(CG_ACTIVE_GRAPH);
    CG_ACTIVE_GRAPH = NULL;
}

bool cg_graph_is_active(void) {
    return CG_ACTIVE_GRAPH != NULL;
}

/*============================================================================
 * GRAPH REGISTRATION
 *============================================================================*/

void cg_graph_register(cg_tensor* t, cg_tensor** parents, int num_parents) {
    if (!CG_ACTIVE_GRAPH || !t || !t->requires_grad) return;
    
    cg_graph_node* node = (cg_graph_node*)calloc(1, sizeof(cg_graph_node));
    node->tensor = t;
    node->num_parents = num_parents;
    node->ref_count = 0;
    node->visited = false;
    
    if (num_parents > 0) {
        node->parents = (cg_tensor**)calloc(num_parents, sizeof(cg_tensor*));
        for (int i = 0; i < num_parents; i++) {
            node->parents[i] = parents[i];
        }
    }
    
    /* Grow array if needed */
    if (CG_ACTIVE_GRAPH->num_nodes >= CG_ACTIVE_GRAPH->capacity) {
        CG_ACTIVE_GRAPH->capacity *= 2;
        CG_ACTIVE_GRAPH->nodes = (cg_graph_node**)realloc(
            CG_ACTIVE_GRAPH->nodes, 
            CG_ACTIVE_GRAPH->capacity * sizeof(cg_graph_node*));
    }
    
    CG_ACTIVE_GRAPH->nodes[CG_ACTIVE_GRAPH->num_nodes++] = node;
}

/*============================================================================
 * TOPOLOGICAL SORT (KAHN'S ALGORITHM)
 *============================================================================*/

static int find_node_index(cg_graph* g, cg_tensor* tensor) {
    for (int i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i]->tensor == tensor) return i;
    }
    return -1;
}

static cg_graph_node** cg_graph_topo_sort(int* sorted_size) {
    cg_graph* g = CG_ACTIVE_GRAPH;
    if (!g || g->num_nodes == 0) {
        *sorted_size = 0;
        return NULL;
    }
    
    int* in_degree = (int*)calloc(g->num_nodes, sizeof(int));
    cg_graph_node** sorted = (cg_graph_node**)calloc(g->num_nodes, sizeof(cg_graph_node*));
    
    /* Compute in-degrees: how many children point to this node */
    for (int i = 0; i < g->num_nodes; i++) {
        cg_graph_node* node = g->nodes[i];
        for (int j = 0; j < node->num_parents; j++) {
            int parent_idx = find_node_index(g, node->parents[j]);
            if (parent_idx >= 0) {
                in_degree[parent_idx]++;
            }
        }
    }
    
    /* Queue for nodes with 0 in-degree (leaf nodes in reverse order) */
    int* queue = (int*)calloc(g->num_nodes, sizeof(int));
    int front = 0, rear = 0;
    
    for (int i = 0; i < g->num_nodes; i++) {
        if (in_degree[i] == 0) {
            queue[rear++] = i;
        }
    }
    
    *sorted_size = 0;
    while (front < rear) {
        int idx = queue[front++];
        sorted[(*sorted_size)++] = g->nodes[idx];
        
        /* Decrement in-degree of parents */
        cg_graph_node* node = g->nodes[idx];
        for (int j = 0; j < node->num_parents; j++) {
            int parent_idx = find_node_index(g, node->parents[j]);
            if (parent_idx >= 0) {
                in_degree[parent_idx]--;
                if (in_degree[parent_idx] == 0) {
                    queue[rear++] = parent_idx;
                }
            }
        }
    }
    
    free(in_degree);
    free(queue);
    
    return sorted;
}

/*============================================================================
 * BACKWARD PASS (ITERATIVE)
 *============================================================================*/

void cg_graph_backward(cg_tensor* loss) {
    if (!CG_ACTIVE_GRAPH || !loss) return;
    
    /* Initialize loss gradient to 1.0 */
    if (!loss->grad) {
        loss->grad = (float*)calloc(loss->size, sizeof(float));
    }
    for (int i = 0; i < loss->size; i++) {
        loss->grad[i] = 1.0f;
    }
    
    /* Get topologically sorted nodes */
    int sorted_size;
    cg_graph_node** sorted = cg_graph_topo_sort(&sorted_size);
    
    if (!sorted) return;
    
    /* Iterate in forward topological order (which is reverse for backprop) */
    /* The sort gives us nodes from leaves to root, we want root to leaves */
    for (int i = sorted_size - 1; i >= 0; i--) {
        cg_graph_node* node = sorted[i];
        if (node->tensor->backward_fn && node->tensor->grad) {
            node->tensor->backward_fn(node->tensor);
        }
    }
    
    free(sorted);
}

/*============================================================================
 * GRAPH-AWARE TENSOR OPERATIONS
 *============================================================================*/

/* Binary operation context */
typedef struct {
    cg_tensor* a;
    cg_tensor* b;
} binary_op_ctx;

/* Backward for add */
static void graph_add_backward(cg_tensor* out) {
    binary_op_ctx* ctx = (binary_op_ctx*)out->backward_ctx;
    if (!ctx) return;
    
    /* grad_a += grad_out, grad_b += grad_out */
    if (ctx->a->requires_grad && ctx->a->grad) {
        for (int i = 0; i < out->size; i++) {
            ctx->a->grad[i] += out->grad[i];
        }
    }
    if (ctx->b->requires_grad && ctx->b->grad) {
        for (int i = 0; i < out->size; i++) {
            ctx->b->grad[i] += out->grad[i];
        }
    }
}

/* Backward for mul */
static void graph_mul_backward(cg_tensor* out) {
    binary_op_ctx* ctx = (binary_op_ctx*)out->backward_ctx;
    if (!ctx) return;
    
    /* grad_a += grad_out * b, grad_b += grad_out * a */
    if (ctx->a->requires_grad && ctx->a->grad) {
        for (int i = 0; i < out->size; i++) {
            ctx->a->grad[i] += out->grad[i] * ctx->b->data[i];
        }
    }
    if (ctx->b->requires_grad && ctx->b->grad) {
        for (int i = 0; i < out->size; i++) {
            ctx->b->grad[i] += out->grad[i] * ctx->a->data[i];
        }
    }
}

/* Graph-aware add */
cg_tensor* cg_graph_add(cg_tensor* a, cg_tensor* b) {
    cg_tensor* out = cg_tensor_new(a->shape, a->ndim, a->requires_grad || b->requires_grad);
    cg_tensor_add(a, b, out);
    
    if (CG_ACTIVE_GRAPH && out->requires_grad) {
        binary_op_ctx* ctx = (binary_op_ctx*)malloc(sizeof(binary_op_ctx));
        ctx->a = a;
        ctx->b = b;
        out->backward_ctx = ctx;
        out->backward_fn = graph_add_backward;
        
        cg_tensor* parents[] = {a, b};
        cg_graph_register(out, parents, 2);
    }
    
    return out;
}

/* Graph-aware mul */
cg_tensor* cg_graph_mul(cg_tensor* a, cg_tensor* b) {
    cg_tensor* out = cg_tensor_new(a->shape, a->ndim, a->requires_grad || b->requires_grad);
    cg_tensor_mul(a, b, out);
    
    if (CG_ACTIVE_GRAPH && out->requires_grad) {
        binary_op_ctx* ctx = (binary_op_ctx*)malloc(sizeof(binary_op_ctx));
        ctx->a = a;
        ctx->b = b;
        out->backward_ctx = ctx;
        out->backward_fn = graph_mul_backward;
        
        cg_tensor* parents[] = {a, b};
        cg_graph_register(out, parents, 2);
    }
    
    return out;
}

/*============================================================================
 * MATMUL WITH GRAPH
 *============================================================================*/

typedef struct {
    cg_tensor* a;  /* (M, K) */
    cg_tensor* b;  /* (K, N) */
} matmul_ctx;

static void graph_matmul_backward(cg_tensor* out) {
    matmul_ctx* ctx = (matmul_ctx*)out->backward_ctx;
    if (!ctx) return;
    
    int M = ctx->a->shape[0];
    int K = ctx->a->shape[1];
    int N = ctx->b->shape[1];
    
    /* grad_a = grad_out @ b^T */
    if (ctx->a->requires_grad && ctx->a->grad) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                float sum = 0.0f;
                for (int n = 0; n < N; n++) {
                    sum += out->grad[i * N + n] * ctx->b->data[j * N + n];
                }
                ctx->a->grad[i * K + j] += sum;
            }
        }
    }
    
    /* grad_b = a^T @ grad_out */
    if (ctx->b->requires_grad && ctx->b->grad) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int m = 0; m < M; m++) {
                    sum += ctx->a->data[m * K + i] * out->grad[m * N + j];
                }
                ctx->b->grad[i * N + j] += sum;
            }
        }
    }
}

cg_tensor* cg_graph_matmul(cg_tensor* a, cg_tensor* b) {
    int out_shape[] = {a->shape[0], b->shape[1]};
    cg_tensor* out = cg_tensor_new(out_shape, 2, a->requires_grad || b->requires_grad);
    cg_tensor_matmul(a, b, out);
    
    if (CG_ACTIVE_GRAPH && out->requires_grad) {
        matmul_ctx* ctx = (matmul_ctx*)malloc(sizeof(matmul_ctx));
        ctx->a = a;
        ctx->b = b;
        out->backward_ctx = ctx;
        out->backward_fn = graph_matmul_backward;
        
        cg_tensor* parents[] = {a, b};
        cg_graph_register(out, parents, 2);
    }
    
    return out;
}
