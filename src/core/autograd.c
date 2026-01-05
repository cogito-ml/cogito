/**
 * Autograd Implementation
 * 
 * Reverse-mode automatic differentiation.
 * Builds computational graph during forward pass, then backpropagates.
 */

#include "cg_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/*============================================================================
 * TOPOLOGICAL SORT FOR BACKWARD PASS
 *============================================================================*/

/* Node for topological sort */
typedef struct {
    cg_tensor** tensors;
    int count;
    int capacity;
} tensor_list;

static tensor_list* tensor_list_new(int initial_capacity) {
    tensor_list* list = (tensor_list*)malloc(sizeof(tensor_list));
    list->tensors = (cg_tensor**)malloc(initial_capacity * sizeof(cg_tensor*));
    list->count = 0;
    list->capacity = initial_capacity;
    return list;
}

static void tensor_list_add(tensor_list* list, cg_tensor* t) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->tensors = (cg_tensor**)realloc(list->tensors, 
                                              list->capacity * sizeof(cg_tensor*));
    }
    list->tensors[list->count++] = t;
}

static void tensor_list_free(tensor_list* list) {
    if (list) {
        free(list->tensors);
        free(list);
    }
}

static bool tensor_in_list(tensor_list* list, cg_tensor* t) {
    for (int i = 0; i < list->count; i++) {
        if (list->tensors[i] == t) return true;
    }
    return false;
}

/* Build topological order via DFS */
static void build_topo(cg_tensor* t, tensor_list* visited, tensor_list* topo) {
    if (!t || !t->requires_grad || tensor_in_list(visited, t)) {
        return;
    }
    
    tensor_list_add(visited, t);
    
    /* Visit parents first */
    for (int i = 0; i < t->num_parents; i++) {
        if (t->parents[i]) {
            build_topo(t->parents[i], visited, topo);
        }
    }
    
    /* Add self after all parents */
    tensor_list_add(topo, t);
}

/*============================================================================
 * BACKWARD PASS
 *============================================================================*/

void cg_backward(cg_tensor* loss) {
    assert(loss != NULL);
    assert(loss->requires_grad);
    assert(loss->size == 1 && "Loss must be a scalar");
    
    /* Initialize loss gradient to 1.0 */
    if (!loss->grad) {
        loss->grad = (float*)calloc(loss->size, sizeof(float));
    }
    loss->grad[0] = 1.0f;
    
    /* Build topological order */
    tensor_list* visited = tensor_list_new(64);
    tensor_list* topo = tensor_list_new(64);
    build_topo(loss, visited, topo);
    
    /* Traverse in reverse topological order */
    for (int i = topo->count - 1; i >= 0; i--) {
        cg_tensor* t = topo->tensors[i];
        
        if (t->backward_fn) {
            t->backward_fn(t);
        }
    }
    
    tensor_list_free(visited);
    tensor_list_free(topo);
}

/*============================================================================
 * BACKWARD CONTEXTS
 *============================================================================*/

/* Context for binary operations (add, sub, mul, div) */
typedef struct {
    cg_tensor* a;
    cg_tensor* b;
} binary_ctx;

/* Context for matmul */
typedef struct {
    cg_tensor* a;
    cg_tensor* b;
} matmul_ctx;

/* Context for scalar operations */
typedef struct {
    float scalar;
} scalar_ctx;

/*============================================================================
 * BACKWARD FUNCTIONS
 *============================================================================*/

/* Add backward: grad_a = grad_out, grad_b = grad_out */
static void add_backward(cg_tensor* self) {
    binary_ctx* ctx = (binary_ctx*)self->backward_ctx;
    
    if (ctx->a && ctx->a->requires_grad && ctx->a->grad) {
        /* Accumulate gradient to a */
        if (ctx->a->size == self->size) {
            for (int i = 0; i < self->size; i++) {
                ctx->a->grad[i] += self->grad[i];
            }
        } else {
            /* Handle broadcasting: sum over broadcast dimensions */
            for (int i = 0; i < self->size; i++) {
                int remaining = i;
                int a_idx = 0;
                
                for (int d = 0; d < self->ndim; d++) {
                    int coord = remaining / self->strides[d];
                    remaining = remaining % self->strides[d];
                    
                    int a_dim = ctx->a->ndim - self->ndim + d;
                    if (a_dim >= 0 && a_dim < ctx->a->ndim) {
                        int a_coord = (ctx->a->shape[a_dim] == 1) ? 0 : coord;
                        a_idx += a_coord * ctx->a->strides[a_dim];
                    }
                }
                
                ctx->a->grad[a_idx] += self->grad[i];
            }
        }
    }
    
    if (ctx->b && ctx->b->requires_grad && ctx->b->grad) {
        if (ctx->b->size == self->size) {
            for (int i = 0; i < self->size; i++) {
                ctx->b->grad[i] += self->grad[i];
            }
        } else {
            for (int i = 0; i < self->size; i++) {
                int remaining = i;
                int b_idx = 0;
                
                for (int d = 0; d < self->ndim; d++) {
                    int coord = remaining / self->strides[d];
                    remaining = remaining % self->strides[d];
                    
                    int b_dim = ctx->b->ndim - self->ndim + d;
                    if (b_dim >= 0 && b_dim < ctx->b->ndim) {
                        int b_coord = (ctx->b->shape[b_dim] == 1) ? 0 : coord;
                        b_idx += b_coord * ctx->b->strides[b_dim];
                    }
                }
                
                ctx->b->grad[b_idx] += self->grad[i];
            }
        }
    }
}

/* Sub backward: grad_a = grad_out, grad_b = -grad_out */
static void sub_backward(cg_tensor* self) {
    binary_ctx* ctx = (binary_ctx*)self->backward_ctx;
    
    if (ctx->a && ctx->a->requires_grad && ctx->a->grad) {
        for (int i = 0; i < self->size; i++) {
            ctx->a->grad[i] += self->grad[i];
        }
    }
    
    if (ctx->b && ctx->b->requires_grad && ctx->b->grad) {
        for (int i = 0; i < self->size; i++) {
            ctx->b->grad[i] -= self->grad[i];
        }
    }
}

/* Mul backward: grad_a = grad_out * b, grad_b = grad_out * a */
static void mul_backward(cg_tensor* self) {
    binary_ctx* ctx = (binary_ctx*)self->backward_ctx;
    
    if (ctx->a && ctx->a->requires_grad && ctx->a->grad) {
        for (int i = 0; i < self->size; i++) {
            ctx->a->grad[i] += self->grad[i] * ctx->b->data[i];
        }
    }
    
    if (ctx->b && ctx->b->requires_grad && ctx->b->grad) {
        for (int i = 0; i < self->size; i++) {
            ctx->b->grad[i] += self->grad[i] * ctx->a->data[i];
        }
    }
}

/* Matmul backward: 
 * grad_a = grad_out @ b^T
 * grad_b = a^T @ grad_out 
 */
static void matmul_backward(cg_tensor* self) {
    matmul_ctx* ctx = (matmul_ctx*)self->backward_ctx;
    
    int m = ctx->a->shape[0];
    int k = ctx->a->shape[1];
    int n = ctx->b->shape[1];
    
    if (ctx->a && ctx->a->requires_grad && ctx->a->grad) {
        /* grad_a = grad_out @ b^T */
        /* grad_out: [m, n], b^T: [n, k] -> result: [m, k] */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int p = 0; p < n; p++) {
                    sum += self->grad[i * n + p] * ctx->b->data[j * n + p];
                }
                ctx->a->grad[i * k + j] += sum;
            }
        }
    }
    
    if (ctx->b && ctx->b->requires_grad && ctx->b->grad) {
        /* grad_b = a^T @ grad_out */
        /* a^T: [k, m], grad_out: [m, n] -> result: [k, n] */
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < m; p++) {
                    sum += ctx->a->data[p * k + i] * self->grad[p * n + j];
                }
                ctx->b->grad[i * n + j] += sum;
            }
        }
    }
}

/* Sum backward: grad_input = broadcast(grad_out) */
static void sum_backward(cg_tensor* self) {
    cg_tensor* parent = self->parents[0];
    if (!parent || !parent->requires_grad || !parent->grad) return;
    
    /* grad flows back equally to all summed elements */
    for (int i = 0; i < parent->size; i++) {
        parent->grad[i] += self->grad[0];
    }
}

/* Mean backward: grad_input = grad_out / n */
static void mean_backward(cg_tensor* self) {
    cg_tensor* parent = self->parents[0];
    if (!parent || !parent->requires_grad || !parent->grad) return;
    
    float scale = 1.0f / (float)parent->size;
    for (int i = 0; i < parent->size; i++) {
        parent->grad[i] += self->grad[0] * scale;
    }
}

/* Exp backward: grad_input = grad_out * exp(input) = grad_out * output */
static void exp_backward(cg_tensor* self) {
    cg_tensor* parent = self->parents[0];
    if (!parent || !parent->requires_grad || !parent->grad) return;
    
    for (int i = 0; i < self->size; i++) {
        parent->grad[i] += self->grad[i] * self->data[i];
    }
}

/* Log backward: grad_input = grad_out / input */
static void log_backward(cg_tensor* self) {
    cg_tensor* parent = self->parents[0];
    if (!parent || !parent->requires_grad || !parent->grad) return;
    
    for (int i = 0; i < self->size; i++) {
        parent->grad[i] += self->grad[i] / parent->data[i];
    }
}

/* Pow backward: grad_input = grad_out * power * input^(power-1) */
static void pow_backward(cg_tensor* self) {
    cg_tensor* parent = self->parents[0];
    scalar_ctx* ctx = (scalar_ctx*)self->backward_ctx;
    
    if (!parent || !parent->requires_grad || !parent->grad) return;
    
    float power = ctx->scalar;
    for (int i = 0; i < self->size; i++) {
        parent->grad[i] += self->grad[i] * power * powf(parent->data[i], power - 1.0f);
    }
}

/*============================================================================
 * AUTOGRAD-ENABLED OPERATIONS
 * These wrap the basic operations and set up the backward pass.
 *============================================================================*/

cg_tensor* cg_add(cg_tensor* a, cg_tensor* b) {
    assert(a != NULL && b != NULL);
    
    /* Determine output shape (broadcasting) */
    int out_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int out_shape[CG_MAX_DIMS];
    
    for (int i = 0; i < out_ndim; i++) {
        int a_dim = a->ndim - out_ndim + i;
        int b_dim = b->ndim - out_ndim + i;
        
        int a_size = (a_dim >= 0) ? a->shape[a_dim] : 1;
        int b_size = (b_dim >= 0) ? b->shape[b_dim] : 1;
        
        assert(a_size == b_size || a_size == 1 || b_size == 1);
        out_shape[i] = (a_size > b_size) ? a_size : b_size;
    }
    
    bool requires_grad = a->requires_grad || b->requires_grad;
    cg_tensor* out = cg_tensor_new(out_shape, out_ndim, requires_grad);
    if (!out) return NULL;
    
    cg_tensor_add(a, b, out);
    
    if (requires_grad) {
        binary_ctx* ctx = (binary_ctx*)malloc(sizeof(binary_ctx));
        ctx->a = a;
        ctx->b = b;
        
        out->backward_ctx = ctx;
        out->backward_fn = add_backward;
        out->parents[0] = a;
        out->parents[1] = b;
        out->num_parents = 2;
        
        cg_tensor_retain(a);
        cg_tensor_retain(b);
    }
    
    return out;
}

cg_tensor* cg_sub(cg_tensor* a, cg_tensor* b) {
    assert(a != NULL && b != NULL);
    assert(cg_tensor_shape_equal(a, b));
    
    bool requires_grad = a->requires_grad || b->requires_grad;
    cg_tensor* out = cg_tensor_new(a->shape, a->ndim, requires_grad);
    if (!out) return NULL;
    
    cg_tensor_sub(a, b, out);
    
    if (requires_grad) {
        binary_ctx* ctx = (binary_ctx*)malloc(sizeof(binary_ctx));
        ctx->a = a;
        ctx->b = b;
        
        out->backward_ctx = ctx;
        out->backward_fn = sub_backward;
        out->parents[0] = a;
        out->parents[1] = b;
        out->num_parents = 2;
        
        cg_tensor_retain(a);
        cg_tensor_retain(b);
    }
    
    return out;
}

cg_tensor* cg_mul(cg_tensor* a, cg_tensor* b) {
    assert(a != NULL && b != NULL);
    assert(cg_tensor_shape_equal(a, b));
    
    bool requires_grad = a->requires_grad || b->requires_grad;
    cg_tensor* out = cg_tensor_new(a->shape, a->ndim, requires_grad);
    if (!out) return NULL;
    
    cg_tensor_mul(a, b, out);
    
    if (requires_grad) {
        binary_ctx* ctx = (binary_ctx*)malloc(sizeof(binary_ctx));
        ctx->a = a;
        ctx->b = b;
        
        out->backward_ctx = ctx;
        out->backward_fn = mul_backward;
        out->parents[0] = a;
        out->parents[1] = b;
        out->num_parents = 2;
        
        cg_tensor_retain(a);
        cg_tensor_retain(b);
    }
    
    return out;
}

cg_tensor* cg_matmul(cg_tensor* a, cg_tensor* b) {
    assert(a != NULL && b != NULL);
    assert(a->ndim == 2 && b->ndim == 2);
    assert(a->shape[1] == b->shape[0]);
    
    int out_shape[] = {a->shape[0], b->shape[1]};
    bool requires_grad = a->requires_grad || b->requires_grad;
    cg_tensor* out = cg_tensor_new(out_shape, 2, requires_grad);
    if (!out) return NULL;
    
    cg_tensor_matmul(a, b, out);
    
    if (requires_grad) {
        matmul_ctx* ctx = (matmul_ctx*)malloc(sizeof(matmul_ctx));
        ctx->a = a;
        ctx->b = b;
        
        out->backward_ctx = ctx;
        out->backward_fn = matmul_backward;
        out->parents[0] = a;
        out->parents[1] = b;
        out->num_parents = 2;
        
        cg_tensor_retain(a);
        cg_tensor_retain(b);
    }
    
    return out;
}

cg_tensor* cg_sum_all(cg_tensor* a) {
    assert(a != NULL);
    
    int out_shape[] = {1};
    cg_tensor* out = cg_tensor_new(out_shape, 1, a->requires_grad);
    if (!out) return NULL;
    
    cg_tensor_sum(a, -1, out);
    
    if (a->requires_grad) {
        out->backward_fn = sum_backward;
        out->parents[0] = a;
        out->num_parents = 1;
        cg_tensor_retain(a);
    }
    
    return out;
}

cg_tensor* cg_mean_all(cg_tensor* a) {
    assert(a != NULL);
    
    int out_shape[] = {1};
    cg_tensor* out = cg_tensor_new(out_shape, 1, a->requires_grad);
    if (!out) return NULL;
    
    cg_tensor_mean(a, -1, out);
    
    if (a->requires_grad) {
        out->backward_fn = mean_backward;
        out->parents[0] = a;
        out->num_parents = 1;
        cg_tensor_retain(a);
    }
    
    return out;
}

cg_tensor* cg_exp(cg_tensor* a) {
    assert(a != NULL);
    
    cg_tensor* out = cg_tensor_new(a->shape, a->ndim, a->requires_grad);
    if (!out) return NULL;
    
    cg_tensor_exp(a, out);
    
    if (a->requires_grad) {
        out->backward_fn = exp_backward;
        out->parents[0] = a;
        out->num_parents = 1;
        cg_tensor_retain(a);
    }
    
    return out;
}

cg_tensor* cg_log_op(cg_tensor* a) {
    assert(a != NULL);
    
    cg_tensor* out = cg_tensor_new(a->shape, a->ndim, a->requires_grad);
    if (!out) return NULL;
    
    cg_tensor_log(a, out);
    
    if (a->requires_grad) {
        out->backward_fn = log_backward;
        out->parents[0] = a;
        out->num_parents = 1;
        cg_tensor_retain(a);
    }
    
    return out;
}

cg_tensor* cg_pow_op(cg_tensor* a, float power) {
    assert(a != NULL);
    
    cg_tensor* out = cg_tensor_new(a->shape, a->ndim, a->requires_grad);
    if (!out) return NULL;
    
    cg_tensor_pow(a, power, out);
    
    if (a->requires_grad) {
        scalar_ctx* ctx = (scalar_ctx*)malloc(sizeof(scalar_ctx));
        ctx->scalar = power;
        
        out->backward_ctx = ctx;
        out->backward_fn = pow_backward;
        out->parents[0] = a;
        out->num_parents = 1;
        cg_tensor_retain(a);
    }
    
    return out;
}
