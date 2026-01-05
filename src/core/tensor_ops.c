/**
 * Tensor Operations Implementation
 * 
 * Element-wise ops, reductions, and matrix multiplication.
 * Optional BLAS acceleration for matmul.
 */

#include "cg_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef CG_USE_BLAS
#include <cblas.h>
#endif

/*============================================================================
 * ELEMENT-WISE OPERATIONS
 *============================================================================*/

/* Helper: broadcast index computation */
static int broadcast_index(int flat_idx, cg_tensor* t, cg_tensor* ref) {
    /* Convert flat index in reference tensor to index in tensor t */
    /* Handles broadcasting: dimensions of size 1 are broadcast */
    
    if (t->size == ref->size) {
        return flat_idx;  /* Same shape, no broadcast needed */
    }
    
    int result_idx = 0;
    int remaining = flat_idx;
    
    for (int i = 0; i < ref->ndim; i++) {
        int coord = remaining / ref->strides[i];
        remaining = remaining % ref->strides[i];
        
        /* Map coordinate to tensor t */
        int t_dim = t->ndim - ref->ndim + i;
        if (t_dim >= 0 && t_dim < t->ndim) {
            int t_coord = (t->shape[t_dim] == 1) ? 0 : coord;
            result_idx += t_coord * t->strides[t_dim];
        }
    }
    
    return result_idx;
}

void cg_tensor_add(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    assert(a != NULL && b != NULL && out != NULL);
    
    /* Simple case: same shape */
    if (a->size == b->size && a->size == out->size) {
        for (int i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] + b->data[i];
        }
    } else {
        /* Broadcasting case */
        for (int i = 0; i < out->size; i++) {
            int a_idx = broadcast_index(i, a, out);
            int b_idx = broadcast_index(i, b, out);
            out->data[i] = a->data[a_idx] + b->data[b_idx];
        }
    }
}

void cg_tensor_sub(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    assert(a != NULL && b != NULL && out != NULL);
    
    if (a->size == b->size && a->size == out->size) {
        for (int i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] - b->data[i];
        }
    } else {
        for (int i = 0; i < out->size; i++) {
            int a_idx = broadcast_index(i, a, out);
            int b_idx = broadcast_index(i, b, out);
            out->data[i] = a->data[a_idx] - b->data[b_idx];
        }
    }
}

void cg_tensor_mul(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    assert(a != NULL && b != NULL && out != NULL);
    
    if (a->size == b->size && a->size == out->size) {
        for (int i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] * b->data[i];
        }
    } else {
        for (int i = 0; i < out->size; i++) {
            int a_idx = broadcast_index(i, a, out);
            int b_idx = broadcast_index(i, b, out);
            out->data[i] = a->data[a_idx] * b->data[b_idx];
        }
    }
}

void cg_tensor_div(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    assert(a != NULL && b != NULL && out != NULL);
    
    if (a->size == b->size && a->size == out->size) {
        for (int i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] / b->data[i];
        }
    } else {
        for (int i = 0; i < out->size; i++) {
            int a_idx = broadcast_index(i, a, out);
            int b_idx = broadcast_index(i, b, out);
            out->data[i] = a->data[a_idx] / b->data[b_idx];
        }
    }
}

void cg_tensor_scale(cg_tensor* a, float scalar, cg_tensor* out) {
    assert(a != NULL && out != NULL);
    assert(a->size == out->size);
    
    for (int i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * scalar;
    }
}

/*============================================================================
 * MATRIX MULTIPLICATION
 *============================================================================*/

/* Naive matmul implementation */
static void matmul_naive(float* a, float* b, float* c,
                         int m, int n, int k) {
    /* c[m, n] = a[m, k] @ b[k, n] */
    memset(c, 0, m * n * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/* Blocked matmul for better cache utilization */
#define BLOCK_SIZE 32

static void matmul_blocked(float* a, float* b, float* c,
                           int m, int n, int k) {
    memset(c, 0, m * n * sizeof(float));
    
    for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int pp = 0; pp < k; pp += BLOCK_SIZE) {
                /* Block limits */
                int i_max = (ii + BLOCK_SIZE < m) ? ii + BLOCK_SIZE : m;
                int j_max = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;
                int p_max = (pp + BLOCK_SIZE < k) ? pp + BLOCK_SIZE : k;
                
                /* Multiply blocks */
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        float sum = c[i * n + j];
                        for (int p = pp; p < p_max; p++) {
                            sum += a[i * k + p] * b[p * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

void cg_tensor_matmul(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    assert(a != NULL && b != NULL && out != NULL);
    assert(a->ndim >= 2 && b->ndim >= 2);
    assert(a->shape[a->ndim - 1] == b->shape[b->ndim - 2]);
    
    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];
    
    assert(out->shape[out->ndim - 2] == m);
    assert(out->shape[out->ndim - 1] == n);
    
#ifdef CG_USE_BLAS
    /* Use BLAS for matmul when available */
    /* BLAS expects column-major, but we're row-major */
    /* So we compute C^T = B^T @ A^T which gives us C in row-major */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0f, a->data, k,
                b->data, n,
                0.0f, out->data, n);
#else
    /* Use blocked matmul for better cache performance */
    if (m * n * k > 1000000) {
        matmul_blocked(a->data, b->data, out->data, m, n, k);
    } else {
        matmul_naive(a->data, b->data, out->data, m, n, k);
    }
#endif
}

void cg_tensor_transpose(cg_tensor* a, cg_tensor* out) {
    assert(a != NULL && out != NULL);
    assert(a->ndim == 2 && out->ndim == 2);
    assert(a->shape[0] == out->shape[1] && a->shape[1] == out->shape[0]);
    
    int rows = a->shape[0];
    int cols = a->shape[1];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[j * rows + i] = a->data[i * cols + j];
        }
    }
}

/*============================================================================
 * REDUCTION OPERATIONS
 *============================================================================*/

void cg_tensor_sum(cg_tensor* a, int axis, cg_tensor* out) {
    assert(a != NULL && out != NULL);
    
    if (axis < 0) {
        /* Sum all elements */
        float sum = 0.0f;
        for (int i = 0; i < a->size; i++) {
            sum += a->data[i];
        }
        out->data[0] = sum;
        return;
    }
    
    assert(axis < a->ndim);
    
    /* Initialize output to zero */
    memset(out->data, 0, out->size * sizeof(float));
    
    /* Iterate through all elements of input */
    for (int i = 0; i < a->size; i++) {
        /* Convert flat index to multi-dimensional index */
        int remaining = i;
        int out_idx = 0;
        int out_dim = 0;
        
        for (int d = 0; d < a->ndim; d++) {
            int coord = remaining / a->strides[d];
            remaining = remaining % a->strides[d];
            
            if (d != axis) {
                out_idx += coord * out->strides[out_dim];
                out_dim++;
            }
        }
        
        out->data[out_idx] += a->data[i];
    }
}

void cg_tensor_mean(cg_tensor* a, int axis, cg_tensor* out) {
    cg_tensor_sum(a, axis, out);
    
    float divisor;
    if (axis < 0) {
        divisor = (float)a->size;
    } else {
        divisor = (float)a->shape[axis];
    }
    
    for (int i = 0; i < out->size; i++) {
        out->data[i] /= divisor;
    }
}

void cg_tensor_max(cg_tensor* a, int axis, cg_tensor* out) {
    assert(a != NULL && out != NULL);
    
    if (axis < 0) {
        /* Max of all elements */
        float max_val = a->data[0];
        for (int i = 1; i < a->size; i++) {
            if (a->data[i] > max_val) max_val = a->data[i];
        }
        out->data[0] = max_val;
        return;
    }
    
    assert(axis < a->ndim);
    
    /* Initialize output to -infinity */
    for (int i = 0; i < out->size; i++) {
        out->data[i] = -INFINITY;
    }
    
    for (int i = 0; i < a->size; i++) {
        int remaining = i;
        int out_idx = 0;
        int out_dim = 0;
        
        for (int d = 0; d < a->ndim; d++) {
            int coord = remaining / a->strides[d];
            remaining = remaining % a->strides[d];
            
            if (d != axis) {
                out_idx += coord * out->strides[out_dim];
                out_dim++;
            }
        }
        
        if (a->data[i] > out->data[out_idx]) {
            out->data[out_idx] = a->data[i];
        }
    }
}

/*============================================================================
 * UNARY OPERATIONS
 *============================================================================*/

void cg_tensor_exp(cg_tensor* a, cg_tensor* out) {
    assert(a != NULL && out != NULL);
    assert(a->size == out->size);
    
    for (int i = 0; i < out->size; i++) {
        out->data[i] = expf(a->data[i]);
    }
}

void cg_tensor_log(cg_tensor* a, cg_tensor* out) {
    assert(a != NULL && out != NULL);
    assert(a->size == out->size);
    
    for (int i = 0; i < out->size; i++) {
        out->data[i] = logf(a->data[i]);
    }
}

void cg_tensor_pow(cg_tensor* a, float power, cg_tensor* out) {
    assert(a != NULL && out != NULL);
    assert(a->size == out->size);
    
    for (int i = 0; i < out->size; i++) {
        out->data[i] = powf(a->data[i], power);
    }
}
