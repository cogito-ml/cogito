/**
 * Core Tensor Implementation
 * 
 * Tensor creation, memory management, and utility functions.
 */

#include "cg_tensor.h"
#include "cg_arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*============================================================================
 * INTERNAL HELPERS
 *============================================================================*/

/* Calculate total size from shape */
static int calculate_size(int* shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

/* Compute strides for row-major layout */
static void compute_strides(int* shape, int* strides, int ndim) {
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

/* Initialize tensor fields (assumes memory already allocated) */
static void tensor_init(cg_tensor* t, int* shape, int ndim, bool requires_grad) {
    assert(ndim > 0 && ndim <= CG_MAX_DIMS);
    
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int));
    compute_strides(shape, t->strides, ndim);
    t->size = calculate_size(shape, ndim);
    
    t->requires_grad = requires_grad;
    t->backward_fn = NULL;
    t->num_parents = 0;
    t->backward_ctx = NULL;
    
    t->ref_count = 1;
    t->is_view = false;
    t->arena = NULL;
    
    /* Initialize parent pointers to NULL */
    for (int i = 0; i < CG_MAX_PARENTS; i++) {
        t->parents[i] = NULL;
    }
}

/*============================================================================
 * TENSOR CREATION
 *============================================================================*/

cg_tensor* cg_tensor_new(int* shape, int ndim, bool requires_grad) {
    assert(shape != NULL && ndim > 0 && ndim <= CG_MAX_DIMS);
    
    cg_tensor* t = (cg_tensor*)calloc(1, sizeof(cg_tensor));
    if (!t) return NULL;
    
    tensor_init(t, shape, ndim, requires_grad);
    
    t->data = (float*)calloc(t->size, sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }
    
    if (requires_grad) {
        t->grad = (float*)calloc(t->size, sizeof(float));
        if (!t->grad) {
            free(t->data);
            free(t);
            return NULL;
        }
    } else {
        t->grad = NULL;
    }
    
    return t;
}

cg_tensor* cg_tensor_zeros(int* shape, int ndim, bool requires_grad) {
    /* calloc already zeros the memory */
    return cg_tensor_new(shape, ndim, requires_grad);
}

cg_tensor* cg_tensor_ones(int* shape, int ndim, bool requires_grad) {
    cg_tensor* t = cg_tensor_new(shape, ndim, requires_grad);
    if (!t) return NULL;
    
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 1.0f;
    }
    
    return t;
}

cg_tensor* cg_tensor_full(int* shape, int ndim, float value, bool requires_grad) {
    cg_tensor* t = cg_tensor_new(shape, ndim, requires_grad);
    if (!t) return NULL;
    
    for (int i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
    
    return t;
}

cg_tensor* cg_tensor_rand(int* shape, int ndim, unsigned int seed, bool requires_grad) {
    cg_tensor* t = cg_tensor_new(shape, ndim, requires_grad);
    if (!t) return NULL;
    
    srand(seed);
    for (int i = 0; i < t->size; i++) {
        t->data[i] = (float)rand() / (float)RAND_MAX;
    }
    
    return t;
}

cg_tensor* cg_tensor_randn(int* shape, int ndim, unsigned int seed, bool requires_grad) {
    cg_tensor* t = cg_tensor_new(shape, ndim, requires_grad);
    if (!t) return NULL;
    
    srand(seed);
    
    /* Box-Muller transform for normal distribution */
    for (int i = 0; i < t->size; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);  /* (0, 1) */
        float u2 = (float)rand() / (float)RAND_MAX;  /* [0, 1) */
        
        float mag = sqrtf(-2.0f * logf(u1));
        float z0 = mag * cosf(2.0f * (float)M_PI * u2);
        float z1 = mag * sinf(2.0f * (float)M_PI * u2);
        
        t->data[i] = z0;
        if (i + 1 < t->size) {
            t->data[i + 1] = z1;
        }
    }
    
    return t;
}

cg_tensor* cg_tensor_from_data(float* data, int* shape, int ndim, bool requires_grad) {
    assert(data != NULL);
    
    cg_tensor* t = cg_tensor_new(shape, ndim, requires_grad);
    if (!t) return NULL;
    
    memcpy(t->data, data, t->size * sizeof(float));
    
    return t;
}

cg_tensor* cg_tensor_clone(cg_tensor* t) {
    assert(t != NULL);
    
    cg_tensor* clone = cg_tensor_new(t->shape, t->ndim, t->requires_grad);
    if (!clone) return NULL;
    
    memcpy(clone->data, t->data, t->size * sizeof(float));
    
    if (t->grad && clone->grad) {
        memcpy(clone->grad, t->grad, t->size * sizeof(float));
    }
    
    return clone;
}

/*============================================================================
 * ARENA-BASED TENSOR CREATION
 *============================================================================*/

cg_tensor* cg_tensor_zeros_arena(cg_arena* arena, int* shape, int ndim, bool requires_grad) {
    assert(arena != NULL && shape != NULL && ndim > 0 && ndim <= CG_MAX_DIMS);
    
    cg_tensor* t = (cg_tensor*)cg_arena_alloc(arena, sizeof(cg_tensor), 16);
    if (!t) return NULL;
    
    memset(t, 0, sizeof(cg_tensor));
    tensor_init(t, shape, ndim, requires_grad);
    t->arena = arena;
    
    t->data = (float*)cg_arena_alloc(arena, t->size * sizeof(float), 16);
    if (!t->data) return NULL;
    memset(t->data, 0, t->size * sizeof(float));
    
    if (requires_grad) {
        t->grad = (float*)cg_arena_alloc(arena, t->size * sizeof(float), 16);
        if (!t->grad) return NULL;
        memset(t->grad, 0, t->size * sizeof(float));
    }
    
    return t;
}

cg_tensor* cg_tensor_randn_arena(cg_arena* arena, int* shape, int ndim, unsigned int seed, bool requires_grad) {
    cg_tensor* t = cg_tensor_zeros_arena(arena, shape, ndim, requires_grad);
    if (!t) return NULL;
    
    srand(seed);
    
    for (int i = 0; i < t->size; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        
        float mag = sqrtf(-2.0f * logf(u1));
        float z0 = mag * cosf(2.0f * (float)M_PI * u2);
        float z1 = mag * sinf(2.0f * (float)M_PI * u2);
        
        t->data[i] = z0;
        if (i + 1 < t->size) {
            t->data[i + 1] = z1;
        }
    }
    
    return t;
}

/*============================================================================
 * MEMORY MANAGEMENT
 *============================================================================*/

void cg_tensor_retain(cg_tensor* t) {
    if (t) t->ref_count++;
}

void cg_tensor_release(cg_tensor* t) {
    if (!t) return;
    
    t->ref_count--;
    if (t->ref_count <= 0) {
        cg_tensor_free(t);
    }
}

void cg_tensor_free(cg_tensor* t) {
    if (!t) return;
    
    /* Don't free if allocated from arena */
    if (t->arena) return;
    
    /* Release parent references */
    for (int i = 0; i < t->num_parents; i++) {
        if (t->parents[i]) {
            cg_tensor_release(t->parents[i]);
        }
    }
    
    /* Free backward context if present */
    if (t->backward_ctx) {
        free(t->backward_ctx);
    }
    
    if (!t->is_view) {
        free(t->data);
    }
    free(t->grad);
    free(t);
}

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

void cg_tensor_print(cg_tensor* t, const char* name) {
    if (!t) {
        printf("%s: NULL\n", name ? name : "tensor");
        return;
    }
    
    printf("%s: shape=[", name ? name : "tensor");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("], requires_grad=%s\n", t->requires_grad ? "true" : "false");
    
    /* Print data (limit output for large tensors) */
    if (t->size <= 20) {
        printf("  data: [");
        for (int i = 0; i < t->size; i++) {
            printf("%.4f%s", t->data[i], i < t->size - 1 ? ", " : "");
        }
        printf("]\n");
    } else {
        printf("  data: [%.4f, %.4f, %.4f, ..., %.4f, %.4f, %.4f]\n",
               t->data[0], t->data[1], t->data[2],
               t->data[t->size - 3], t->data[t->size - 2], t->data[t->size - 1]);
    }
    
    if (t->grad) {
        if (t->size <= 20) {
            printf("  grad: [");
            for (int i = 0; i < t->size; i++) {
                printf("%.4f%s", t->grad[i], i < t->size - 1 ? ", " : "");
            }
            printf("]\n");
        } else {
            printf("  grad: [%.4f, %.4f, %.4f, ..., %.4f, %.4f, %.4f]\n",
                   t->grad[0], t->grad[1], t->grad[2],
                   t->grad[t->size - 3], t->grad[t->size - 2], t->grad[t->size - 1]);
        }
    }
}

float cg_tensor_get(cg_tensor* t, int* indices) {
    assert(t != NULL && indices != NULL);
    
    int flat_idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        assert(indices[i] >= 0 && indices[i] < t->shape[i]);
        flat_idx += indices[i] * t->strides[i];
    }
    
    return t->data[flat_idx];
}

void cg_tensor_set(cg_tensor* t, int* indices, float value) {
    assert(t != NULL && indices != NULL);
    
    int flat_idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        assert(indices[i] >= 0 && indices[i] < t->shape[i]);
        flat_idx += indices[i] * t->strides[i];
    }
    
    t->data[flat_idx] = value;
}

void cg_tensor_reshape(cg_tensor* t, int* new_shape, int new_ndim) {
    assert(t != NULL && new_shape != NULL && new_ndim > 0 && new_ndim <= CG_MAX_DIMS);
    
    int new_size = calculate_size(new_shape, new_ndim);
    assert(new_size == t->size && "Reshape must preserve total size");
    
    t->ndim = new_ndim;
    memcpy(t->shape, new_shape, new_ndim * sizeof(int));
    compute_strides(new_shape, t->strides, new_ndim);
}

bool cg_tensor_shape_equal(cg_tensor* a, cg_tensor* b) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    
    return true;
}

void cg_tensor_zero_grad(cg_tensor* t) {
    if (t && t->grad) {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

/*============================================================================
 * ERROR STRINGS
 *============================================================================*/

const char* cg_error_string(int err) {
    switch (err) {
        case 0: return "Success";
        case 1: return "Null pointer";
        case 2: return "Shape mismatch";
        case 3: return "Out of memory";
        case 4: return "Invalid argument";
        case 5: return "File not found";
        case 6: return "Invalid format";
        default: return "Unknown error";
    }
}
