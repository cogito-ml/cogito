/**
 * Tensor System for Cogito
 * 
 * Core tensor structure with automatic differentiation support.
 * All operations track gradients when requires_grad is true.
 */

#ifndef CG_TENSOR_H
#define CG_TENSOR_H

#include <stddef.h>
#include <stdbool.h>

/* Forward declarations */
typedef struct cg_tensor cg_tensor;
typedef struct cg_arena cg_arena;

/* Maximum number of dimensions supported */
#define CG_MAX_DIMS 8

/* Maximum number of parent tensors for backward pass */
#define CG_MAX_PARENTS 4

/* Backward function signature */
typedef void (*cg_backward_fn)(cg_tensor* self);

/**
 * Core tensor structure.
 * 
 * Memory layout: Row-major (C-style), last dimension varies fastest.
 * Example: For shape [2, 3, 4], strides would be [12, 4, 1].
 */
struct cg_tensor {
    /* Data storage */
    float* data;                   /* Tensor data (row-major) */
    float* grad;                   /* Gradient data (same shape as data) */
    
    /* Shape information */
    int shape[CG_MAX_DIMS];        /* Dimension sizes */
    int strides[CG_MAX_DIMS];      /* Strides for indexing */
    int ndim;                      /* Number of dimensions */
    int size;                      /* Total number of elements */
    
    /* Autograd support */
    bool requires_grad;            /* Whether to track gradients */
    cg_backward_fn backward_fn;    /* Backward pass function */
    cg_tensor* parents[CG_MAX_PARENTS];  /* Parent tensors in compute graph */
    int num_parents;               /* Number of parents */
    void* backward_ctx;            /* Extra context for backward (e.g., saved tensors) */
    
    /* Memory management */
    int ref_count;                 /* Reference count for memory management */
    bool is_view;                  /* True if this tensor shares data with another */
    cg_arena* arena;               /* Arena this tensor was allocated from (or NULL) */
};

/*============================================================================
 * TENSOR CREATION
 *============================================================================*/

/**
 * Create a new uninitialized tensor.
 * 
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param requires_grad Whether to track gradients
 * @return New tensor or NULL on failure
 */
cg_tensor* cg_tensor_new(int* shape, int ndim, bool requires_grad);

/**
 * Create a tensor filled with zeros.
 */
cg_tensor* cg_tensor_zeros(int* shape, int ndim, bool requires_grad);

/**
 * Create a tensor filled with ones.
 */
cg_tensor* cg_tensor_ones(int* shape, int ndim, bool requires_grad);

/**
 * Create a tensor filled with a constant value.
 */
cg_tensor* cg_tensor_full(int* shape, int ndim, float value, bool requires_grad);

/**
 * Create a tensor with random values from uniform distribution [0, 1).
 */
cg_tensor* cg_tensor_rand(int* shape, int ndim, unsigned int seed, bool requires_grad);

/**
 * Create a tensor with random values from standard normal distribution.
 * Uses Box-Muller transform.
 */
cg_tensor* cg_tensor_randn(int* shape, int ndim, unsigned int seed, bool requires_grad);

/**
 * Create tensor from existing data (copies the data).
 */
cg_tensor* cg_tensor_from_data(float* data, int* shape, int ndim, bool requires_grad);

/**
 * Create a copy of a tensor.
 */
cg_tensor* cg_tensor_clone(cg_tensor* t);

/*============================================================================
 * ARENA-BASED TENSOR CREATION
 *============================================================================*/

/**
 * Create tensor in an arena (for bulk deallocation).
 */
cg_tensor* cg_tensor_zeros_arena(cg_arena* arena, int* shape, int ndim, bool requires_grad);
cg_tensor* cg_tensor_randn_arena(cg_arena* arena, int* shape, int ndim, unsigned int seed, bool requires_grad);

/*============================================================================
 * MEMORY MANAGEMENT
 *============================================================================*/

/**
 * Increment reference count.
 */
void cg_tensor_retain(cg_tensor* t);

/**
 * Decrement reference count and free if zero.
 */
void cg_tensor_release(cg_tensor* t);

/**
 * Free tensor memory (use release for reference counted tensors).
 */
void cg_tensor_free(cg_tensor* t);

/*============================================================================
 * TENSOR OPERATIONS (In-place output for efficiency)
 *============================================================================*/

/**
 * Element-wise addition: out = a + b
 * Broadcasting: If shapes differ, broadcasts according to numpy rules.
 */
void cg_tensor_add(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * Element-wise subtraction: out = a - b
 */
void cg_tensor_sub(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * Element-wise multiplication: out = a * b
 */
void cg_tensor_mul(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * Element-wise division: out = a / b
 */
void cg_tensor_div(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * Scalar multiplication: out = a * scalar
 */
void cg_tensor_scale(cg_tensor* a, float scalar, cg_tensor* out);

/**
 * Matrix multiplication: out = a @ b
 * For 2D tensors: standard matmul
 * For higher dims: batch matmul
 */
void cg_tensor_matmul(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * Transpose a 2D tensor: out = a^T
 */
void cg_tensor_transpose(cg_tensor* a, cg_tensor* out);

/**
 * Sum reduction over an axis (or all elements if axis < 0).
 */
void cg_tensor_sum(cg_tensor* a, int axis, cg_tensor* out);

/**
 * Mean reduction over an axis (or all elements if axis < 0).
 */
void cg_tensor_mean(cg_tensor* a, int axis, cg_tensor* out);

/**
 * Maximum over an axis.
 */
void cg_tensor_max(cg_tensor* a, int axis, cg_tensor* out);

/**
 * Element-wise exponential: out = exp(a)
 */
void cg_tensor_exp(cg_tensor* a, cg_tensor* out);

/**
 * Element-wise natural logarithm: out = log(a)
 */
void cg_tensor_log(cg_tensor* a, cg_tensor* out);

/**
 * Element-wise power: out = a^power
 */
void cg_tensor_pow(cg_tensor* a, float power, cg_tensor* out);

/*============================================================================
 * AUTOGRAD
 *============================================================================*/

/**
 * Perform backward pass from this tensor.
 * The tensor should be a scalar (single element) loss value.
 * Assumes gradient of output is 1.0.
 */
void cg_backward(cg_tensor* loss);

/**
 * Zero all gradients in the computation graph.
 */
void cg_tensor_zero_grad(cg_tensor* t);

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

/**
 * Print tensor to stdout for debugging.
 */
void cg_tensor_print(cg_tensor* t, const char* name);

/**
 * Get element at multi-dimensional index.
 */
float cg_tensor_get(cg_tensor* t, int* indices);

/**
 * Set element at multi-dimensional index.
 */
void cg_tensor_set(cg_tensor* t, int* indices, float value);

/**
 * Reshape tensor (must have same total size).
 */
void cg_tensor_reshape(cg_tensor* t, int* new_shape, int new_ndim);

/**
 * Check if two tensors have the same shape.
 */
bool cg_tensor_shape_equal(cg_tensor* a, cg_tensor* b);

#endif /* CG_TENSOR_H */
