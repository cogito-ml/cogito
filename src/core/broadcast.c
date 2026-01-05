/**
 * Broadcasting - NumPy-style broadcasting for element-wise operations
 */

#include "cg_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*============================================================================
 * BROADCASTING UTILITIES
 *============================================================================*/

/**
 * Compute broadcasted output shape from two input shapes.
 * Returns true if shapes are broadcastable, false otherwise.
 */
bool cg_broadcast_shapes(int* shape_a, int ndim_a, int* shape_b, int ndim_b,
                         int* out_shape, int* out_ndim) {
    int max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    *out_ndim = max_ndim;
    
    /* Right-align shapes and check compatibility */
    for (int i = 0; i < max_ndim; i++) {
        int dim_a = (i < ndim_a) ? shape_a[ndim_a - 1 - i] : 1;
        int dim_b = (i < ndim_b) ? shape_b[ndim_b - 1 - i] : 1;
        
        if (dim_a == dim_b) {
            out_shape[max_ndim - 1 - i] = dim_a;
        } else if (dim_a == 1) {
            out_shape[max_ndim - 1 - i] = dim_b;
        } else if (dim_b == 1) {
            out_shape[max_ndim - 1 - i] = dim_a;
        } else {
            return false;  /* Incompatible shapes */
        }
    }
    
    return true;
}

/**
 * Compute broadcast strides for indexing into original arrays.
 * A stride of 0 means that dimension is broadcast (repeated).
 */
void cg_broadcast_strides(int* shape, int ndim, int* out_shape, int out_ndim,
                          int* broadcast_strides) {
    /* Compute original strides */
    int strides[CG_MAX_DIMS];
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    
    /* Compute broadcast strides (0 where dimension is 1) */
    int offset = out_ndim - ndim;
    for (int i = 0; i < out_ndim; i++) {
        if (i < offset) {
            broadcast_strides[i] = 0;  /* Prepended dimension */
        } else {
            int orig_idx = i - offset;
            if (shape[orig_idx] == 1) {
                broadcast_strides[i] = 0;  /* Broadcast this dim */
            } else {
                broadcast_strides[i] = strides[orig_idx];
            }
        }
    }
}

/**
 * Convert flat index in output to flat index in broadcast source.
 */
static int broadcast_index(int flat_idx, int* out_shape, int* broadcast_strides, int ndim) {
    int idx = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        int coord = flat_idx % out_shape[d];
        flat_idx /= out_shape[d];
        idx += coord * broadcast_strides[d];
    }
    return idx;
}

/*============================================================================
 * BROADCASTED OPERATIONS
 *============================================================================*/

/**
 * Broadcasted element-wise addition: out = a + b
 */
cg_tensor* cg_tensor_add_broadcast(cg_tensor* a, cg_tensor* b) {
    int out_shape[CG_MAX_DIMS];
    int out_ndim;
    
    if (!cg_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        fprintf(stderr, "cg_tensor_add_broadcast: incompatible shapes\n");
        return NULL;
    }
    
    /* Compute broadcast strides */
    int strides_a[CG_MAX_DIMS], strides_b[CG_MAX_DIMS];
    cg_broadcast_strides(a->shape, a->ndim, out_shape, out_ndim, strides_a);
    cg_broadcast_strides(b->shape, b->ndim, out_shape, out_ndim, strides_b);
    
    /* Create output */
    cg_tensor* out = cg_tensor_new(out_shape, out_ndim, a->requires_grad || b->requires_grad);
    
    /* Compute total size */
    int size = 1;
    for (int i = 0; i < out_ndim; i++) size *= out_shape[i];
    
    /* Perform broadcasted addition */
    for (int i = 0; i < size; i++) {
        int idx_a = broadcast_index(i, out_shape, strides_a, out_ndim);
        int idx_b = broadcast_index(i, out_shape, strides_b, out_ndim);
        out->data[i] = a->data[idx_a] + b->data[idx_b];
    }
    
    return out;
}

/**
 * Broadcasted element-wise multiplication: out = a * b
 */
cg_tensor* cg_tensor_mul_broadcast(cg_tensor* a, cg_tensor* b) {
    int out_shape[CG_MAX_DIMS];
    int out_ndim;
    
    if (!cg_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        fprintf(stderr, "cg_tensor_mul_broadcast: incompatible shapes\n");
        return NULL;
    }
    
    int strides_a[CG_MAX_DIMS], strides_b[CG_MAX_DIMS];
    cg_broadcast_strides(a->shape, a->ndim, out_shape, out_ndim, strides_a);
    cg_broadcast_strides(b->shape, b->ndim, out_shape, out_ndim, strides_b);
    
    cg_tensor* out = cg_tensor_new(out_shape, out_ndim, a->requires_grad || b->requires_grad);
    
    int size = 1;
    for (int i = 0; i < out_ndim; i++) size *= out_shape[i];
    
    for (int i = 0; i < size; i++) {
        int idx_a = broadcast_index(i, out_shape, strides_a, out_ndim);
        int idx_b = broadcast_index(i, out_shape, strides_b, out_ndim);
        out->data[i] = a->data[idx_a] * b->data[idx_b];
    }
    
    return out;
}

/**
 * Broadcasted subtraction: out = a - b
 */
cg_tensor* cg_tensor_sub_broadcast(cg_tensor* a, cg_tensor* b) {
    int out_shape[CG_MAX_DIMS];
    int out_ndim;
    
    if (!cg_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        return NULL;
    }
    
    int strides_a[CG_MAX_DIMS], strides_b[CG_MAX_DIMS];
    cg_broadcast_strides(a->shape, a->ndim, out_shape, out_ndim, strides_a);
    cg_broadcast_strides(b->shape, b->ndim, out_shape, out_ndim, strides_b);
    
    cg_tensor* out = cg_tensor_new(out_shape, out_ndim, a->requires_grad || b->requires_grad);
    
    int size = 1;
    for (int i = 0; i < out_ndim; i++) size *= out_shape[i];
    
    for (int i = 0; i < size; i++) {
        int idx_a = broadcast_index(i, out_shape, strides_a, out_ndim);
        int idx_b = broadcast_index(i, out_shape, strides_b, out_ndim);
        out->data[i] = a->data[idx_a] - b->data[idx_b];
    }
    
    return out;
}

/**
 * Broadcasted division: out = a / b
 */
cg_tensor* cg_tensor_div_broadcast(cg_tensor* a, cg_tensor* b) {
    int out_shape[CG_MAX_DIMS];
    int out_ndim;
    
    if (!cg_broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        return NULL;
    }
    
    int strides_a[CG_MAX_DIMS], strides_b[CG_MAX_DIMS];
    cg_broadcast_strides(a->shape, a->ndim, out_shape, out_ndim, strides_a);
    cg_broadcast_strides(b->shape, b->ndim, out_shape, out_ndim, strides_b);
    
    cg_tensor* out = cg_tensor_new(out_shape, out_ndim, a->requires_grad || b->requires_grad);
    
    int size = 1;
    for (int i = 0; i < out_ndim; i++) size *= out_shape[i];
    
    for (int i = 0; i < size; i++) {
        int idx_a = broadcast_index(i, out_shape, strides_a, out_ndim);
        int idx_b = broadcast_index(i, out_shape, strides_b, out_ndim);
        out->data[i] = a->data[idx_a] / b->data[idx_b];
    }
    
    return out;
}

/*============================================================================
 * TENSOR STACKING
 *============================================================================*/

/**
 * Stack multiple tensors along a new first dimension.
 * All tensors must have the same shape.
 */
cg_tensor* cg_tensor_stack(cg_tensor** tensors, int count) {
    if (count == 0 || !tensors || !tensors[0]) return NULL;
    
    cg_tensor* first = tensors[0];
    int new_shape[CG_MAX_DIMS];
    new_shape[0] = count;
    for (int i = 0; i < first->ndim; i++) {
        new_shape[i + 1] = first->shape[i];
    }
    
    bool requires_grad = false;
    for (int i = 0; i < count; i++) {
        if (tensors[i]->requires_grad) requires_grad = true;
    }
    
    cg_tensor* out = cg_tensor_new(new_shape, first->ndim + 1, requires_grad);
    
    int elem_size = first->size;
    for (int i = 0; i < count; i++) {
        memcpy(out->data + i * elem_size, tensors[i]->data, elem_size * sizeof(float));
    }
    
    return out;
}

/**
 * Concatenate tensors along an existing dimension.
 */
cg_tensor* cg_tensor_cat(cg_tensor** tensors, int count, int dim) {
    if (count == 0 || !tensors || !tensors[0]) return NULL;
    
    cg_tensor* first = tensors[0];
    
    /* Compute output shape */
    int new_shape[CG_MAX_DIMS];
    memcpy(new_shape, first->shape, first->ndim * sizeof(int));
    
    int total_dim = first->shape[dim];
    for (int i = 1; i < count; i++) {
        total_dim += tensors[i]->shape[dim];
    }
    new_shape[dim] = total_dim;
    
    bool requires_grad = false;
    for (int i = 0; i < count; i++) {
        if (tensors[i]->requires_grad) requires_grad = true;
    }
    
    cg_tensor* out = cg_tensor_new(new_shape, first->ndim, requires_grad);
    
    /* Simple case: concatenate along first dimension */
    if (dim == 0) {
        int offset = 0;
        for (int i = 0; i < count; i++) {
            memcpy(out->data + offset, tensors[i]->data, tensors[i]->size * sizeof(float));
            offset += tensors[i]->size;
        }
    } else {
        /* General case - iterate and copy slices */
        /* For simplicity, just handle dim=0 for now */
        fprintf(stderr, "cg_tensor_cat: only dim=0 supported\n");
    }
    
    return out;
}
