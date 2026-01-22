/**
 * Symbolic Shapes - Dynamic dimension tracking
 */

#include "cg_symbolic.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*============================================================================
 * EXPRESSION
 *============================================================================*/

cg_symbolic_expr* cg_expr_const(int64_t value) {
    cg_symbolic_expr* expr = (cg_symbolic_expr*)calloc(1, sizeof(cg_symbolic_expr));
    expr->type = EXPR_CONST;
    expr->data.constant = value;
    return expr;
}

cg_symbolic_expr* cg_expr_var(const char* name) {
    cg_symbolic_expr* expr = (cg_symbolic_expr*)calloc(1, sizeof(cg_symbolic_expr));
    expr->type = EXPR_VAR;
    expr->data.variable = strdup(name);
    return expr;
}

static cg_symbolic_expr* expr_binary(cg_expr_type type, 
                                     cg_symbolic_expr* left, 
                                     cg_symbolic_expr* right) {
    cg_symbolic_expr* expr = (cg_symbolic_expr*)calloc(1, sizeof(cg_symbolic_expr));
    expr->type = type;
    expr->data.binary.left = left;
    expr->data.binary.right = right;
    return expr;
}

cg_symbolic_expr* cg_expr_add(cg_symbolic_expr* left, cg_symbolic_expr* right) {
    /* Constant folding */
    if (left->type == EXPR_CONST && right->type == EXPR_CONST) {
        return cg_expr_const(left->data.constant + right->data.constant);
    }
    /* Identity: x + 0 = x */
    if (right->type == EXPR_CONST && right->data.constant == 0) {
        return left;
    }
    return expr_binary(EXPR_ADD, left, right);
}

cg_symbolic_expr* cg_expr_sub(cg_symbolic_expr* left, cg_symbolic_expr* right) {
    if (left->type == EXPR_CONST && right->type == EXPR_CONST) {
        return cg_expr_const(left->data.constant - right->data.constant);
    }
    return expr_binary(EXPR_SUB, left, right);
}

cg_symbolic_expr* cg_expr_mul(cg_symbolic_expr* left, cg_symbolic_expr* right) {
    if (left->type == EXPR_CONST && right->type == EXPR_CONST) {
        return cg_expr_const(left->data.constant * right->data.constant);
    }
    /* Identity: x * 1 = x */
    if (right->type == EXPR_CONST && right->data.constant == 1) {
        return left;
    }
    /* Zero: x * 0 = 0 */
    if (right->type == EXPR_CONST && right->data.constant == 0) {
        return cg_expr_const(0);
    }
    return expr_binary(EXPR_MUL, left, right);
}

cg_symbolic_expr* cg_expr_div(cg_symbolic_expr* left, cg_symbolic_expr* right) {
    if (left->type == EXPR_CONST && right->type == EXPR_CONST) {
        return cg_expr_const(left->data.constant / right->data.constant);
    }
    return expr_binary(EXPR_DIV, left, right);
}

cg_symbolic_expr* cg_expr_floordiv(cg_symbolic_expr* left, cg_symbolic_expr* right) {
    if (left->type == EXPR_CONST && right->type == EXPR_CONST) {
        int64_t a = left->data.constant;
        int64_t b = right->data.constant;
        return cg_expr_const(a / b - (a % b != 0 && (a ^ b) < 0));
    }
    return expr_binary(EXPR_FLOORDIV, left, right);
}

int64_t cg_expr_eval(cg_symbolic_expr* expr, int64_t* bindings, 
                     const char** binding_names, int num_bindings) {
    switch (expr->type) {
        case EXPR_CONST:
            return expr->data.constant;
            
        case EXPR_VAR:
            for (int i = 0; i < num_bindings; i++) {
                if (strcmp(binding_names[i], expr->data.variable) == 0) {
                    return bindings[i];
                }
            }
            fprintf(stderr, "Unbound variable: %s\n", expr->data.variable);
            return -1;
            
        case EXPR_ADD:
            return cg_expr_eval(expr->data.binary.left, bindings, binding_names, num_bindings) +
                   cg_expr_eval(expr->data.binary.right, bindings, binding_names, num_bindings);
                   
        case EXPR_SUB:
            return cg_expr_eval(expr->data.binary.left, bindings, binding_names, num_bindings) -
                   cg_expr_eval(expr->data.binary.right, bindings, binding_names, num_bindings);
                   
        case EXPR_MUL:
            return cg_expr_eval(expr->data.binary.left, bindings, binding_names, num_bindings) *
                   cg_expr_eval(expr->data.binary.right, bindings, binding_names, num_bindings);
                   
        case EXPR_DIV:
            return cg_expr_eval(expr->data.binary.left, bindings, binding_names, num_bindings) /
                   cg_expr_eval(expr->data.binary.right, bindings, binding_names, num_bindings);
                   
        case EXPR_FLOORDIV: {
            int64_t a = cg_expr_eval(expr->data.binary.left, bindings, binding_names, num_bindings);
            int64_t b = cg_expr_eval(expr->data.binary.right, bindings, binding_names, num_bindings);
            return a / b - (a % b != 0 && (a ^ b) < 0);
        }
        
        case EXPR_MAX: {
            int64_t a = cg_expr_eval(expr->data.binary.left, bindings, binding_names, num_bindings);
            int64_t b = cg_expr_eval(expr->data.binary.right, bindings, binding_names, num_bindings);
            return a > b ? a : b;
        }
        
        case EXPR_MIN: {
            int64_t a = cg_expr_eval(expr->data.binary.left, bindings, binding_names, num_bindings);
            int64_t b = cg_expr_eval(expr->data.binary.right, bindings, binding_names, num_bindings);
            return a < b ? a : b;
        }
        
        default:
            return -1;
    }
}

char* cg_expr_to_string(cg_symbolic_expr* expr) {
    char* buf = (char*)malloc(256);
    
    switch (expr->type) {
        case EXPR_CONST:
            snprintf(buf, 256, "%lld", (long long)expr->data.constant);
            break;
        case EXPR_VAR:
            snprintf(buf, 256, "%s", expr->data.variable);
            break;
        case EXPR_ADD:
        case EXPR_SUB:
        case EXPR_MUL:
        case EXPR_DIV:
        case EXPR_FLOORDIV: {
            char* left = cg_expr_to_string(expr->data.binary.left);
            char* right = cg_expr_to_string(expr->data.binary.right);
            const char* ops[] = {"+", "-", "*", "/", "//"};
            snprintf(buf, 256, "(%s %s %s)", left, 
                     ops[expr->type - EXPR_ADD], right);
            free(left);
            free(right);
            break;
        }
        default:
            snprintf(buf, 256, "?");
            break;
    }
    
    return buf;
}

void cg_expr_free(cg_symbolic_expr* expr) {
    if (!expr) return;
    
    if (expr->type == EXPR_VAR) {
        free(expr->data.variable);
    } else if (expr->type >= EXPR_ADD) {
        cg_expr_free(expr->data.binary.left);
        cg_expr_free(expr->data.binary.right);
    }
    
    free(expr);
}

/*============================================================================
 * DIMENSION
 *============================================================================*/

cg_symbolic_dim* cg_dim_const(int64_t value) {
    cg_symbolic_dim* dim = (cg_symbolic_dim*)calloc(1, sizeof(cg_symbolic_dim));
    dim->type = DIM_CONSTANT;
    dim->value = value;
    dim->min_value = value;
    dim->max_value = value;
    return dim;
}

cg_symbolic_dim* cg_dim_var(const char* name) {
    cg_symbolic_dim* dim = (cg_symbolic_dim*)calloc(1, sizeof(cg_symbolic_dim));
    dim->type = DIM_SYMBOLIC;
    dim->name = strdup(name);
    dim->value = -1;  /* Unknown */
    dim->min_value = 1;
    dim->max_value = INT64_MAX;
    return dim;
}

cg_symbolic_dim* cg_dim_inferred(cg_symbolic_expr* expr) {
    cg_symbolic_dim* dim = (cg_symbolic_dim*)calloc(1, sizeof(cg_symbolic_dim));
    dim->type = DIM_INFERRED;
    dim->expr = expr;
    dim->min_value = 0;
    dim->max_value = INT64_MAX;
    return dim;
}

void cg_dim_constrain(cg_symbolic_dim* dim, int64_t min, int64_t max) {
    if (dim->min_value < min) dim->min_value = min;
    if (dim->max_value > max) dim->max_value = max;
}

bool cg_dim_compatible(cg_symbolic_dim* a, cg_symbolic_dim* b) {
    /* Constants must match exactly */
    if (a->type == DIM_CONSTANT && b->type == DIM_CONSTANT) {
        return a->value == b->value;
    }
    
    /* Broadcast dimension always compatible */
    if (a->type == DIM_BROADCAST || b->type == DIM_BROADCAST) {
        return true;
    }
    
    /* Same symbolic name */
    if (a->type == DIM_SYMBOLIC && b->type == DIM_SYMBOLIC) {
        return strcmp(a->name, b->name) == 0;
    }
    
    /* Constant with symbolic - check bounds */
    if (a->type == DIM_CONSTANT && b->type == DIM_SYMBOLIC) {
        return a->value >= b->min_value && a->value <= b->max_value;
    }
    if (b->type == DIM_CONSTANT && a->type == DIM_SYMBOLIC) {
        return b->value >= a->min_value && b->value <= a->max_value;
    }
    
    /* Default: assume compatible (runtime check) */
    return true;
}

int64_t cg_dim_resolve(cg_symbolic_dim* dim, int64_t* bindings, 
                       const char** binding_names, int num_bindings) {
    switch (dim->type) {
        case DIM_CONSTANT:
            return dim->value;
            
        case DIM_SYMBOLIC:
            for (int i = 0; i < num_bindings; i++) {
                if (strcmp(binding_names[i], dim->name) == 0) {
                    return bindings[i];
                }
            }
            return -1;
            
        case DIM_INFERRED:
            return cg_expr_eval(dim->expr, bindings, binding_names, num_bindings);
            
        case DIM_BROADCAST:
            return 1;  /* Or resolved based on context */
            
        default:
            return -1;
    }
}

void cg_dim_free(cg_symbolic_dim* dim) {
    if (!dim) return;
    free(dim->name);
    cg_expr_free(dim->expr);
    free(dim);
}

/*============================================================================
 * SHAPE
 *============================================================================*/

cg_symbolic_shape* cg_shape_new(cg_symbolic_dim** dims, int ndim) {
    cg_symbolic_shape* shape = (cg_symbolic_shape*)calloc(1, sizeof(cg_symbolic_shape));
    shape->ndim = ndim;
    shape->dims = (cg_symbolic_dim**)malloc(ndim * sizeof(cg_symbolic_dim*));
    memcpy(shape->dims, dims, ndim * sizeof(cg_symbolic_dim*));
    return shape;
}

cg_symbolic_shape* cg_shape_from_ints(int* dims, int ndim) {
    cg_symbolic_dim** sym_dims = (cg_symbolic_dim**)malloc(ndim * sizeof(cg_symbolic_dim*));
    for (int i = 0; i < ndim; i++) {
        sym_dims[i] = cg_dim_const(dims[i]);
    }
    cg_symbolic_shape* shape = cg_shape_new(sym_dims, ndim);
    free(sym_dims);
    return shape;
}

bool cg_shape_is_dynamic(cg_symbolic_shape* shape) {
    for (int i = 0; i < shape->ndim; i++) {
        if (shape->dims[i]->type != DIM_CONSTANT) {
            return true;
        }
    }
    return false;
}

cg_symbolic_shape* cg_shape_broadcast(cg_symbolic_shape* a, cg_symbolic_shape* b) {
    int max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    cg_symbolic_dim** out_dims = (cg_symbolic_dim**)malloc(max_ndim * sizeof(cg_symbolic_dim*));
    
    for (int i = 0; i < max_ndim; i++) {
        int a_idx = a->ndim - 1 - (max_ndim - 1 - i);
        int b_idx = b->ndim - 1 - (max_ndim - 1 - i);
        
        cg_symbolic_dim* dim_a = (a_idx >= 0) ? a->dims[a_idx] : cg_dim_const(1);
        cg_symbolic_dim* dim_b = (b_idx >= 0) ? b->dims[b_idx] : cg_dim_const(1);
        
        if (dim_a->type == DIM_CONSTANT && dim_a->value == 1) {
            out_dims[i] = dim_b;
        } else if (dim_b->type == DIM_CONSTANT && dim_b->value == 1) {
            out_dims[i] = dim_a;
        } else if (cg_dim_compatible(dim_a, dim_b)) {
            out_dims[i] = dim_a;  /* Take non-broadcast dim */
        } else {
            /* Error: incompatible */
            free(out_dims);
            return NULL;
        }
    }
    
    cg_symbolic_shape* out = cg_shape_new(out_dims, max_ndim);
    free(out_dims);
    return out;
}

cg_symbolic_shape* cg_shape_matmul(cg_symbolic_shape* a, cg_symbolic_shape* b) {
    if (a->ndim < 1 || b->ndim < 1) return NULL;
    
    /* Last dim of a must match second-to-last of b */
    cg_symbolic_dim* out_dims[8];
    int out_ndim = a->ndim;
    
    for (int i = 0; i < a->ndim - 1; i++) {
        out_dims[i] = a->dims[i];
    }
    out_dims[a->ndim - 1] = b->dims[b->ndim - 1];  /* Take last dim of b */
    
    return cg_shape_new(out_dims, out_ndim);
}

cg_symbolic_shape* cg_shape_conv2d(cg_symbolic_shape* input, int out_channels,
                                   int kernel_size, int stride, int padding) {
    if (input->ndim != 4) return NULL;
    
    cg_symbolic_dim* out_dims[4];
    out_dims[0] = input->dims[0];  /* Batch */
    out_dims[1] = cg_dim_const(out_channels);
    
    /* H_out = (H + 2*pad - kernel) / stride + 1 */
    cg_symbolic_expr* h_expr = cg_expr_add(
        cg_expr_floordiv(
            cg_expr_sub(
                cg_expr_add(
                    (input->dims[2]->type == DIM_CONSTANT) 
                        ? cg_expr_const(input->dims[2]->value)
                        : cg_expr_var(input->dims[2]->name),
                    cg_expr_const(2 * padding - kernel_size)
                ),
                cg_expr_const(0)
            ),
            cg_expr_const(stride)
        ),
        cg_expr_const(1)
    );
    out_dims[2] = cg_dim_inferred(h_expr);
    
    /* Similar for W */
    cg_symbolic_expr* w_expr = cg_expr_add(
        cg_expr_floordiv(
            cg_expr_add(
                (input->dims[3]->type == DIM_CONSTANT)
                    ? cg_expr_const(input->dims[3]->value)
                    : cg_expr_var(input->dims[3]->name),
                cg_expr_const(2 * padding - kernel_size)
            ),
            cg_expr_const(stride)
        ),
        cg_expr_const(1)
    );
    out_dims[3] = cg_dim_inferred(w_expr);
    
    return cg_shape_new(out_dims, 4);
}

uint64_t cg_shape_hash(cg_symbolic_shape* shape) {
    uint64_t hash = 5381;
    for (int i = 0; i < shape->ndim; i++) {
        cg_symbolic_dim* dim = shape->dims[i];
        hash = ((hash << 5) + hash) ^ dim->type;
        if (dim->type == DIM_CONSTANT) {
            hash = ((hash << 5) + hash) ^ (uint64_t)dim->value;
        } else if (dim->type == DIM_SYMBOLIC && dim->name) {
            for (const char* c = dim->name; *c; c++) {
                hash = ((hash << 5) + hash) ^ (uint64_t)*c;
            }
        }
    }
    return hash;
}

bool cg_shape_to_ints(cg_symbolic_shape* shape, int* out_dims, int* out_ndim) {
    *out_ndim = shape->ndim;
    for (int i = 0; i < shape->ndim; i++) {
        if (shape->dims[i]->type != DIM_CONSTANT) {
            return false;
        }
        out_dims[i] = (int)shape->dims[i]->value;
    }
    return true;
}

void cg_shape_free(cg_symbolic_shape* shape) {
    if (!shape) return;
    for (int i = 0; i < shape->ndim; i++) {
        cg_dim_free(shape->dims[i]);
    }
    free(shape->dims);
    free(shape);
}

void cg_shape_print(cg_symbolic_shape* shape) {
    printf("[");
    for (int i = 0; i < shape->ndim; i++) {
        cg_symbolic_dim* dim = shape->dims[i];
        switch (dim->type) {
            case DIM_CONSTANT:
                printf("%lld", (long long)dim->value);
                break;
            case DIM_SYMBOLIC:
                printf("%s", dim->name);
                break;
            case DIM_INFERRED: {
                char* s = cg_expr_to_string(dim->expr);
                printf("%s", s);
                free(s);
                break;
            }
            case DIM_BROADCAST:
                printf("*");
                break;
        }
        if (i < shape->ndim - 1) printf(", ");
    }
    printf("]");
}

/*============================================================================
 * GUARD CHECK
 *============================================================================*/

bool cg_guards_check(cg_shape_specialization* spec, int64_t* bindings,
                     const char** binding_names, int num_bindings) {
    for (int i = 0; i < spec->num_guards; i++) {
        cg_shape_guard* guard = &spec->guards[i];
        
        for (int j = 0; j < num_bindings; j++) {
            if (strcmp(binding_names[j], guard->dim_name) == 0) {
                if (bindings[j] != guard->expected_value) {
                    return false;
                }
                break;
            }
        }
    }
    return true;
}

/*============================================================================
 * SYMBOLIC ERROR REPORTING
 *============================================================================*/

void cg_symbolic_report_mismatch(cg_symbolic_shape* a, cg_symbolic_shape* b, 
                                 const char* context) {
    printf("\n[Symbolic Error] Shape Mismatch in %s\n", context);
    
    printf("Shape A: ");
    cg_shape_print(a);
    printf("\n");
    
    printf("Shape B: ");
    cg_shape_print(b);
    printf("\n");
    
    /* Find mismatched dimension */
    int min_ndim = (a->ndim < b->ndim) ? a->ndim : b->ndim;
    for (int i = 0; i < min_ndim; i++) {
        /* Check compatibility logic again to pinpoint error */
        if (!cg_dim_compatible(a->dims[i], b->dims[i])) {
            printf("Mismatch at dim %d:\n", i);
            char* s_a = NULL;
            char* s_b = NULL;
            
            if (a->dims[i]->type == DIM_INFERRED) s_a = cg_expr_to_string(a->dims[i]->expr);
            if (b->dims[i]->type == DIM_INFERRED) s_b = cg_expr_to_string(b->dims[i]->expr);
            
            printf("  A[%d]: %s (Type: %d)\n", i, s_a ? s_a : "Direct", a->dims[i]->type);
            printf("  B[%d]: %s (Type: %d)\n", i, s_b ? s_b : "Direct", b->dims[i]->type);
            
            if (s_a) free(s_a);
            if (s_b) free(s_b);
        }
    }
}
