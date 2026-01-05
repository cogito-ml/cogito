/**
 * Symbolic Shapes - Dynamic dimensions with compile-time inference
 */

#ifndef CG_SYMBOLIC_H
#define CG_SYMBOLIC_H

#include <stdbool.h>
#include <stdint.h>

/*============================================================================
 * DIMENSION TYPE
 *============================================================================*/

typedef enum {
    DIM_CONSTANT,           /* Fixed size (e.g., 768) */
    DIM_SYMBOLIC,           /* Variable (e.g., batch_size) */
    DIM_INFERRED,           /* Computed from others */
    DIM_BROADCAST           /* Can be 1 or match another dim */
} cg_dim_type;

/*============================================================================
 * SYMBOLIC EXPRESSION
 *============================================================================*/

typedef enum {
    EXPR_CONST,
    EXPR_VAR,
    EXPR_ADD,
    EXPR_SUB,
    EXPR_MUL,
    EXPR_DIV,
    EXPR_FLOORDIV,
    EXPR_MOD,
    EXPR_MAX,
    EXPR_MIN
} cg_expr_type;

typedef struct cg_symbolic_expr {
    cg_expr_type type;
    union {
        int64_t constant;
        char* variable;
        struct {
            struct cg_symbolic_expr* left;
            struct cg_symbolic_expr* right;
        } binary;
    } data;
} cg_symbolic_expr;

/*============================================================================
 * SYMBOLIC DIMENSION
 *============================================================================*/

typedef struct {
    cg_dim_type type;
    int64_t value;              /* For CONSTANT dims */
    char* name;                 /* For SYMBOLIC dims (e.g., "batch") */
    cg_symbolic_expr* expr;     /* For INFERRED dims */
    
    /* Constraints */
    int64_t min_value;          /* Lower bound */
    int64_t max_value;          /* Upper bound */
    bool divisible_by;          /* Must be divisible by this */
    int64_t divisor;
} cg_symbolic_dim;

/*============================================================================
 * SYMBOLIC SHAPE
 *============================================================================*/

typedef struct {
    cg_symbolic_dim** dims;
    int ndim;
    uint64_t hash;              /* For shape cache lookup */
} cg_symbolic_shape;

/*============================================================================
 * DIMENSION API
 *============================================================================*/

/**
 * Create constant dimension.
 */
cg_symbolic_dim* cg_dim_const(int64_t value);

/**
 * Create symbolic (variable) dimension.
 */
cg_symbolic_dim* cg_dim_var(const char* name);

/**
 * Create inferred dimension from expression.
 */
cg_symbolic_dim* cg_dim_inferred(cg_symbolic_expr* expr);

/**
 * Add constraint to dimension.
 */
void cg_dim_constrain(cg_symbolic_dim* dim, int64_t min, int64_t max);

/**
 * Check if two dimensions are compatible.
 */
bool cg_dim_compatible(cg_symbolic_dim* a, cg_symbolic_dim* b);

/**
 * Substitute runtime value into symbolic dim.
 */
int64_t cg_dim_resolve(cg_symbolic_dim* dim, int64_t* bindings, const char** binding_names, int num_bindings);

/**
 * Free dimension.
 */
void cg_dim_free(cg_symbolic_dim* dim);

/*============================================================================
 * EXPRESSION API
 *============================================================================*/

cg_symbolic_expr* cg_expr_const(int64_t value);
cg_symbolic_expr* cg_expr_var(const char* name);
cg_symbolic_expr* cg_expr_add(cg_symbolic_expr* left, cg_symbolic_expr* right);
cg_symbolic_expr* cg_expr_sub(cg_symbolic_expr* left, cg_symbolic_expr* right);
cg_symbolic_expr* cg_expr_mul(cg_symbolic_expr* left, cg_symbolic_expr* right);
cg_symbolic_expr* cg_expr_div(cg_symbolic_expr* left, cg_symbolic_expr* right);
cg_symbolic_expr* cg_expr_floordiv(cg_symbolic_expr* left, cg_symbolic_expr* right);

int64_t cg_expr_eval(cg_symbolic_expr* expr, int64_t* bindings, 
                     const char** binding_names, int num_bindings);
char* cg_expr_to_string(cg_symbolic_expr* expr);
void cg_expr_free(cg_symbolic_expr* expr);

/*============================================================================
 * SHAPE API
 *============================================================================*/

/**
 * Create symbolic shape from dimensions.
 */
cg_symbolic_shape* cg_shape_new(cg_symbolic_dim** dims, int ndim);

/**
 * Create shape from integer array (all constant dims).
 */
cg_symbolic_shape* cg_shape_from_ints(int* dims, int ndim);

/**
 * Infer output shape for binary operation.
 */
cg_symbolic_shape* cg_shape_broadcast(cg_symbolic_shape* a, cg_symbolic_shape* b);

/**
 * Infer output shape for matmul.
 */
cg_symbolic_shape* cg_shape_matmul(cg_symbolic_shape* a, cg_symbolic_shape* b);

/**
 * Infer output shape for convolution.
 */
cg_symbolic_shape* cg_shape_conv2d(cg_symbolic_shape* input, int out_channels,
                                   int kernel_size, int stride, int padding);

/**
 * Check if shape has any symbolic dimensions.
 */
bool cg_shape_is_dynamic(cg_symbolic_shape* shape);

/**
 * Compute hash for shape (for kernel caching).
 */
uint64_t cg_shape_hash(cg_symbolic_shape* shape);

/**
 * Convert to concrete int array (fails if dynamic).
 */
bool cg_shape_to_ints(cg_symbolic_shape* shape, int* out_dims, int* out_ndim);

void cg_shape_free(cg_symbolic_shape* shape);
void cg_shape_print(cg_symbolic_shape* shape);

/*============================================================================
 * GUARD SYSTEM (for shape specialization)
 *============================================================================*/

typedef struct {
    char* dim_name;
    int64_t expected_value;
} cg_shape_guard;

typedef struct {
    cg_shape_guard* guards;
    int num_guards;
    void* specialized_kernel;
} cg_shape_specialization;

/**
 * Check if guards match current bindings.
 */
bool cg_guards_check(cg_shape_specialization* spec, int64_t* bindings, 
                     const char** binding_names, int num_bindings);

#endif /* CG_SYMBOLIC_H */
