/**
 * BFloat16 Mixed Precision Training
 */

#ifndef CG_BF16_H
#define CG_BF16_H

#include "cg_tensor.h"
#include <stdint.h>

/*============================================================================
 * BFLOAT16 TYPE
 *============================================================================*/

/**
 * BFloat16: 1 sign + 8 exponent + 7 mantissa
 * Same range as FP32, less precision.
 */
typedef uint16_t cg_bf16;

/* Conversion macros */
static inline cg_bf16 cg_float_to_bf16(float f) {
    uint32_t bits = *(uint32_t*)&f;
    /* Round to nearest even */
    bits += 0x7FFF + ((bits >> 16) & 1);
    return (cg_bf16)(bits >> 16);
}

static inline float cg_bf16_to_float(cg_bf16 b) {
    uint32_t bits = ((uint32_t)b) << 16;
    return *(float*)&bits;
}

/*============================================================================
 * BF16 TENSOR
 *============================================================================*/

typedef struct {
    cg_bf16* data;              /* BF16 data */
    cg_bf16* grad;              /* BF16 gradients (optional) */
    float* data_fp32;           /* FP32 master copy (optional) */
    float* grad_fp32;           /* FP32 accumulated grads */
    
    int shape[8];
    int ndim;
    int size;
    
    bool requires_grad;
    bool has_master_copy;       /* Keep FP32 version for optimizer */
} cg_tensor_bf16;

cg_tensor_bf16* cg_tensor_bf16_new(int* shape, int ndim, bool requires_grad);
cg_tensor_bf16* cg_tensor_bf16_from_fp32(cg_tensor* fp32);
cg_tensor* cg_tensor_bf16_to_fp32(cg_tensor_bf16* bf16);
void cg_tensor_bf16_free(cg_tensor_bf16* t);

/*============================================================================
 * MIXED PRECISION TRAINER
 *============================================================================*/

typedef struct {
    /* Model parameters */
    cg_tensor** master_params;       /* FP32 master weights */
    cg_tensor_bf16** model_params;   /* BF16 compute weights */
    int num_params;
    
    /* Loss scaling */
    float loss_scale;                /* Current scale */
    float loss_scale_min;            /* Minimum scale */
    float loss_scale_max;            /* Maximum scale */
    int scale_growth_interval;       /* Steps between scale increase */
    int steps_since_update;          /* Steps since last scale change */
    int overflow_count;              /* Recent overflow count */
    
    /* Gradient clipping */
    float max_grad_norm;
    
    /* Optimizer reference */
    void* optimizer;                 /* FP32 optimizer */
} cg_mixed_precision;

cg_mixed_precision* cg_mixed_precision_new(cg_tensor** params, int num_params);

/**
 * Forward pass in BF16.
 */
cg_tensor_bf16* cg_mp_forward(cg_mixed_precision* mp, 
                               cg_tensor_bf16* (*model_fn)(void*, cg_tensor_bf16*),
                               void* model, cg_tensor_bf16* input);

/**
 * Backward pass with loss scaling.
 */
bool cg_mp_backward(cg_mixed_precision* mp, cg_tensor_bf16* loss);

/**
 * Optimizer step with gradient unscaling.
 */
void cg_mp_step(cg_mixed_precision* mp);

/**
 * Check for gradient overflow.
 */
bool cg_mp_check_overflow(cg_mixed_precision* mp);

void cg_mixed_precision_free(cg_mixed_precision* mp);

/*============================================================================
 * BF16 OPERATIONS
 *============================================================================*/

void cg_bf16_add(cg_tensor_bf16* a, cg_tensor_bf16* b, cg_tensor_bf16* out);
void cg_bf16_mul(cg_tensor_bf16* a, cg_tensor_bf16* b, cg_tensor_bf16* out);
void cg_bf16_matmul(cg_tensor_bf16* a, cg_tensor_bf16* b, cg_tensor_bf16* out);
void cg_bf16_gelu(cg_tensor_bf16* a, cg_tensor_bf16* out);
void cg_bf16_layernorm(cg_tensor_bf16* x, cg_tensor_bf16* gamma, 
                       cg_tensor_bf16* beta, float eps, cg_tensor_bf16* out);

/*============================================================================
 * FP8 (Future - H100)
 *============================================================================*/

typedef uint8_t cg_fp8_e4m3;   /* 4-bit exponent, 3-bit mantissa */
typedef uint8_t cg_fp8_e5m2;   /* 5-bit exponent, 2-bit mantissa */

#endif /* CG_BF16_H */
