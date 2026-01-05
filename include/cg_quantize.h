/**
 * INT4 Quantization - Weight and KV cache quantization
 */

#ifndef CG_QUANTIZE_H
#define CG_QUANTIZE_H

#include "cg_tensor.h"
#include <stdint.h>
#include <stdbool.h>

/*============================================================================
 * QUANTIZATION TYPES
 *============================================================================*/

typedef enum {
    QUANT_NONE = 0,
    QUANT_INT8,                 /* 8-bit symmetric */
    QUANT_INT4,                 /* 4-bit symmetric */
    QUANT_NF4,                  /* 4-bit normal float (QLoRA) */
    QUANT_FP8_E4M3,             /* 8-bit E4M3 float */
    QUANT_FP8_E5M2,             /* 8-bit E5M2 float */
    QUANT_GPTQ,                 /* GPTQ 4-bit */
    QUANT_AWQ,                  /* AWQ 4-bit */
    QUANT_GGML                  /* GGML mixed precision */
} cg_quant_type;

/*============================================================================
 * QUANTIZED TENSOR
 *============================================================================*/

typedef struct {
    uint8_t* data;              /* Packed quantized data */
    float* scales;              /* Per-group scales */
    float* zeros;               /* Per-group zero points (asymmetric) */
    
    int* shape;
    int ndim;
    int size;                   /* Number of elements (not bytes) */
    
    cg_quant_type quant_type;
    int group_size;             /* Elements per quantization group */
    int num_groups;
    
    bool symmetric;             /* Zero point = 0 */
} cg_tensor_quant;

/*============================================================================
 * INT4 KV CACHE
 *============================================================================*/

typedef struct {
    uint8_t* k_cache;           /* Packed INT4 keys [layers, max_seq, n_heads, head_dim/2] */
    uint8_t* v_cache;           /* Packed INT4 values */
    float* k_scales;            /* Per-group scales [layers, max_seq, n_groups] */
    float* v_scales;
    
    int current_len;            /* Current sequence length */
    int max_seq_len;
    int num_layers;
    int n_heads;
    int head_dim;
    int group_size;
} cg_kv_cache_int4;

/*============================================================================
 * QUANTIZATION API
 *============================================================================*/

/**
 * Quantize FP32 tensor to INT4.
 */
cg_tensor_quant* cg_quantize_int4(cg_tensor* tensor, int group_size);

/**
 * Quantize to GPTQ format.
 */
cg_tensor_quant* cg_quantize_gptq(cg_tensor* tensor, cg_tensor* hessian, 
                                   int group_size, int bits);

/**
 * Quantize to AWQ format.
 */
cg_tensor_quant* cg_quantize_awq(cg_tensor* tensor, cg_tensor* activations,
                                  int group_size);

/**
 * Quantize to NF4 (QLoRA).
 */
cg_tensor_quant* cg_quantize_nf4(cg_tensor* tensor, int group_size);

/**
 * Dequantize back to FP32.
 */
cg_tensor* cg_dequantize(cg_tensor_quant* quant);

void cg_tensor_quant_free(cg_tensor_quant* quant);

/*============================================================================
 * QUANTIZED OPERATIONS
 *============================================================================*/

/**
 * INT4 matrix multiplication with on-the-fly dequantization.
 */
void cg_matmul_int4(cg_tensor* input, cg_tensor_quant* weight, cg_tensor* output);

/**
 * INT4 GEMV (for inference).
 */
void cg_gemv_int4(cg_tensor* input, cg_tensor_quant* weight, cg_tensor* output);

/**
 * INT4 attention with quantized KV cache.
 */
void cg_attention_int4_kv(cg_tensor* Q, cg_kv_cache_int4* kv_cache, 
                          int layer_idx, cg_tensor* output);

/*============================================================================
 * KV CACHE API
 *============================================================================*/

/**
 * Create INT4 KV cache.
 */
cg_kv_cache_int4* cg_kv_cache_int4_new(int num_layers, int max_seq_len,
                                        int n_heads, int head_dim, int group_size);

/**
 * Append new K/V to cache (quantizes on-the-fly).
 */
void cg_kv_cache_int4_append(cg_kv_cache_int4* cache, int layer_idx,
                             cg_tensor* new_k, cg_tensor* new_v);

/**
 * Get dequantized K/V for attention.
 */
void cg_kv_cache_int4_get(cg_kv_cache_int4* cache, int layer_idx,
                          cg_tensor* k_out, cg_tensor* v_out);

/**
 * Reset cache for new sequence.
 */
void cg_kv_cache_int4_reset(cg_kv_cache_int4* cache);

void cg_kv_cache_int4_free(cg_kv_cache_int4* cache);

/*============================================================================
 * PACKING/UNPACKING
 *============================================================================*/

/**
 * Pack two INT4 values into one byte.
 */
static inline uint8_t pack_int4(int8_t a, int8_t b) {
    return (uint8_t)((a & 0x0F) | ((b & 0x0F) << 4));
}

/**
 * Unpack INT4 values.
 */
static inline void unpack_int4(uint8_t packed, int8_t* a, int8_t* b) {
    *a = (int8_t)(packed & 0x0F);
    *b = (int8_t)((packed >> 4) & 0x0F);
    /* Sign extend if needed */
    if (*a & 0x08) *a |= 0xF0;
    if (*b & 0x08) *b |= 0xF0;
}

/**
 * Dequantize single INT4 value.
 */
static inline float dequant_int4(int8_t val, float scale, float zero) {
    return ((float)val - zero) * scale;
}

/*============================================================================
 * CALIBRATION
 *============================================================================*/

/**
 * Collect calibration data for quantization.
 */
typedef struct {
    float* min_vals;            /* Per-channel min */
    float* max_vals;            /* Per-channel max */
    float* absmax;              /* Per-channel absmax */
    int num_samples;
    int channels;
} cg_quant_calibration;

cg_quant_calibration* cg_calibration_new(int channels);
void cg_calibration_update(cg_quant_calibration* calib, cg_tensor* activations);
void cg_calibration_compute_scales(cg_quant_calibration* calib, 
                                   float* scales, float* zeros, int group_size);
void cg_calibration_free(cg_quant_calibration* calib);

#endif /* CG_QUANTIZE_H */
