/**
 * Flash Attention v2 - Tiled attention with O(1) memory
 * 
 * Uses online softmax and tiling to avoid materializing NxN attention matrix.
 */

#ifndef CG_FLASH_ATTN_H
#define CG_FLASH_ATTN_H

#include "cg_tensor.h"
#include <stdbool.h>

/*============================================================================
 * CONFIGURATION - AUTO-TUNING
 *============================================================================*/

/* Legacy static defaults (used as fallback) */
#define FLASH_BLOCK_M_DEFAULT 64
#define FLASH_BLOCK_N_DEFAULT 64
#define FLASH_BLOCK_K 64        /* Reduction dimension */
#define FLASH_WARPS 4           /* Warps per block */

/**
 * Runtime Flash Attention configuration.
 * Auto-tuned based on sequence length, head dimension, and hardware.
 */
typedef struct {
    int block_m;            /* Query block size (Br) */
    int block_n;            /* Key/Value block size (Bc) */
    int split_k;            /* Split-K factor for backward pass */
    bool use_persistent;    /* Use persistent kernels for small batches */
    bool use_register_tile; /* Use register tiling for d_head <= 64 */
    int num_warps;          /* Warps per thread block */
    int sm_count;           /* Number of SMs on target GPU */
} cg_flash_attn_config;

/* Backward compatibility macros */
#define FLASH_BLOCK_M FLASH_BLOCK_M_DEFAULT
#define FLASH_BLOCK_N FLASH_BLOCK_N_DEFAULT

typedef enum {
    ATTN_MASK_NONE = 0,
    ATTN_MASK_CAUSAL,           /* Can't attend to future */
    ATTN_MASK_SLIDING_WINDOW,   /* Local attention window */
    ATTN_MASK_CUSTOM            /* User-provided mask */
} cg_attention_mask_type;

/*============================================================================
 * ATTENTION PARAMETERS
 *============================================================================*/

typedef struct {
    /* Input tensors (device pointers) */
    float* Q;                   /* [batch, seqlen_q, n_heads, d_head] */
    float* K;                   /* [batch, seqlen_k, n_heads, d_head] */
    float* V;                   /* [batch, seqlen_k, n_heads, d_head] */
    float* O;                   /* Output [batch, seqlen_q, n_heads, d_head] */
    
    /* Optional */
    float* mask;                /* Custom attention mask */
    float* dropout_mask;        /* Dropout mask for training */
    
    /* Dimensions */
    int batch_size;
    int seqlen_q;
    int seqlen_k;
    int n_heads;
    int n_heads_k;              /* For MQA/GQA: n_heads_k < n_heads */
    int d_head;
    
    /* Scaling */
    float scale;                /* 1/sqrt(d_head) */
    float dropout_prob;
    
    /* Masking */
    cg_attention_mask_type mask_type;
    int window_size;            /* For sliding window attention */
    
    /* Training */
    bool is_training;
    float* softmax_lse;         /* Log-sum-exp for backward [batch, n_heads, seqlen_q] */
    
    /* Backward tensors */
    float* dO;                  /* Gradient of output */
    float* dQ;
    float* dK;
    float* dV;
} cg_flash_attn_params;

/*============================================================================
 * FORWARD API
 *============================================================================*/

/**
 * Flash Attention v2 forward pass.
 * Computes: softmax(Q @ K.T / sqrt(d)) @ V
 * 
 * Memory: O(seqlen) instead of O(seqlen^2)
 */
void cg_flash_attention_forward(cg_flash_attn_params* params);

/**
 * CPU reference implementation (for testing).
 */
void cg_flash_attention_forward_cpu(cg_flash_attn_params* params);

/*============================================================================
 * BACKWARD API
 *============================================================================*/

/**
 * Flash Attention backward pass.
 * Uses recomputation to avoid storing O(N^2) activations.
 */
void cg_flash_attention_backward(cg_flash_attn_params* params);

/**
 * CPU reference implementation.
 */
void cg_flash_attention_backward_cpu(cg_flash_attn_params* params);

/*============================================================================
 * MULTI-QUERY / GROUPED-QUERY ATTENTION
 *============================================================================*/

/**
 * MQA: Multiple query heads share one key/value head.
 */
void cg_flash_attention_mqa(cg_flash_attn_params* params);

/**
 * GQA: Groups of query heads share key/value heads.
 */
void cg_flash_attention_gqa(cg_flash_attn_params* params, int num_kv_groups);

/*============================================================================
 * HELPER FUNCTIONS
 *============================================================================*/

/**
 * Create attention parameters with defaults.
 */
cg_flash_attn_params* cg_flash_attn_params_new(
    cg_tensor* Q, cg_tensor* K, cg_tensor* V,
    cg_attention_mask_type mask_type, float dropout_prob
);

/**
 * Compute optimal configuration based on hardware and shapes.
 * Returns config for tile sizes, split-K, and persistent mode.
 */
cg_flash_attn_config cg_flash_attn_autotune(int seqlen_q, int seqlen_k, 
                                              int d_head, int sm_count);

/**
 * Compute shared memory requirement.
 */
size_t cg_flash_attn_shared_mem(int block_m, int block_n, int d_head);

void cg_flash_attn_params_free(cg_flash_attn_params* params);

/*============================================================================
 * ROTARY POSITIONAL EMBEDDING (RoPE)
 *============================================================================*/

/**
 * Apply RoPE to Q and K before attention.
 */
void cg_apply_rope(float* Q, float* K, int seqlen, int n_heads, 
                   int d_head, int position_offset, float rope_theta);

/**
 * Precompute RoPE frequencies.
 */
void cg_rope_precompute_freqs(float* cos_cache, float* sin_cache,
                              int max_seqlen, int d_head, float theta);

/*============================================================================
 * ALIBI POSITIONAL BIAS
 *============================================================================*/

/**
 * Compute ALiBi bias matrix.
 */
void cg_alibi_bias(float* bias, int seqlen_q, int seqlen_k, 
                   int n_heads, float* slopes);

/**
 * Compute ALiBi slopes for n_heads.
 */
void cg_alibi_slopes(float* slopes, int n_heads);

#endif /* CG_FLASH_ATTN_H */
