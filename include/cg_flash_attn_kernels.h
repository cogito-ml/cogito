/**
 * Flash Attention v2 CUDA Kernels
 * 
 * GPU implementation of Flash Attention using tiling and online softmax.
 * Includes simulation mode for CPU-only environments.
 */

#ifndef CG_FLASH_ATTN_KERNELS_H
#define CG_FLASH_ATTN_KERNELS_H

#include "cg_tensor.h"
#include "cg_flash_attn.h"
#include "cg_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * FLASH ATTENTION CONFIGURATION
 *============================================================================*/

/* Tile sizes for GPU execution */
#define FA_BLOCK_M 64        /* Query tile size */
#define FA_BLOCK_N 64        /* Key/Value tile size */
#define FA_BLOCK_K 32        /* Head dimension tile */

/* Thread configuration */
#define FA_THREADS 256       /* Threads per block */
#define FA_WARPS 8           /* Warps per block (256/32) */

/*============================================================================
 * FORWARD PASS
 *============================================================================*/

/**
 * Flash Attention v2 Forward (GPU optimized).
 * 
 * Uses tiled computation to never materialize the NxN attention matrix.
 * Memory complexity: O(N) instead of O(N^2).
 * 
 * Algorithm:
 *   for each Q block:
 *     Load Q_tile to shared memory
 *     Initialize O_acc = 0, m = -inf, l = 0
 *     for each K,V block:
 *       Load K_tile, V_tile to shared memory
 *       S = Q_tile @ K_tile^T * scale (in registers)
 *       Apply causal mask if needed
 *       m_new = max(m, rowmax(S))
 *       P = exp(S - m_new)
 *       l_new = l * exp(m - m_new) + rowsum(P)
 *       O_acc = O_acc * exp(m - m_new) + P @ V_tile
 *       m = m_new, l = l_new
 *     O_out = O_acc / l
 */
void cg_flash_attention_forward_cuda(cg_flash_attn_params* params);

/**
 * Flash Attention forward with fused RoPE.
 */
void cg_flash_attention_forward_rope_cuda(cg_flash_attn_params* params,
                                           float* cos_cache, float* sin_cache,
                                           int position_offset);

/*============================================================================
 * BACKWARD PASS
 *============================================================================*/

/**
 * Flash Attention v2 Backward (GPU optimized).
 * 
 * Uses recomputation to avoid storing O(N^2) activations.
 * Only stores: O (output), LSE (log-sum-exp per row)
 * 
 * Algorithm:
 *   Recompute P on-the-fly during backward
 *   dV = P^T @ dO
 *   dP = dO @ V^T
 *   dS = P * (dP - rowsum(P * dP))
 *   dQ = dS @ K * scale
 *   dK = dS^T @ Q * scale
 */
void cg_flash_attention_backward_cuda(cg_flash_attn_params* params);

/*============================================================================
 * SPECIALIZED VARIANTS
 *============================================================================*/

/**
 * Multi-Query Attention (MQA).
 * Multiple query heads share one K/V head.
 */
void cg_flash_attention_mqa_cuda(cg_flash_attn_params* params);

/**
 * Grouped-Query Attention (GQA).
 * Groups of query heads share K/V heads.
 */
void cg_flash_attention_gqa_cuda(cg_flash_attn_params* params, int num_kv_groups);

/**
 * Sliding Window Attention.
 * Each query only attends to a local window of keys.
 */
void cg_flash_attention_sliding_window_cuda(cg_flash_attn_params* params,
                                             int window_size);

/**
 * Paged Attention for KV cache.
 * For efficient inference with non-contiguous KV cache pages.
 */
void cg_flash_attention_paged_cuda(cg_flash_attn_params* params,
                                    int* block_tables, int block_size,
                                    int max_context_len);

/*============================================================================
 * ROPE KERNELS
 *============================================================================*/

/**
 * Apply Rotary Position Embedding (GPU).
 */
void cg_rope_forward_cuda(float* x, int batch, int seqlen, int n_heads,
                           int d_head, float* cos_cache, float* sin_cache,
                           int position_offset);

/**
 * RoPE backward pass.
 */
void cg_rope_backward_cuda(float* dx, int batch, int seqlen, int n_heads,
                            int d_head, float* cos_cache, float* sin_cache,
                            int position_offset);

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

/**
 * Get recommended tile sizes based on head dimension and GPU type.
 */
void cg_flash_attn_get_tile_sizes(int d_head, int sm_version,
                                   int* block_m, int* block_n);

/**
 * Estimate shared memory usage for given configuration.
 */
size_t cg_flash_attn_estimate_shmem(int block_m, int block_n, int d_head);

/**
 * Check if configuration fits in GPU constraints.
 */
bool cg_flash_attn_check_config(int block_m, int block_n, int d_head,
                                 size_t max_shmem);

#ifdef __cplusplus
}
#endif

#endif /* CG_FLASH_ATTN_KERNELS_H */
