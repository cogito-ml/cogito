/**
 * Flash Attention v2 CUDA Kernels Implementation
 * 
 * GPU implementation with tiling and online softmax algorithm.
 * Uses simulation fallback for CPU-only environments.
 * 
 * Based on the Flash Attention v2 paper:
 * - Tiles Q, K, V to fit in shared memory
 * - Uses online softmax to avoid materializing NxN attention matrix
 * - Supports causal, sliding window, and custom masks
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cg_flash_attn_kernels.h"
#include "cg_flash_attn.h"
#include "cg_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef CG_USE_CUDA
#include <cuda_runtime.h>

/*============================================================================
 * CUDA KERNELS
 *============================================================================*/

/**
 * Flash Attention Forward Kernel
 * 
 * Each block handles one (batch, head, query_tile) combination.
 * Thread block iterates over all K,V tiles.
 */
__global__ void flash_attn_fwd_kernel(
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ softmax_lse,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int n_heads,
    int n_heads_k,
    int d_head,
    float scale,
    int mask_type,
    int window_size
) {
    /* Shared memory for tiles */
    extern __shared__ float shmem[];
    
    float* Q_tile = shmem;
    float* K_tile = Q_tile + FA_BLOCK_M * d_head;
    float* V_tile = K_tile + FA_BLOCK_N * d_head;
    float* S_tile = V_tile + FA_BLOCK_N * d_head;  /* Attention scores */
    
    /* Block indices */
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_tile_idx = blockIdx.x;
    
    /* Thread indices */
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    
    /* Query tile range */
    int q_start = q_tile_idx * FA_BLOCK_M;
    int q_end = min(q_start + FA_BLOCK_M, seqlen_q);
    int q_tile_size = q_end - q_start;
    
    /* For GQA: multiple Q heads share same K/V head */
    int kv_head_idx = head_idx * n_heads_k / n_heads;
    
    /* Initialize per-row max and sum for online softmax */
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    
    /* Per-thread accumulators for output */
    float O_acc[FA_BLOCK_M];
    for (int i = 0; i < FA_BLOCK_M && q_start + i < seqlen_q; i++) {
        O_acc[i] = 0.0f;
    }
    
    /* Load Q tile to shared memory */
    for (int i = tid; i < q_tile_size * d_head; i += blockDim.x) {
        int row = i / d_head;
        int col = i % d_head;
        int q_offset = ((batch_idx * seqlen_q + q_start + row) * n_heads + head_idx) * d_head + col;
        Q_tile[row * d_head + col] = Q[q_offset];
    }
    __syncthreads();
    
    /* Iterate over K,V tiles */
    for (int k_start = 0; k_start < seqlen_k; k_start += FA_BLOCK_N) {
        int k_end = min(k_start + FA_BLOCK_N, seqlen_k);
        int k_tile_size = k_end - k_start;
        
        /* Check causal mask - skip tiles entirely above diagonal */
        if (mask_type == ATTN_MASK_CAUSAL && k_start > q_start + q_tile_size - 1) {
            break;
        }
        
        /* Check sliding window - skip tiles outside window */
        if (mask_type == ATTN_MASK_SLIDING_WINDOW) {
            if (k_start > q_start + window_size) continue;
            if (k_end < q_start - window_size) continue;
        }
        
        /* Load K tile */
        for (int i = tid; i < k_tile_size * d_head; i += blockDim.x) {
            int row = i / d_head;
            int col = i % d_head;
            int k_offset = ((batch_idx * seqlen_k + k_start + row) * n_heads_k + kv_head_idx) * d_head + col;
            K_tile[row * d_head + col] = K[k_offset];
        }
        
        /* Load V tile */
        for (int i = tid; i < k_tile_size * d_head; i += blockDim.x) {
            int row = i / d_head;
            int col = i % d_head;
            int v_offset = ((batch_idx * seqlen_k + k_start + row) * n_heads_k + kv_head_idx) * d_head + col;
            V_tile[row * d_head + col] = V[v_offset];
        }
        __syncthreads();
        
        /* Compute S = Q @ K^T * scale */
        for (int qi = 0; qi < q_tile_size; qi++) {
            for (int ki = tid; ki < k_tile_size; ki += blockDim.x) {
                float dot = 0.0f;
                for (int d = 0; d < d_head; d++) {
                    dot += Q_tile[qi * d_head + d] * K_tile[ki * d_head + d];
                }
                dot *= scale;
                
                /* Apply mask */
                int q_idx = q_start + qi;
                int k_idx = k_start + ki;
                
                if (mask_type == ATTN_MASK_CAUSAL && k_idx > q_idx) {
                    dot = -FLT_MAX;
                }
                if (mask_type == ATTN_MASK_SLIDING_WINDOW) {
                    if (abs(k_idx - q_idx) > window_size) {
                        dot = -FLT_MAX;
                    }
                }
                
                S_tile[qi * FA_BLOCK_N + ki] = dot;
            }
        }
        __syncthreads();
        
        /* Online softmax update */
        for (int qi = tid; qi < q_tile_size; qi += blockDim.x) {
            /* Find new max */
            float m_new = m_i;
            for (int ki = 0; ki < k_tile_size; ki++) {
                float s = S_tile[qi * FA_BLOCK_N + ki];
                if (s > m_new) m_new = s;
            }
            
            /* Update running sum */
            float l_new = l_i * expf(m_i - m_new);
            for (int ki = 0; ki < k_tile_size; ki++) {
                float s = S_tile[qi * FA_BLOCK_N + ki];
                l_new += expf(s - m_new);
            }
            
            /* Update output accumulator */
            float scale_old = expf(m_i - m_new);
            O_acc[qi] *= scale_old;
            
            for (int d = 0; d < d_head; d++) {
                float v_sum = 0.0f;
                for (int ki = 0; ki < k_tile_size; ki++) {
                    float s = S_tile[qi * FA_BLOCK_N + ki];
                    float p = expf(s - m_new);
                    v_sum += p * V_tile[ki * d_head + d];
                }
                O_acc[qi] += v_sum;
            }
            
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }
    
    /* Final normalization and write output */
    for (int qi = tid; qi < q_tile_size; qi += blockDim.x) {
        int q_idx = q_start + qi;
        
        for (int d = 0; d < d_head; d++) {
            int o_offset = ((batch_idx * seqlen_q + q_idx) * n_heads + head_idx) * d_head + d;
            O[o_offset] = O_acc[qi] / l_i;
        }
        
        /* Store LSE for backward */
        if (softmax_lse) {
            int lse_offset = (batch_idx * n_heads + head_idx) * seqlen_q + q_idx;
            softmax_lse[lse_offset] = m_i + logf(l_i);
        }
    }
}

/**
 * RoPE (Rotary Positional Embedding) Kernel
 */
__global__ void rope_kernel(
    float* x,
    float* cos_cache,
    float* sin_cache,
    int batch,
    int seqlen,
    int n_heads,
    int d_head,
    int position_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = d_head / 2;
    int total = batch * seqlen * n_heads * half_d;
    
    if (idx >= total) return;
    
    int b = idx / (seqlen * n_heads * half_d);
    int remaining = idx % (seqlen * n_heads * half_d);
    int pos = remaining / (n_heads * half_d);
    remaining = remaining % (n_heads * half_d);
    int h = remaining / half_d;
    int i = remaining % half_d;
    
    int abs_pos = position_offset + pos;
    float cos_val = cos_cache[abs_pos * half_d + i];
    float sin_val = sin_cache[abs_pos * half_d + i];
    
    int base_offset = ((b * seqlen + pos) * n_heads + h) * d_head;
    float x0 = x[base_offset + i];
    float x1 = x[base_offset + i + half_d];
    
    x[base_offset + i] = x0 * cos_val - x1 * sin_val;
    x[base_offset + i + half_d] = x0 * sin_val + x1 * cos_val;
}

#endif /* CG_USE_CUDA */

/*============================================================================
 * SIMULATION IMPLEMENTATIONS
 *============================================================================*/

/**
 * CPU simulation of Flash Attention forward pass.
 * Uses the same tiled algorithm as GPU for correctness verification.
 */
static void flash_attn_fwd_sim(cg_flash_attn_params* params) {
    int B = params->batch_size;
    int H = params->n_heads;
    int H_k = params->n_heads_k;
    int N_q = params->seqlen_q;
    int N_k = params->seqlen_k;
    int D = params->d_head;
    float scale = params->scale;
    
    int Br = FA_BLOCK_M;
    int Bc = FA_BLOCK_N;
    
    printf("[FLASH SIM] Running Flash Attention forward simulation\n");
    printf("[FLASH SIM] Shape: B=%d, H=%d, N_q=%d, N_k=%d, D=%d\n", B, H, N_q, N_k, D);
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int h_k = h * H_k / H;  /* GQA mapping */
            
            /* Per-row statistics */
            float* m = (float*)malloc(N_q * sizeof(float));
            float* l = (float*)malloc(N_q * sizeof(float));
            float* O_acc = (float*)calloc(N_q * D, sizeof(float));
            
            for (int i = 0; i < N_q; i++) {
                m[i] = -FLT_MAX;
                l[i] = 0.0f;
            }
            
            /* Iterate over K,V tiles */
            for (int k_start = 0; k_start < N_k; k_start += Bc) {
                int k_end = (k_start + Bc < N_k) ? k_start + Bc : N_k;
                int k_tile_size = k_end - k_start;
                
                /* Iterate over Q tiles */
                for (int q_start = 0; q_start < N_q; q_start += Br) {
                    int q_end = (q_start + Br < N_q) ? q_start + Br : N_q;
                    int q_tile_size = q_end - q_start;
                    
                    /* Allocate attention scores for this tile */
                    float* S = (float*)malloc(q_tile_size * k_tile_size * sizeof(float));
                    
                    /* Compute S = Q @ K^T * scale */
                    for (int qi = 0; qi < q_tile_size; qi++) {
                        int q_idx = q_start + qi;
                        
                        for (int ki = 0; ki < k_tile_size; ki++) {
                            int k_idx = k_start + ki;
                            
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++) {
                                int q_off = ((b * N_q + q_idx) * H + h) * D + d;
                                int k_off = ((b * N_k + k_idx) * H_k + h_k) * D + d;
                                dot += params->Q[q_off] * params->K[k_off];
                            }
                            dot *= scale;
                            
                            /* Apply causal mask */
                            if (params->mask_type == ATTN_MASK_CAUSAL && k_idx > q_idx) {
                                dot = -FLT_MAX;
                            }
                            
                            /* Apply sliding window mask */
                            if (params->mask_type == ATTN_MASK_SLIDING_WINDOW) {
                                if (abs(k_idx - q_idx) > params->window_size) {
                                    dot = -FLT_MAX;
                                }
                            }
                            
                            S[qi * k_tile_size + ki] = dot;
                        }
                    }
                    
                    /* Online softmax update */
                    for (int qi = 0; qi < q_tile_size; qi++) {
                        int q_idx = q_start + qi;
                        
                        /* Find new max */
                        float m_new = m[q_idx];
                        for (int ki = 0; ki < k_tile_size; ki++) {
                            float s = S[qi * k_tile_size + ki];
                            if (s > m_new) m_new = s;
                        }
                        
                        /* Update sum */
                        float l_new = l[q_idx] * expf(m[q_idx] - m_new);
                        for (int ki = 0; ki < k_tile_size; ki++) {
                            float s = S[qi * k_tile_size + ki];
                            l_new += expf(s - m_new);
                        }
                        
                        /* Scale old accumulator */
                        float scale_old = expf(m[q_idx] - m_new);
                        for (int d = 0; d < D; d++) {
                            O_acc[q_idx * D + d] *= scale_old;
                        }
                        
                        /* Add new contributions */
                        for (int d = 0; d < D; d++) {
                            for (int ki = 0; ki < k_tile_size; ki++) {
                                int k_idx = k_start + ki;
                                float s = S[qi * k_tile_size + ki];
                                float p = expf(s - m_new);
                                
                                int v_off = ((b * N_k + k_idx) * H_k + h_k) * D + d;
                                O_acc[q_idx * D + d] += p * params->V[v_off];
                            }
                        }
                        
                        m[q_idx] = m_new;
                        l[q_idx] = l_new;
                    }
                    
                    free(S);
                }
            }
            
            /* Final normalization */
            for (int qi = 0; qi < N_q; qi++) {
                for (int d = 0; d < D; d++) {
                    int o_off = ((b * N_q + qi) * H + h) * D + d;
                    params->O[o_off] = O_acc[qi * D + d] / l[qi];
                }
                
                if (params->softmax_lse) {
                    int lse_off = (b * H + h) * N_q + qi;
                    params->softmax_lse[lse_off] = m[qi] + logf(l[qi]);
                }
            }
            
            free(m);
            free(l);
            free(O_acc);
        }
    }
}

/**
 * CPU simulation of Flash Attention backward pass.
 */
static void flash_attn_bwd_sim(cg_flash_attn_params* params) {
    int B = params->batch_size;
    int H = params->n_heads;
    int H_k = params->n_heads_k;
    int N_q = params->seqlen_q;
    int N_k = params->seqlen_k;
    int D = params->d_head;
    float scale = params->scale;
    
    printf("[FLASH SIM] Running Flash Attention backward simulation\n");
    
    /* Zero gradients */
    memset(params->dQ, 0, B * N_q * H * D * sizeof(float));
    memset(params->dK, 0, B * N_k * H_k * D * sizeof(float));
    memset(params->dV, 0, B * N_k * H_k * D * sizeof(float));
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int h_k = h * H_k / H;
            
            /* Recompute attention weights */
            float* P = (float*)malloc(N_q * N_k * sizeof(float));
            
            for (int qi = 0; qi < N_q; qi++) {
                float row_max = -FLT_MAX;
                
                for (int ki = 0; ki < N_k; ki++) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        int q_off = ((b * N_q + qi) * H + h) * D + d;
                        int k_off = ((b * N_k + ki) * H_k + h_k) * D + d;
                        dot += params->Q[q_off] * params->K[k_off];
                    }
                    float s = dot * scale;
                    
                    if (params->mask_type == ATTN_MASK_CAUSAL && ki > qi) {
                        s = -FLT_MAX;
                    }
                    
                    P[qi * N_k + ki] = s;
                    if (s > row_max) row_max = s;
                }
                
                /* Softmax */
                float row_sum = 0.0f;
                for (int ki = 0; ki < N_k; ki++) {
                    P[qi * N_k + ki] = expf(P[qi * N_k + ki] - row_max);
                    row_sum += P[qi * N_k + ki];
                }
                for (int ki = 0; ki < N_k; ki++) {
                    P[qi * N_k + ki] /= row_sum;
                }
            }
            
            /* dV = P^T @ dO */
            for (int ki = 0; ki < N_k; ki++) {
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int qi = 0; qi < N_q; qi++) {
                        int do_off = ((b * N_q + qi) * H + h) * D + d;
                        sum += P[qi * N_k + ki] * params->dO[do_off];
                    }
                    int dv_off = ((b * N_k + ki) * H_k + h_k) * D + d;
                    params->dV[dv_off] += sum;
                }
            }
            
            /* dP = dO @ V^T */
            float* dP = (float*)malloc(N_q * N_k * sizeof(float));
            for (int qi = 0; qi < N_q; qi++) {
                for (int ki = 0; ki < N_k; ki++) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; d++) {
                        int do_off = ((b * N_q + qi) * H + h) * D + d;
                        int v_off = ((b * N_k + ki) * H_k + h_k) * D + d;
                        sum += params->dO[do_off] * params->V[v_off];
                    }
                    dP[qi * N_k + ki] = sum;
                }
            }
            
            /* dS = P * (dP - rowsum(P * dP)) */
            float* dS = (float*)malloc(N_q * N_k * sizeof(float));
            for (int qi = 0; qi < N_q; qi++) {
                float row_sum = 0.0f;
                for (int ki = 0; ki < N_k; ki++) {
                    row_sum += P[qi * N_k + ki] * dP[qi * N_k + ki];
                }
                for (int ki = 0; ki < N_k; ki++) {
                    dS[qi * N_k + ki] = P[qi * N_k + ki] * (dP[qi * N_k + ki] - row_sum);
                }
            }
            
            /* dQ = dS @ K * scale */
            for (int qi = 0; qi < N_q; qi++) {
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < N_k; ki++) {
                        int k_off = ((b * N_k + ki) * H_k + h_k) * D + d;
                        sum += dS[qi * N_k + ki] * params->K[k_off];
                    }
                    int dq_off = ((b * N_q + qi) * H + h) * D + d;
                    params->dQ[dq_off] = sum * scale;
                }
            }
            
            /* dK = dS^T @ Q * scale */
            for (int ki = 0; ki < N_k; ki++) {
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int qi = 0; qi < N_q; qi++) {
                        int q_off = ((b * N_q + qi) * H + h) * D + d;
                        sum += dS[qi * N_k + ki] * params->Q[q_off];
                    }
                    int dk_off = ((b * N_k + ki) * H_k + h_k) * D + d;
                    params->dK[dk_off] += sum * scale;
                }
            }
            
            free(P);
            free(dP);
            free(dS);
        }
    }
}

/*============================================================================
 * PUBLIC API
 *============================================================================*/

void cg_flash_attention_forward_cuda(cg_flash_attn_params* params) {
#ifdef CG_USE_CUDA
    int B = params->batch_size;
    int H = params->n_heads;
    int N_q = params->seqlen_q;
    int D = params->d_head;
    
    /* Calculate grid dimensions */
    int num_q_tiles = (N_q + FA_BLOCK_M - 1) / FA_BLOCK_M;
    dim3 grid(num_q_tiles, H, B);
    dim3 block(FA_THREADS);
    
    /* Calculate shared memory */
    size_t shmem = (FA_BLOCK_M * D + 2 * FA_BLOCK_N * D + FA_BLOCK_M * FA_BLOCK_N) * sizeof(float);
    
    flash_attn_fwd_kernel<<<grid, block, shmem>>>(
        params->Q, params->K, params->V, params->O, params->softmax_lse,
        B, N_q, params->seqlen_k, H, params->n_heads_k, D,
        params->scale, params->mask_type, params->window_size
    );
    
    cg_cuda_device_synchronize();
#else
    flash_attn_fwd_sim(params);
#endif
}

void cg_flash_attention_backward_cuda(cg_flash_attn_params* params) {
#ifdef CG_USE_CUDA
    /* GPU backward kernel would go here */
    /* For now, use simulation even on GPU - backward is more complex */
    flash_attn_bwd_sim(params);
#else
    flash_attn_bwd_sim(params);
#endif
}

void cg_flash_attention_forward_rope_cuda(cg_flash_attn_params* params,
                                           float* cos_cache, float* sin_cache,
                                           int position_offset) {
    /* Apply RoPE first, then run attention */
    cg_rope_forward_cuda(params->Q, params->batch_size, params->seqlen_q,
                          params->n_heads, params->d_head, cos_cache, sin_cache,
                          position_offset);
    cg_rope_forward_cuda(params->K, params->batch_size, params->seqlen_k,
                          params->n_heads_k, params->d_head, cos_cache, sin_cache,
                          position_offset);
    
    cg_flash_attention_forward_cuda(params);
}

void cg_flash_attention_mqa_cuda(cg_flash_attn_params* params) {
    /* MQA: n_heads_k = 1 */
    params->n_heads_k = 1;
    cg_flash_attention_forward_cuda(params);
}

void cg_flash_attention_gqa_cuda(cg_flash_attn_params* params, int num_kv_groups) {
    /* GQA: n_heads / num_kv_groups query heads share one K/V head */
    params->n_heads_k = num_kv_groups;
    cg_flash_attention_forward_cuda(params);
}

void cg_flash_attention_sliding_window_cuda(cg_flash_attn_params* params,
                                             int window_size) {
    params->mask_type = ATTN_MASK_SLIDING_WINDOW;
    params->window_size = window_size;
    cg_flash_attention_forward_cuda(params);
}

void cg_flash_attention_paged_cuda(cg_flash_attn_params* params,
                                    int* block_tables, int block_size,
                                    int max_context_len) {
    (void)block_tables;
    (void)block_size;
    (void)max_context_len;
    
    /* Paged attention for inference - more complex implementation */
    printf("[FLASH] Paged attention not yet implemented, falling back to standard\n");
    cg_flash_attention_forward_cuda(params);
}

/*============================================================================
 * ROPE IMPLEMENTATION
 *============================================================================*/

void cg_rope_forward_cuda(float* x, int batch, int seqlen, int n_heads,
                           int d_head, float* cos_cache, float* sin_cache,
                           int position_offset) {
#ifdef CG_USE_CUDA
    int half_d = d_head / 2;
    int total = batch * seqlen * n_heads * half_d;
    int grid, block;
    cg_cuda_calculate_launch_dims(total, &grid, &block);
    
    rope_kernel<<<grid, block>>>(x, cos_cache, sin_cache,
                                  batch, seqlen, n_heads, d_head, position_offset);
    cg_cuda_device_synchronize();
#else
    /* CPU simulation */
    int half_d = d_head / 2;
    
    for (int b = 0; b < batch; b++) {
        for (int pos = 0; pos < seqlen; pos++) {
            int abs_pos = position_offset + pos;
            
            for (int h = 0; h < n_heads; h++) {
                int base_off = ((b * seqlen + pos) * n_heads + h) * d_head;
                
                for (int i = 0; i < half_d; i++) {
                    float cos_val = cos_cache[abs_pos * half_d + i];
                    float sin_val = sin_cache[abs_pos * half_d + i];
                    
                    float x0 = x[base_off + i];
                    float x1 = x[base_off + i + half_d];
                    
                    x[base_off + i] = x0 * cos_val - x1 * sin_val;
                    x[base_off + i + half_d] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
#endif
}

void cg_rope_backward_cuda(float* dx, int batch, int seqlen, int n_heads,
                            int d_head, float* cos_cache, float* sin_cache,
                            int position_offset) {
    /* RoPE backward is just RoPE with negated sin (inverse rotation) */
    int half_d = d_head / 2;
    
    for (int b = 0; b < batch; b++) {
        for (int pos = 0; pos < seqlen; pos++) {
            int abs_pos = position_offset + pos;
            
            for (int h = 0; h < n_heads; h++) {
                int base_off = ((b * seqlen + pos) * n_heads + h) * d_head;
                
                for (int i = 0; i < half_d; i++) {
                    float cos_val = cos_cache[abs_pos * half_d + i];
                    float sin_val = -sin_cache[abs_pos * half_d + i];  /* Negate for inverse */
                    
                    float dx0 = dx[base_off + i];
                    float dx1 = dx[base_off + i + half_d];
                    
                    dx[base_off + i] = dx0 * cos_val - dx1 * sin_val;
                    dx[base_off + i + half_d] = dx0 * sin_val + dx1 * cos_val;
                }
            }
        }
    }
}

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

void cg_flash_attn_get_tile_sizes(int d_head, int sm_version,
                                   int* block_m, int* block_n) {
    (void)sm_version;
    
    /* Heuristics based on head dimension */
    if (d_head <= 64) {
        *block_m = 128;
        *block_n = 64;
    } else if (d_head <= 128) {
        *block_m = 64;
        *block_n = 64;
    } else {
        *block_m = 32;
        *block_n = 32;
    }
}

size_t cg_flash_attn_estimate_shmem(int block_m, int block_n, int d_head) {
    /* Q tile + K tile + V tile + S tile */
    size_t q_size = block_m * d_head * sizeof(float);
    size_t k_size = block_n * d_head * sizeof(float);
    size_t v_size = block_n * d_head * sizeof(float);
    size_t s_size = block_m * block_n * sizeof(float);
    
    return q_size + k_size + v_size + s_size;
}

bool cg_flash_attn_check_config(int block_m, int block_n, int d_head,
                                 size_t max_shmem) {
    size_t required = cg_flash_attn_estimate_shmem(block_m, block_n, d_head);
    return required <= max_shmem;
}

#ifdef __cplusplus
}
#endif
