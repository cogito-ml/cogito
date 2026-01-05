/**
 * Flash Attention v2 - CPU Reference
 *
 * Uses tiled algorithm with online softmax for O(N) memory.
 */

#include "cg_flash_attn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*============================================================================
 * FLASH ATTENTION FORWARD (CPU)
 *============================================================================*/

void cg_flash_attention_forward_cpu(cg_flash_attn_params* params) {
    int B = params->batch_size;
    int H = params->n_heads;
    int N_q = params->seqlen_q;
    int N_k = params->seqlen_k;
    int D = params->d_head;
    float scale = params->scale;
    
    /* Block sizes */
    int Br = FLASH_BLOCK_M;  /* Query block size */
    int Bc = FLASH_BLOCK_N;  /* Key block size */
    
    /* For each batch and head */
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            
            /* Per-row statistics for online softmax */
            float* m = (float*)malloc(N_q * sizeof(float));  /* Row max */
            float* l = (float*)malloc(N_q * sizeof(float));  /* Row sum */
            float* O_acc = (float*)calloc(N_q * D, sizeof(float));  /* Accumulator */
            
            /* Initialize */
            for (int i = 0; i < N_q; i++) {
                m[i] = -FLT_MAX;
                l[i] = 0.0f;
            }
            
            /* Iterate over key blocks */
            for (int k_start = 0; k_start < N_k; k_start += Bc) {
                int k_end = (k_start + Bc < N_k) ? k_start + Bc : N_k;
                int k_block_size = k_end - k_start;
                
                /* Iterate over query blocks */
                for (int q_start = 0; q_start < N_q; q_start += Br) {
                    int q_end = (q_start + Br < N_q) ? q_start + Br : N_q;
                    int q_block_size = q_end - q_start;
                    
                    /* Compute S = Q_block @ K_block^T * scale */
                    float* S = (float*)malloc(q_block_size * k_block_size * sizeof(float));
                    
                    for (int qi = 0; qi < q_block_size; qi++) {
                        int q_idx = q_start + qi;
                        
                        for (int ki = 0; ki < k_block_size; ki++) {
                            int k_idx = k_start + ki;
                            
                            /* Dot product Q[q_idx] @ K[k_idx] */
                            float dot = 0.0f;
                            for (int d = 0; d < D; d++) {
                                int q_offset = ((b * N_q + q_idx) * H + h) * D + d;
                                int k_offset = ((b * N_k + k_idx) * H + h) * D + d;
                                dot += params->Q[q_offset] * params->K[k_offset];
                            }
                            
                            S[qi * k_block_size + ki] = dot * scale;
                        }
                    }
                    
                    /* Apply causal mask if needed */
                    if (params->mask_type == ATTN_MASK_CAUSAL) {
                        for (int qi = 0; qi < q_block_size; qi++) {
                            for (int ki = 0; ki < k_block_size; ki++) {
                                int q_idx = q_start + qi;
                                int k_idx = k_start + ki;
                                
                                if (k_idx > q_idx) {
                                    S[qi * k_block_size + ki] = -FLT_MAX;
                                }
                            }
                        }
                    }
                    
                    /* Online softmax: update m, l, O */
                    for (int qi = 0; qi < q_block_size; qi++) {
                        int q_idx = q_start + qi;
                        
                        /* Find new max */
                        float m_new = m[q_idx];
                        for (int ki = 0; ki < k_block_size; ki++) {
                            float s = S[qi * k_block_size + ki];
                            if (s > m_new) m_new = s;
                        }
                        
                        /* Compute exp(S - m_new) and sum */
                        float l_new = l[q_idx] * expf(m[q_idx] - m_new);
                        for (int ki = 0; ki < k_block_size; ki++) {
                            float s = S[qi * k_block_size + ki];
                            l_new += expf(s - m_new);
                        }
                        
                        /* Update output accumulator */
                        float scale_old = expf(m[q_idx] - m_new);
                        
                        for (int d = 0; d < D; d++) {
                            /* Scale old accumulator */
                            O_acc[q_idx * D + d] *= scale_old;
                            
                            /* Add new contribution */
                            for (int ki = 0; ki < k_block_size; ki++) {
                                int k_idx = k_start + ki;
                                float s = S[qi * k_block_size + ki];
                                float p = expf(s - m_new);
                                
                                int v_offset = ((b * N_k + k_idx) * H + h) * D + d;
                                O_acc[q_idx * D + d] += p * params->V[v_offset];
                            }
                        }
                        
                        m[q_idx] = m_new;
                        l[q_idx] = l_new;
                    }
                    
                    free(S);
                }
            }
            
            /* Final normalization: O = O_acc / l */
            for (int qi = 0; qi < N_q; qi++) {
                for (int d = 0; d < D; d++) {
                    int o_offset = ((b * N_q + qi) * H + h) * D + d;
                    params->O[o_offset] = O_acc[qi * D + d] / l[qi];
                }
                
                /* Store LSE for backward */
                if (params->softmax_lse) {
                    int lse_offset = (b * H + h) * N_q + qi;
                    params->softmax_lse[lse_offset] = m[qi] + logf(l[qi]);
                }
            }
            
            free(m);
            free(l);
            free(O_acc);
        }
    }
}

/*============================================================================
 * FLASH ATTENTION BACKWARD (CPU)
 *============================================================================*/

void cg_flash_attention_backward_cpu(cg_flash_attn_params* params) {
    int B = params->batch_size;
    int H = params->n_heads;
    int N_q = params->seqlen_q;
    int N_k = params->seqlen_k;
    int D = params->d_head;
    float scale = params->scale;
    
    /* Zero gradients */
    memset(params->dQ, 0, B * N_q * H * D * sizeof(float));
    memset(params->dK, 0, B * N_k * H * D * sizeof(float));
    memset(params->dV, 0, B * N_k * H * D * sizeof(float));
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            
            /* Recompute attention weights (forward pass) */
            float* P = (float*)malloc(N_q * N_k * sizeof(float));
            
            /* Compute attention scores S = Q @ K^T * scale */
            for (int qi = 0; qi < N_q; qi++) {
                float row_max = -FLT_MAX;
                
                for (int ki = 0; ki < N_k; ki++) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        int q_off = ((b * N_q + qi) * H + h) * D + d;
                        int k_off = ((b * N_k + ki) * H + h) * D + d;
                        dot += params->Q[q_off] * params->K[k_off];
                    }
                    
                    float s = dot * scale;
                    
                    /* Causal mask */
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
            
            /* Backward pass */
            /* dV = P^T @ dO */
            for (int ki = 0; ki < N_k; ki++) {
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int qi = 0; qi < N_q; qi++) {
                        int do_off = ((b * N_q + qi) * H + h) * D + d;
                        sum += P[qi * N_k + ki] * params->dO[do_off];
                    }
                    int dv_off = ((b * N_k + ki) * H + h) * D + d;
                    params->dV[dv_off] = sum;
                }
            }
            
            /* dP = dO @ V^T */
            float* dP = (float*)malloc(N_q * N_k * sizeof(float));
            for (int qi = 0; qi < N_q; qi++) {
                for (int ki = 0; ki < N_k; ki++) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; d++) {
                        int do_off = ((b * N_q + qi) * H + h) * D + d;
                        int v_off = ((b * N_k + ki) * H + h) * D + d;
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
                        int k_off = ((b * N_k + ki) * H + h) * D + d;
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
                    int dk_off = ((b * N_k + ki) * H + h) * D + d;
                    params->dK[dk_off] = sum * scale;
                }
            }
            
            free(P);
            free(dP);
            free(dS);
        }
    }
}

/*============================================================================
 * ROTARY POSITIONAL EMBEDDING (RoPE)
 *============================================================================*/

void cg_rope_precompute_freqs(float* cos_cache, float* sin_cache,
                              int max_seqlen, int d_head, float theta) {
    for (int pos = 0; pos < max_seqlen; pos++) {
        for (int i = 0; i < d_head / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / d_head);
            float angle = pos * freq;
            cos_cache[pos * (d_head / 2) + i] = cosf(angle);
            sin_cache[pos * (d_head / 2) + i] = sinf(angle);
        }
    }
}

void cg_apply_rope(float* Q, float* K, int seqlen, int n_heads, 
                   int d_head, int position_offset, float rope_theta) {
    /* Precompute freqs for this range */
    int half_d = d_head / 2;
    
    for (int pos = 0; pos < seqlen; pos++) {
        int abs_pos = position_offset + pos;
        
        for (int h = 0; h < n_heads; h++) {
            float* q = Q + (pos * n_heads + h) * d_head;
            float* k = K + (pos * n_heads + h) * d_head;
            
            for (int i = 0; i < half_d; i++) {
                float freq = 1.0f / powf(rope_theta, (float)(2 * i) / d_head);
                float angle = abs_pos * freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                
                /* Apply rotation to Q */
                float q0 = q[i];
                float q1 = q[i + half_d];
                q[i] = q0 * cos_val - q1 * sin_val;
                q[i + half_d] = q0 * sin_val + q1 * cos_val;
                
                /* Apply rotation to K */
                float k0 = k[i];
                float k1 = k[i + half_d];
                k[i] = k0 * cos_val - k1 * sin_val;
                k[i + half_d] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

/*============================================================================
 * ALIBI POSITIONAL BIAS
 *============================================================================*/

void cg_alibi_slopes(float* slopes, int n_heads) {
    /* Compute slopes: 2^(-8/n_heads * (i+1)) */
    float base = powf(2.0f, -8.0f / n_heads);
    for (int h = 0; h < n_heads; h++) {
        slopes[h] = powf(base, (float)(h + 1));
    }
}

void cg_alibi_bias(float* bias, int seqlen_q, int seqlen_k, 
                   int n_heads, float* slopes) {
    for (int h = 0; h < n_heads; h++) {
        float slope = slopes[h];
        for (int qi = 0; qi < seqlen_q; qi++) {
            for (int ki = 0; ki < seqlen_k; ki++) {
                /* Relative position: ki - qi */
                int rel_pos = ki - qi;
                bias[(h * seqlen_q + qi) * seqlen_k + ki] = slope * rel_pos;
            }
        }
    }
}

/*============================================================================
 * PARAMETER HELPERS
 *============================================================================*/

cg_flash_attn_params* cg_flash_attn_params_new(
    cg_tensor* Q, cg_tensor* K, cg_tensor* V,
    cg_attention_mask_type mask_type, float dropout_prob
) {
    cg_flash_attn_params* params = (cg_flash_attn_params*)calloc(1, sizeof(cg_flash_attn_params));
    
    params->Q = Q->data;
    params->K = K->data;
    params->V = V->data;
    
    /* Infer dimensions from Q: [batch, seqlen, n_heads, d_head] */
    params->batch_size = Q->shape[0];
    params->seqlen_q = Q->shape[1];
    params->n_heads = Q->shape[2];
    params->d_head = Q->shape[3];
    params->seqlen_k = K->shape[1];
    params->n_heads_k = K->shape[2];
    
    params->scale = 1.0f / sqrtf((float)params->d_head);
    params->dropout_prob = dropout_prob;
    params->mask_type = mask_type;
    
    /* Allocate output */
    int out_shape[] = {params->batch_size, params->seqlen_q, params->n_heads, params->d_head};
    cg_tensor* O = cg_tensor_zeros(out_shape, 4, false);
    params->O = O->data;
    
    return params;
}

void cg_flash_attn_params_free(cg_flash_attn_params* params) {
    free(params);
}

/*============================================================================
 * AUTO-TUNING
 *============================================================================*/

void cg_flash_attn_tune_blocks(int d_head, int* block_m, int* block_n) {
    /* Heuristic based on d_head */
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

size_t cg_flash_attn_shared_mem(int block_m, int block_n, int d_head) {
    /* Q block + K block + V block + stats */
    size_t q_size = block_m * d_head * sizeof(float);
    size_t k_size = block_n * d_head * sizeof(float);
    size_t v_size = block_n * d_head * sizeof(float);
    size_t stats = block_m * 2 * sizeof(float);  /* max + sum */
    
    return q_size + k_size + v_size + stats;
}
