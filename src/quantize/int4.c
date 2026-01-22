/**
 * INT4 Quantization
 */

#include "cg_quantize.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

cg_tensor_quant* cg_quantize_int4(cg_tensor* tensor, int group_size) {
    cg_tensor_quant* q = (cg_tensor_quant*)calloc(1, sizeof(cg_tensor_quant));
    q->quant_type = QUANT_INT4;
    q->group_size = group_size;
    q->size = tensor->size;
    q->ndim = tensor->ndim;
    q->shape = (int*)malloc(tensor->ndim * sizeof(int));
    memcpy(q->shape, tensor->shape, tensor->ndim * sizeof(int));
    
    q->num_groups = (tensor->size + group_size - 1) / group_size;
    q->scales = (float*)malloc(q->num_groups * sizeof(float));
    q->zeros = (float*)calloc(q->num_groups, sizeof(float));
    q->data = (uint8_t*)malloc((tensor->size + 1) / 2);  /* 2 values per byte */
    q->symmetric = true;
    
    /* Compute per-group scales */
    for (int g = 0; g < q->num_groups; g++) {
        int start = g * group_size;
        int end = start + group_size;
        if (end > tensor->size) end = tensor->size;
        
        float absmax = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(tensor->data[i]);
            if (abs_val > absmax) absmax = abs_val;
        }
        
        q->scales[g] = absmax / 7.0f;  /* INT4: -8 to 7 range */
        if (q->scales[g] < 1e-10f) q->scales[g] = 1.0f;
    }
    
    /* Quantize */
    for (int i = 0; i < tensor->size; i += 2) {
        int g = i / group_size;
        float scale = q->scales[g];
        
        int8_t v0 = (int8_t)roundf(tensor->data[i] / scale);
        if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
        
        int8_t v1 = 0;
        if (i + 1 < tensor->size) {
            v1 = (int8_t)roundf(tensor->data[i + 1] / scale);
            if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8;
        }
        
        q->data[i / 2] = pack_int4(v0, v1);
    }
    
    return q;
}

cg_tensor* cg_dequantize(cg_tensor_quant* q) {
    cg_tensor* t = cg_tensor_new(q->shape, q->ndim, false);
    
    for (int i = 0; i < q->size; i += 2) {
        int g = i / q->group_size;
        float scale = q->scales[g];
        float zero = q->zeros ? q->zeros[g] : 0.0f;
        
        int8_t v0, v1;
        unpack_int4(q->data[i / 2], &v0, &v1);
        
        t->data[i] = dequant_int4(v0, scale, zero);
        if (i + 1 < q->size) {
            t->data[i + 1] = dequant_int4(v1, scale, zero);
        }
    }
    
    return t;
}

void cg_tensor_quant_free(cg_tensor_quant* q) {
    if (!q) return;
    free(q->data);
    free(q->scales);
    free(q->zeros);
    free(q->shape);
    free(q);
}

void cg_matmul_int4(cg_tensor* input, cg_tensor_quant* weight, cg_tensor* output) {
    int M = input->shape[0];
    int K = input->shape[1];
    int N = weight->shape[1];
    
    memset(output->data, 0, output->size * sizeof(float));
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int w_idx = k * N + j;
                int g = w_idx / weight->group_size;
                float scale = weight->scales[g];
                
                int8_t v0, v1;
                unpack_int4(weight->data[w_idx / 2], &v0, &v1);
                int8_t w_val = (w_idx % 2 == 0) ? v0 : v1;
                
                sum += input->data[i * K + k] * (float)w_val * scale;
            }
            output->data[i * N + j] = sum;
        }
    }
}

cg_kv_cache_int4* cg_kv_cache_int4_new(int num_layers, int max_seq_len,
                                        int n_heads, int head_dim, int group_size) {
    cg_kv_cache_int4* c = (cg_kv_cache_int4*)calloc(1, sizeof(cg_kv_cache_int4));
    c->num_layers = num_layers;
    c->max_seq_len = max_seq_len;
    c->n_heads = n_heads;
    c->head_dim = head_dim;
    c->group_size = group_size;
    
    size_t cache_size = (size_t)num_layers * max_seq_len * n_heads * (head_dim / 2);
    c->k_cache = (uint8_t*)calloc(cache_size, 1);
    c->v_cache = (uint8_t*)calloc(cache_size, 1);
    
    size_t num_groups = num_layers * max_seq_len * n_heads * head_dim / group_size;
    c->k_scales = (float*)calloc(num_groups, sizeof(float));
    c->v_scales = (float*)calloc(num_groups, sizeof(float));
    
    return c;
}

void cg_kv_cache_int4_free(cg_kv_cache_int4* c) {
    if (!c) return;
    free(c->k_cache);
    free(c->v_cache);
    free(c->k_scales);
    free(c->v_scales);
    free(c);
}

void cg_kv_cache_int4_reset(cg_kv_cache_int4* c) { c->current_len = 0; }

void cg_kv_cache_int4_append(cg_kv_cache_int4* c, int layer_idx,
                             cg_tensor* new_k, cg_tensor* new_v) {
    /* 
     * Append new token(s) to cache. 
     * new_k/v shape: [batch, 1, n_heads, head_dim] (assuming single token append)
     */
    if (c->current_len >= c->max_seq_len) return;
    
    int pos = c->current_len;
    int head_dim = c->head_dim;
    int n_heads = c->n_heads;
    int group_size = c->group_size;
    int num_groups_per_head = head_dim / group_size;
    
    /* Calculate offsets */
    /* Cache layout: [layers, max_seq, n_heads, head_dim/2] */
    size_t layer_stride = (size_t)c->max_seq_len * n_heads * (head_dim / 2);
    size_t pos_stride = (size_t)n_heads * (head_dim / 2);
    
    size_t scale_layer_stride = (size_t)c->max_seq_len * n_heads * num_groups_per_head;
    size_t scale_pos_stride = (size_t)n_heads * num_groups_per_head;
    
    uint8_t* k_ptr = c->k_cache + layer_idx * layer_stride + pos * pos_stride;
    uint8_t* v_ptr = c->v_cache + layer_idx * layer_stride + pos * pos_stride;
    float* k_s_ptr = c->k_scales + layer_idx * scale_layer_stride + pos * scale_pos_stride;
    float* v_s_ptr = c->v_scales + layer_idx * scale_layer_stride + pos * scale_pos_stride;
    
    /* Quantize K and V */
    /* Iterate over heads */
    for (int h = 0; h < n_heads; h++) {
        /* Iterate over groups in head_dim */
        for (int g = 0; g < num_groups_per_head; g++) {
            /* 1. Find max in group */
            float max_val_k = 0.0f;
            float max_val_v = 0.0f;
            int start = g * group_size;
            
            for (int i = 0; i < group_size; i++) {
                int dim_idx = start + i;
                /* Assume new_k is [1, 1, n_heads, head_dim] or flattened */
                /* Index: h * head_dim + dim_idx */
                float val_k = fabsf(new_k->data[h * head_dim + dim_idx]);
                if (val_k > max_val_k) max_val_k = val_k;
                
                float val_v = fabsf(new_v->data[h * head_dim + dim_idx]);
                if (val_v > max_val_v) max_val_v = val_v;
            }
            
            float scale_k = max_val_k / 7.0f;
            if (scale_k < 1e-8f) scale_k = 1.0f;
            k_s_ptr[h * num_groups_per_head + g] = scale_k;
            
            float scale_v = max_val_v / 7.0f;
            if (scale_v < 1e-8f) scale_v = 1.0f;
            v_s_ptr[h * num_groups_per_head + g] = scale_v;
            
            /* 2. Quantize and Pack */
            for (int i = 0; i < group_size; i += 2) {
                int dim_idx = start + i;
                
                /* K */
                float val_k0 = new_k->data[h * head_dim + dim_idx];
                float val_k1 = new_k->data[h * head_dim + dim_idx + 1];
                int8_t vk0 = (int8_t)roundf(val_k0 / scale_k);
                int8_t vk1 = (int8_t)roundf(val_k1 / scale_k);
                
                /* Clamp */
                if (vk0 > 7) vk0 = 7; if (vk0 < -8) vk0 = -8;
                if (vk1 > 7) vk1 = 7; if (vk1 < -8) vk1 = -8;
                
                k_ptr[h * (head_dim/2) + (start + i)/2] = pack_int4(vk0, vk1);
                
                /* V */
                float val_v0 = new_v->data[h * head_dim + dim_idx];
                float val_v1 = new_v->data[h * head_dim + dim_idx + 1];
                int8_t vv0 = (int8_t)roundf(val_v0 / scale_v);
                int8_t vv1 = (int8_t)roundf(val_v1 / scale_v);
                
                if (vv0 > 7) vv0 = 7; if (vv0 < -8) vv0 = -8;
                if (vv1 > 7) vv1 = 7; if (vv1 < -8) vv1 = -8;
                
                v_ptr[h * (head_dim/2) + (start + i)/2] = pack_int4(vv0, vv1);
            }
        }
    }
    
    c->current_len++;
}

void cg_kv_cache_int4_get(cg_kv_cache_int4* c, int layer_idx,
                          cg_tensor* k_out, cg_tensor* v_out) {
    /* Dequantize entire history for attention */
    int len = c->current_len;
    int n_heads = c->n_heads;
    int head_dim = c->head_dim;
    int group_size = c->group_size;
    int num_groups = head_dim / group_size;
    
    /* Pointers */
    size_t layer_stride = (size_t)c->max_seq_len * n_heads * (head_dim / 2);
    size_t scale_stride = (size_t)c->max_seq_len * n_heads * num_groups;
    
    uint8_t* k_base = c->k_cache + layer_idx * layer_stride;
    uint8_t* v_base = c->v_cache + layer_idx * layer_stride;
    float* ks_base = c->k_scales + layer_idx * scale_stride;
    float* vs_base = c->v_scales + layer_idx * scale_stride;
    
    /* Iterate tokens */
    for (int t = 0; t < len; t++) {
        size_t t_off = t * n_heads * (head_dim / 2);
        size_t s_off = t * n_heads * num_groups;
        
        for (int h = 0; h < n_heads; h++) {
            for (int g = 0; g < num_groups; g++) {
                float scale_k = ks_base[s_off + h * num_groups + g];
                float scale_v = vs_base[s_off + h * num_groups + g];
                
                int start = g * group_size;
                
                for (int i = 0; i < group_size; i += 2) {
                    uint8_t pk = k_base[t_off + h * (head_dim/2) + (start + i)/2];
                    uint8_t pv = v_base[t_off + h * (head_dim/2) + (start + i)/2];
                    
                    int8_t vk0, vk1;
                    unpack_int4(pk, &vk0, &vk1);
                    
                    int8_t vv0, vv1;
                    unpack_int4(pv, &vv0, &vv1);
                    
                    /* Output index: [t, h, start+i] (flattened) */
                    int out_idx_0 = ((t * n_heads + h) * head_dim) + (start + i);
                    int out_idx_1 = out_idx_0 + 1;
                    
                    k_out->data[out_idx_0] = (float)vk0 * scale_k;
                    k_out->data[out_idx_1] = (float)vk1 * scale_k;
                    
                    v_out->data[out_idx_0] = (float)vv0 * scale_v;
                    v_out->data[out_idx_1] = (float)vv1 * scale_v;
                }
            }
        }
    }
}

cg_quant_calibration* cg_calibration_new(int channels) {
    cg_quant_calibration* c = (cg_quant_calibration*)calloc(1, sizeof(cg_quant_calibration));
    c->channels = channels;
    c->min_vals = (float*)malloc(channels * sizeof(float));
    c->max_vals = (float*)malloc(channels * sizeof(float));
    c->absmax = (float*)calloc(channels, sizeof(float));
    for (int i = 0; i < channels; i++) {
        c->min_vals[i] = 1e9f;
        c->max_vals[i] = -1e9f;
    }
    return c;
}

void cg_calibration_update(cg_quant_calibration* c, cg_tensor* act) {
    int batch = act->shape[0];
    int features = act->shape[1];
    
    for (int b = 0; b < batch; b++) {
        for (int f = 0; f < features && f < c->channels; f++) {
            float val = act->data[b * features + f];
            if (val < c->min_vals[f]) c->min_vals[f] = val;
            if (val > c->max_vals[f]) c->max_vals[f] = val;
            if (fabsf(val) > c->absmax[f]) c->absmax[f] = fabsf(val);
        }
    }
    c->num_samples++;
}

void cg_calibration_free(cg_quant_calibration* c) {
    if (!c) return;
    free(c->min_vals);
    free(c->max_vals);
    free(c->absmax);
    free(c);
}
