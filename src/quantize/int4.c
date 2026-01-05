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
