/**
 * BFloat16 Mixed Precision Training
 */

#include "cg_bf16.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*============================================================================
 * BF16 TENSOR
 *============================================================================*/

cg_tensor_bf16* cg_tensor_bf16_new(int* shape, int ndim, bool requires_grad) {
    cg_tensor_bf16* t = (cg_tensor_bf16*)calloc(1, sizeof(cg_tensor_bf16));
    t->ndim = ndim;
    t->size = 1;
    
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    
    t->data = (cg_bf16*)calloc(t->size, sizeof(cg_bf16));
    t->requires_grad = requires_grad;
    
    if (requires_grad) {
        t->grad_fp32 = (float*)calloc(t->size, sizeof(float));
    }
    
    return t;
}

cg_tensor_bf16* cg_tensor_bf16_from_fp32(cg_tensor* fp32) {
    cg_tensor_bf16* bf16 = cg_tensor_bf16_new(fp32->shape, fp32->ndim, fp32->requires_grad);
    
    for (int i = 0; i < bf16->size; i++) {
        bf16->data[i] = cg_float_to_bf16(fp32->data[i]);
    }
    
    /* Keep master copy */
    bf16->has_master_copy = true;
    bf16->data_fp32 = (float*)malloc(bf16->size * sizeof(float));
    memcpy(bf16->data_fp32, fp32->data, bf16->size * sizeof(float));
    
    return bf16;
}

cg_tensor* cg_tensor_bf16_to_fp32(cg_tensor_bf16* bf16) {
    cg_tensor* fp32 = cg_tensor_new(bf16->shape, bf16->ndim, bf16->requires_grad);
    
    for (int i = 0; i < fp32->size; i++) {
        fp32->data[i] = cg_bf16_to_float(bf16->data[i]);
    }
    
    return fp32;
}

void cg_tensor_bf16_free(cg_tensor_bf16* t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->data_fp32);
    free(t->grad_fp32);
    free(t);
}

/*============================================================================
 * MIXED PRECISION TRAINER
 *============================================================================*/

cg_mixed_precision* cg_mixed_precision_new(cg_tensor** params, int num_params) {
    cg_mixed_precision* mp = (cg_mixed_precision*)calloc(1, sizeof(cg_mixed_precision));
    mp->num_params = num_params;
    
    /* Allocate arrays */
    mp->master_params = (cg_tensor**)malloc(num_params * sizeof(cg_tensor*));
    mp->model_params = (cg_tensor_bf16**)malloc(num_params * sizeof(cg_tensor_bf16*));
    
    /* Copy params and create BF16 versions */
    for (int i = 0; i < num_params; i++) {
        mp->master_params[i] = cg_tensor_clone(params[i]);
        mp->model_params[i] = cg_tensor_bf16_from_fp32(params[i]);
    }
    
    /* Initialize loss scaling */
    mp->loss_scale = 65536.0f;   /* Start high */
    mp->loss_scale_min = 1.0f;
    mp->loss_scale_max = 65536.0f;
    mp->scale_growth_interval = 2000;
    mp->steps_since_update = 0;
    mp->overflow_count = 0;
    mp->max_grad_norm = 1.0f;
    
    return mp;
}

bool cg_mp_check_overflow(cg_mixed_precision* mp) {
    /* Check for inf/nan in BF16 gradients */
    for (int p = 0; p < mp->num_params; p++) {
        cg_tensor_bf16* param = mp->model_params[p];
        if (!param->grad_fp32) continue;
        
        for (int i = 0; i < param->size; i++) {
            float g = param->grad_fp32[i];
            if (isnan(g) || isinf(g)) {
                return true;
            }
        }
    }
    return false;
}

void cg_mp_step(cg_mixed_precision* mp) {
    /* 1. Check for gradient overflow */
    bool overflow = cg_mp_check_overflow(mp);
    
    if (overflow) {
        /* Reduce loss scale and skip update */
        mp->loss_scale = fmaxf(mp->loss_scale / 2.0f, mp->loss_scale_min);
        mp->overflow_count++;
        mp->steps_since_update = 0;
        printf("MP: Overflow detected, scale -> %.1f\n", mp->loss_scale);
        
        /* Zero gradients */
        for (int p = 0; p < mp->num_params; p++) {
            if (mp->model_params[p]->grad_fp32) {
                memset(mp->model_params[p]->grad_fp32, 0,
                       mp->model_params[p]->size * sizeof(float));
            }
        }
        return;
    }
    
    /* 2. Unscale gradients */
    float unscale = 1.0f / mp->loss_scale;
    for (int p = 0; p < mp->num_params; p++) {
        cg_tensor_bf16* param = mp->model_params[p];
        if (!param->grad_fp32) continue;
        
        for (int i = 0; i < param->size; i++) {
            param->grad_fp32[i] *= unscale;
        }
    }
    
    /* 3. Gradient clipping */
    float grad_norm = 0.0f;
    for (int p = 0; p < mp->num_params; p++) {
        cg_tensor_bf16* param = mp->model_params[p];
        if (!param->grad_fp32) continue;
        
        for (int i = 0; i < param->size; i++) {
            grad_norm += param->grad_fp32[i] * param->grad_fp32[i];
        }
    }
    grad_norm = sqrtf(grad_norm);
    
    if (grad_norm > mp->max_grad_norm) {
        float clip_scale = mp->max_grad_norm / grad_norm;
        for (int p = 0; p < mp->num_params; p++) {
            cg_tensor_bf16* param = mp->model_params[p];
            if (!param->grad_fp32) continue;
            
            for (int i = 0; i < param->size; i++) {
                param->grad_fp32[i] *= clip_scale;
            }
        }
    }
    
    /* 4. Update master params (FP32 optimizer step would go here) */
    float lr = 0.001f;  /* Example */
    for (int p = 0; p < mp->num_params; p++) {
        cg_tensor* master = mp->master_params[p];
        cg_tensor_bf16* model = mp->model_params[p];
        
        if (!model->grad_fp32) continue;
        
        for (int i = 0; i < master->size; i++) {
            master->data[i] -= lr * model->grad_fp32[i];
        }
        
        /* 5. Cast back to BF16 */
        for (int i = 0; i < model->size; i++) {
            model->data[i] = cg_float_to_bf16(master->data[i]);
        }
        
        /* Zero grad for next step */
        memset(model->grad_fp32, 0, model->size * sizeof(float));
    }
    
    /* 6. Update loss scale */
    mp->steps_since_update++;
    if (mp->steps_since_update >= mp->scale_growth_interval) {
        mp->loss_scale = fminf(mp->loss_scale * 2.0f, mp->loss_scale_max);
        mp->steps_since_update = 0;
    }
}

void cg_mixed_precision_free(cg_mixed_precision* mp) {
    if (!mp) return;
    
    for (int i = 0; i < mp->num_params; i++) {
        cg_tensor_free(mp->master_params[i]);
        cg_tensor_bf16_free(mp->model_params[i]);
    }
    
    free(mp->master_params);
    free(mp->model_params);
    free(mp);
}

/*============================================================================
 * BF16 OPERATIONS (CPU fallback)
 *============================================================================*/

void cg_bf16_add(cg_tensor_bf16* a, cg_tensor_bf16* b, cg_tensor_bf16* out) {
    for (int i = 0; i < out->size; i++) {
        float va = cg_bf16_to_float(a->data[i]);
        float vb = cg_bf16_to_float(b->data[i]);
        out->data[i] = cg_float_to_bf16(va + vb);
    }
}

void cg_bf16_mul(cg_tensor_bf16* a, cg_tensor_bf16* b, cg_tensor_bf16* out) {
    for (int i = 0; i < out->size; i++) {
        float va = cg_bf16_to_float(a->data[i]);
        float vb = cg_bf16_to_float(b->data[i]);
        out->data[i] = cg_float_to_bf16(va * vb);
    }
}

void cg_bf16_matmul(cg_tensor_bf16* a, cg_tensor_bf16* b, cg_tensor_bf16* out) {
    /* CPU fallback: convert to FP32, compute, convert back */
    int M = a->shape[0];
    int K = a->shape[1];
    int N = b->shape[1];
    
    /* Zero output */
    for (int i = 0; i < out->size; i++) {
        out->data[i] = cg_float_to_bf16(0.0f);
    }
    
    /* Naive matmul */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float va = cg_bf16_to_float(a->data[i * K + k]);
                float vb = cg_bf16_to_float(b->data[k * N + j]);
                sum += va * vb;
            }
            out->data[i * N + j] = cg_float_to_bf16(sum);
        }
    }
}

void cg_bf16_gelu(cg_tensor_bf16* a, cg_tensor_bf16* out) {
    for (int i = 0; i < out->size; i++) {
        float x = cg_bf16_to_float(a->data[i]);
        /* GELU approximation */
        float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out->data[i] = cg_float_to_bf16(gelu);
    }
}

void cg_bf16_layernorm(cg_tensor_bf16* x, cg_tensor_bf16* gamma, 
                       cg_tensor_bf16* beta, float eps, cg_tensor_bf16* out) {
    /* Assume x is [batch, features] */
    int batch = x->shape[0];
    int features = x->shape[1];
    
    for (int b = 0; b < batch; b++) {
        /* Compute mean */
        float mean = 0.0f;
        for (int f = 0; f < features; f++) {
            mean += cg_bf16_to_float(x->data[b * features + f]);
        }
        mean /= features;
        
        /* Compute variance */
        float var = 0.0f;
        for (int f = 0; f < features; f++) {
            float diff = cg_bf16_to_float(x->data[b * features + f]) - mean;
            var += diff * diff;
        }
        var /= features;
        
        /* Normalize */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int f = 0; f < features; f++) {
            float val = cg_bf16_to_float(x->data[b * features + f]);
            float norm = (val - mean) * inv_std;
            
            /* Scale and shift */
            if (gamma) norm *= cg_bf16_to_float(gamma->data[f]);
            if (beta) norm += cg_bf16_to_float(beta->data[f]);
            
            out->data[b * features + f] = cg_float_to_bf16(norm);
        }
    }
}
