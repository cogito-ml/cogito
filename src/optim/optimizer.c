/**
 * Cogito Optimizers & Mixed Precision Support
 */

#include "cg_optim.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

/* MIXED PRECISION & NUMERICAL STABILITY */

/* Stochastic Rounding RNG */
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Cast FP32 to BF16 with stochastic rounding */
/* Note: Since we don't have a real bf16 type in C, we simulate by masking mantissa */
static float stochastic_round_bf16(float val, uint32_t* rng) {
    uint32_t u_val;
    memcpy(&u_val, &val, 4);
    
    /* BF16 keeps top 16 bits. We add a random perturbation to the 17th bit before truncation */
    /* 1.0 in float is 0x3F800000. LSB of BF16 is 2^-7 relative to exp. */
    /* Simplified stochastic rounding: add random value in range [0, LSB) */
    
    uint32_t rand = xorshift32(rng) & 0xFFFF; // 16 random bits
    u_val += rand; 
    u_val &= 0xFFFF0000; // Truncate to 16 bits
    
    float res;
    memcpy(&res, &u_val, 4);
    return res;
}

void cg_optimizer_enable_loss_scaling(cg_optimizer* opt, float init_scale, 
                                      float factor, int window) {
    opt->use_mixed_precision = true;
    opt->loss_scale = init_scale;
    opt->scale_factor = factor > 1.0f ? factor : 2.0f;
    opt->scale_window = window > 0 ? window : 2000;
    opt->scale_backoff = 0;
}

void cg_optimizer_enable_master_weights(cg_optimizer* opt) {
    opt->use_master_weights = true;
    opt->master_params = (cg_tensor**)malloc(opt->num_params * sizeof(cg_tensor*));
    
    for (int i = 0; i < opt->num_params; i++) {
        cg_tensor* p = opt->params[i];
        /* Create FP32 copy */
        opt->master_params[i] = cg_tensor_new(p->shape, p->ndim, false); /* Assume FP32 default */
        /* Copy data directly */
        if (p->data && opt->master_params[i]->data) {
            memcpy(opt->master_params[i]->data, p->data, p->size * sizeof(float));
        }
    }
}

bool cg_optimizer_update_scale(cg_optimizer* opt, float grad_norm) {
    if (!opt->use_mixed_precision) return false;
    
    bool overflow = isnan(grad_norm) || isinf(grad_norm);
    
    if (overflow) {
        /* Reduce scale */
        opt->loss_scale /= opt->scale_factor;
        opt->scale_backoff = 0;
        printf("[AMPS] Overflow! Reducing scale to %.1f\n", opt->loss_scale);
        return true; /* Skip step */
    }
    
    /* Increase scale if stable for window steps */
    opt->scale_backoff++;
    if (opt->scale_backoff >= opt->scale_window) {
        opt->loss_scale *= opt->scale_factor;
        opt->scale_backoff = 0;
        printf("[AMPS] Stable. Increasing scale to %.1f\n", opt->loss_scale);
    }
    
    return false;
}

/* BASE OPTIMIZER */

void cg_optimizer_step(cg_optimizer* opt) {
    if (opt->step) opt->step(opt);
}

void cg_optimizer_zero_grad(cg_optimizer* opt) {
    if (opt->zero_grad) opt->zero_grad(opt);
}

void cg_optimizer_free(cg_optimizer* opt) {
    if (opt->free) opt->free(opt);
    free(opt->params);
    if (opt->master_params) {
        /* Free master params... */
        free(opt->master_params);
    }
    free(opt);
}
