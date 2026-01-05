/**
 * Adam Optimizer Implementation
 */

#include "cg_optim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void adam_step(cg_optimizer* self) {
    cg_adam* adam = (cg_adam*)self;
    self->t++;
    
    float bc1 = 1.0f - powf(adam->beta1, (float)self->t);  /* Bias correction 1 */
    float bc2 = 1.0f - powf(adam->beta2, (float)self->t);  /* Bias correction 2 */
    
    for (int i = 0; i < self->num_params; i++) {
        cg_tensor* p = self->params[i];
        if (!p || !p->grad) continue;
        
        cg_tensor* m = adam->m[i];
        cg_tensor* v = adam->v[i];
        
        for (int j = 0; j < p->size; j++) {
            float grad = p->grad[j];
            
            /* AdamW: decoupled weight decay */
            if (adam->weight_decay > 0) {
                p->data[j] -= self->learning_rate * adam->weight_decay * p->data[j];
            }
            
            /* Update biased first moment estimate */
            m->data[j] = adam->beta1 * m->data[j] + (1.0f - adam->beta1) * grad;
            
            /* Update biased second moment estimate */
            v->data[j] = adam->beta2 * v->data[j] + (1.0f - adam->beta2) * grad * grad;
            
            /* Bias-corrected estimates */
            float m_hat = m->data[j] / bc1;
            float v_hat = v->data[j] / bc2;
            
            /* Update parameters */
            p->data[j] -= self->learning_rate * m_hat / (sqrtf(v_hat) + adam->epsilon);
        }
    }
}

static void adam_zero_grad(cg_optimizer* self) {
    for (int i = 0; i < self->num_params; i++) {
        if (self->params[i] && self->params[i]->grad) {
            memset(self->params[i]->grad, 0, self->params[i]->size * sizeof(float));
        }
    }
}

static void adam_free(cg_optimizer* self) {
    cg_adam* adam = (cg_adam*)self;
    if (adam->m) {
        for (int i = 0; i < self->num_params; i++) {
            if (adam->m[i]) cg_tensor_free(adam->m[i]);
            if (adam->v[i]) cg_tensor_free(adam->v[i]);
        }
        free(adam->m);
        free(adam->v);
    }
    free(self->params);
    free(adam);
}

cg_adam* cg_adam_new(cg_tensor** params, int num_params,
                     float lr, float beta1, float beta2,
                     float epsilon, float weight_decay) {
    cg_adam* adam = (cg_adam*)calloc(1, sizeof(cg_adam));
    if (!adam) return NULL;
    
    cg_optimizer* base = (cg_optimizer*)adam;
    base->learning_rate = lr;
    base->num_params = num_params;
    base->t = 0;
    base->step = adam_step;
    base->zero_grad = adam_zero_grad;
    base->free = adam_free;
    
    base->params = (cg_tensor**)malloc(num_params * sizeof(cg_tensor*));
    memcpy(base->params, params, num_params * sizeof(cg_tensor*));
    
    adam->beta1 = beta1 > 0 ? beta1 : 0.9f;
    adam->beta2 = beta2 > 0 ? beta2 : 0.999f;
    adam->epsilon = epsilon > 0 ? epsilon : 1e-8f;
    adam->weight_decay = weight_decay;
    
    /* Initialize moment buffers */
    adam->m = (cg_tensor**)malloc(num_params * sizeof(cg_tensor*));
    adam->v = (cg_tensor**)malloc(num_params * sizeof(cg_tensor*));
    
    for (int i = 0; i < num_params; i++) {
        adam->m[i] = cg_tensor_zeros(params[i]->shape, params[i]->ndim, false);
        adam->v[i] = cg_tensor_zeros(params[i]->shape, params[i]->ndim, false);
    }
    
    return adam;
}

cg_adam* cg_adam_new_for_sequential(cg_sequential* model,
                                     float lr, float beta1, float beta2,
                                     float epsilon, float weight_decay) {
    cg_tensor** params = cg_sequential_get_params(model);
    int num_params = cg_sequential_num_params(model);
    return cg_adam_new(params, num_params, lr, beta1, beta2, epsilon, weight_decay);
}

/* Learning rate scheduling */
void cg_optimizer_step_lr(cg_optimizer* opt, int step, int step_size, float gamma) {
    if (step > 0 && step % step_size == 0) {
        opt->learning_rate *= gamma;
    }
}

void cg_optimizer_exp_lr(cg_optimizer* opt, float initial_lr, float gamma) {
    opt->learning_rate = initial_lr * powf(gamma, (float)opt->t);
}
