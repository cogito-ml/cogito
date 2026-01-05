/**
 * SGD Optimizer Implementation
 */

#include "cg_optim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void sgd_step(cg_optimizer* self) {
    cg_sgd* sgd = (cg_sgd*)self;
    
    for (int i = 0; i < self->num_params; i++) {
        cg_tensor* p = self->params[i];
        if (!p || !p->grad) continue;
        
        for (int j = 0; j < p->size; j++) {
            float grad = p->grad[j];
            
            /* L2 regularization */
            if (sgd->weight_decay > 0) {
                grad += sgd->weight_decay * p->data[j];
            }
            
            if (sgd->momentum > 0) {
                /* Momentum update */
                float v = sgd->velocities[i]->data[j];
                v = sgd->momentum * v + grad;
                sgd->velocities[i]->data[j] = v;
                
                if (sgd->nesterov) {
                    p->data[j] -= self->learning_rate * (sgd->momentum * v + grad);
                } else {
                    p->data[j] -= self->learning_rate * v;
                }
            } else {
                /* Vanilla SGD */
                p->data[j] -= self->learning_rate * grad;
            }
        }
    }
    
    self->t++;
}

static void sgd_zero_grad(cg_optimizer* self) {
    for (int i = 0; i < self->num_params; i++) {
        if (self->params[i] && self->params[i]->grad) {
            memset(self->params[i]->grad, 0, self->params[i]->size * sizeof(float));
        }
    }
}

static void sgd_free(cg_optimizer* self) {
    cg_sgd* sgd = (cg_sgd*)self;
    if (sgd->velocities) {
        for (int i = 0; i < self->num_params; i++) {
            if (sgd->velocities[i]) cg_tensor_free(sgd->velocities[i]);
        }
        free(sgd->velocities);
    }
    free(self->params);
    free(sgd);
}

cg_sgd* cg_sgd_new(cg_tensor** params, int num_params,
                   float lr, float momentum, float weight_decay, bool nesterov) {
    cg_sgd* sgd = (cg_sgd*)calloc(1, sizeof(cg_sgd));
    if (!sgd) return NULL;
    
    cg_optimizer* base = (cg_optimizer*)sgd;
    base->learning_rate = lr;
    base->num_params = num_params;
    base->t = 0;
    base->step = sgd_step;
    base->zero_grad = sgd_zero_grad;
    base->free = sgd_free;
    
    base->params = (cg_tensor**)malloc(num_params * sizeof(cg_tensor*));
    memcpy(base->params, params, num_params * sizeof(cg_tensor*));
    
    sgd->momentum = momentum;
    sgd->weight_decay = weight_decay;
    sgd->nesterov = nesterov;
    
    /* Initialize velocity buffers if using momentum */
    if (momentum > 0) {
        sgd->velocities = (cg_tensor**)malloc(num_params * sizeof(cg_tensor*));
        for (int i = 0; i < num_params; i++) {
            sgd->velocities[i] = cg_tensor_zeros(params[i]->shape, params[i]->ndim, false);
        }
    }
    
    return sgd;
}

cg_sgd* cg_sgd_new_for_sequential(cg_sequential* model, float lr, float momentum,
                                   float weight_decay, bool nesterov) {
    cg_tensor** params = cg_sequential_get_params(model);
    int num_params = cg_sequential_num_params(model);
    return cg_sgd_new(params, num_params, lr, momentum, weight_decay, nesterov);
}

/* Optimizer interface functions */
void cg_optimizer_step(cg_optimizer* opt) { if (opt && opt->step) opt->step(opt); }
void cg_optimizer_zero_grad(cg_optimizer* opt) { if (opt && opt->zero_grad) opt->zero_grad(opt); }
void cg_optimizer_free(cg_optimizer* opt) { if (opt && opt->free) opt->free(opt); }
