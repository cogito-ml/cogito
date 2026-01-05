/**
 * Optimizers for Cogito
 * 
 * Implements SGD (with momentum) and Adam optimizers.
 */

#ifndef CG_OPTIM_H
#define CG_OPTIM_H

#include "cg_tensor.h"
#include "cg_layers.h"

/* Forward declarations */
typedef struct cg_optimizer cg_optimizer;

/*============================================================================
 * BASE OPTIMIZER INTERFACE
 *============================================================================*/

struct cg_optimizer {
    /* Learning rate */
    float learning_rate;
    
    /* Parameters being optimized */
    cg_tensor** params;
    int num_params;
    
    /* Iteration count (for Adam bias correction) */
    int t;
    
    /* Virtual methods */
    void (*step)(cg_optimizer* self);
    void (*zero_grad)(cg_optimizer* self);
    void (*free)(cg_optimizer* self);
};

/**
 * Perform one optimization step (update parameters).
 */
void cg_optimizer_step(cg_optimizer* opt);

/**
 * Zero gradients of all parameters.
 */
void cg_optimizer_zero_grad(cg_optimizer* opt);

/**
 * Free optimizer resources.
 */
void cg_optimizer_free(cg_optimizer* opt);

/*============================================================================
 * SGD OPTIMIZER
 *============================================================================*/

/**
 * Stochastic Gradient Descent with optional momentum.
 * 
 * Update rule (with momentum):
 *   v_t = momentum * v_{t-1} + grad
 *   param = param - lr * v_t
 * 
 * With Nesterov momentum:
 *   v_t = momentum * v_{t-1} + grad
 *   param = param - lr * (momentum * v_t + grad)
 */
typedef struct {
    cg_optimizer base;
    
    float momentum;                /* Momentum factor (0 = vanilla SGD) */
    float weight_decay;            /* L2 regularization factor */
    bool nesterov;                 /* Use Nesterov momentum */
    
    /* Velocity buffers (one per parameter) */
    cg_tensor** velocities;
} cg_sgd;

/**
 * Create SGD optimizer.
 * 
 * @param params Array of parameter tensors to optimize
 * @param num_params Number of parameters
 * @param lr Learning rate
 * @param momentum Momentum factor (0 for vanilla SGD)
 * @param weight_decay L2 regularization factor
 * @param nesterov Use Nesterov momentum
 */
cg_sgd* cg_sgd_new(cg_tensor** params, int num_params, 
                   float lr, float momentum, float weight_decay, bool nesterov);

/**
 * Convenience: create SGD for a sequential model.
 */
cg_sgd* cg_sgd_new_for_sequential(cg_sequential* model,
                                   float lr, float momentum, 
                                   float weight_decay, bool nesterov);

/*============================================================================
 * ADAM OPTIMIZER
 *============================================================================*/

/**
 * Adam optimizer with bias correction.
 * 
 * Update rule:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * grad
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
 *   m_hat = m_t / (1 - beta1^t)
 *   v_hat = v_t / (1 - beta2^t)
 *   param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
 */
typedef struct {
    cg_optimizer base;
    
    float beta1;                   /* First moment decay (default: 0.9) */
    float beta2;                   /* Second moment decay (default: 0.999) */
    float epsilon;                 /* Numerical stability (default: 1e-8) */
    float weight_decay;            /* Decoupled weight decay (AdamW) */
    
    /* State buffers */
    cg_tensor** m;                 /* First moment estimates */
    cg_tensor** v;                 /* Second moment estimates */
} cg_adam;

/**
 * Create Adam optimizer.
 * 
 * @param params Array of parameter tensors
 * @param num_params Number of parameters
 * @param lr Learning rate
 * @param beta1 First moment decay (default: 0.9)
 * @param beta2 Second moment decay (default: 0.999)
 * @param epsilon Numerical stability constant
 * @param weight_decay Weight decay factor (0 for no regularization)
 */
cg_adam* cg_adam_new(cg_tensor** params, int num_params,
                     float lr, float beta1, float beta2, 
                     float epsilon, float weight_decay);

/**
 * Convenience: create Adam for a sequential model.
 */
cg_adam* cg_adam_new_for_sequential(cg_sequential* model,
                                     float lr, float beta1, float beta2,
                                     float epsilon, float weight_decay);

/*============================================================================
 * LEARNING RATE SCHEDULING
 *============================================================================*/

/**
 * Step decay: multiply lr by gamma every step_size iterations.
 */
void cg_optimizer_step_lr(cg_optimizer* opt, int current_step, 
                          int step_size, float gamma);

/**
 * Exponential decay: lr = initial_lr * gamma^t
 */
void cg_optimizer_exp_lr(cg_optimizer* opt, float initial_lr, float gamma);

#endif /* CG_OPTIM_H */
