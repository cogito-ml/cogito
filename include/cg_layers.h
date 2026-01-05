/**
 * Neural Network Layers for Cogito
 * 
 * All layers implement forward and backward passes.
 * Uses function pointers for polymorphism.
 */

#ifndef CG_LAYERS_H
#define CG_LAYERS_H

#include "cg_tensor.h"

/* Forward declarations */
typedef struct cg_layer cg_layer;

/*============================================================================
 * BASE LAYER INTERFACE
 *============================================================================*/

/**
 * Base layer structure with virtual function table.
 */
struct cg_layer {
    /* Layer name for debugging */
    const char* name;
    
    /* Learnable parameters */
    cg_tensor* weights;
    cg_tensor* bias;
    
    /* Cached values for backward pass */
    cg_tensor* input;              /* Saved input from forward pass */
    cg_tensor* output;             /* Output from forward pass */
    
    /* Virtual methods */
    cg_tensor* (*forward)(cg_layer* self, cg_tensor* input);
    void (*backward)(cg_layer* self, cg_tensor* grad_output);
    void (*free)(cg_layer* self);
    
    /* For optimizer: get all trainable parameters */
    int (*num_params)(cg_layer* self);
    cg_tensor** (*get_params)(cg_layer* self);
};

/**
 * Free a layer and all its resources.
 */
void cg_layer_free(cg_layer* layer);

/**
 * Zero gradients for all parameters in a layer.
 */
void cg_layer_zero_grad(cg_layer* layer);

/*============================================================================
 * LINEAR LAYER (Fully Connected)
 *============================================================================*/

/**
 * Linear layer: y = Wx + b
 * 
 * Input shape: [batch_size, in_features]
 * Output shape: [batch_size, out_features]
 */
typedef struct {
    cg_layer base;
    int in_features;
    int out_features;
} cg_linear;

/**
 * Create a new linear layer.
 * 
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @param bias Whether to include bias term
 * @return New linear layer or NULL on failure
 * 
 * Weight initialization: Xavier uniform
 */
cg_linear* cg_linear_new(int in_features, int out_features, bool bias);

/*============================================================================
 * ACTIVATION LAYERS
 *============================================================================*/

/**
 * ReLU activation: max(0, x)
 */
typedef struct {
    cg_layer base;
    cg_tensor* mask;               /* Saved for backward: where x > 0 */
} cg_relu;

cg_relu* cg_relu_new(void);

/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 */
typedef struct {
    cg_layer base;
} cg_sigmoid;

cg_sigmoid* cg_sigmoid_new(void);

/**
 * Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
typedef struct {
    cg_layer base;
} cg_tanh_layer;

cg_tanh_layer* cg_tanh_new(void);

/**
 * Softmax activation: exp(x_i) / sum(exp(x_j))
 * Applied along last dimension.
 */
typedef struct {
    cg_layer base;
} cg_softmax;

cg_softmax* cg_softmax_new(void);

/*============================================================================
 * DROPOUT LAYER
 *============================================================================*/

/**
 * Dropout layer for regularization.
 * During training, randomly zeros elements with probability p.
 * During inference, scales outputs by (1 - p).
 */
typedef struct {
    cg_layer base;
    float p;                       /* Dropout probability */
    bool training;                 /* Training mode flag */
    cg_tensor* mask;               /* Random mask for backward */
} cg_dropout;

cg_dropout* cg_dropout_new(float p);
void cg_dropout_set_training(cg_dropout* layer, bool training);

/*============================================================================
 * BATCH NORMALIZATION
 *============================================================================*/

/**
 * Batch normalization layer.
 * Normalizes along batch dimension, then scales and shifts.
 */
typedef struct {
    cg_layer base;
    int num_features;
    float epsilon;                 /* For numerical stability */
    float momentum;                /* For running stats */
    
    /* Learnable parameters (in base.weights and base.bias) */
    /* gamma = base.weights, beta = base.bias */
    
    /* Running statistics */
    cg_tensor* running_mean;
    cg_tensor* running_var;
    
    /* Saved for backward */
    cg_tensor* input_normalized;
    cg_tensor* batch_mean;
    cg_tensor* batch_var;
    
    bool training;
} cg_batchnorm;

cg_batchnorm* cg_batchnorm_new(int num_features, float epsilon, float momentum);
void cg_batchnorm_set_training(cg_batchnorm* layer, bool training);

/*============================================================================
 * SEQUENTIAL CONTAINER
 *============================================================================*/

/**
 * Sequential container for stacking layers.
 */
typedef struct {
    cg_layer** layers;
    int num_layers;
    int capacity;
} cg_sequential;

cg_sequential* cg_sequential_new(void);
void cg_sequential_add(cg_sequential* seq, cg_layer* layer);
cg_tensor* cg_sequential_forward(cg_sequential* seq, cg_tensor* input);
void cg_sequential_backward(cg_sequential* seq, cg_tensor* grad_output);
void cg_sequential_free(cg_sequential* seq);
void cg_sequential_zero_grad(cg_sequential* seq);

/**
 * Get all trainable parameters from sequential.
 */
int cg_sequential_num_params(cg_sequential* seq);
cg_tensor** cg_sequential_get_params(cg_sequential* seq);

#endif /* CG_LAYERS_H */
