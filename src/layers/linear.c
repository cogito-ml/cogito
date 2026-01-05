/**
 * Linear Layer Implementation
 * 
 * Fully connected layer: y = Wx + b
 * With Xavier/He weight initialization.
 */

#include "cg_layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Forward declarations for autograd helpers */
extern void cg_tensor_matmul(cg_tensor* a, cg_tensor* b, cg_tensor* out);
extern void cg_tensor_add(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/*============================================================================
 * LINEAR LAYER
 *============================================================================*/

/* Xavier uniform initialization: U[-sqrt(6/(in+out)), sqrt(6/(in+out))] */
static void xavier_init(cg_tensor* t, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    
    for (int i = 0; i < t->size; i++) {
        float u = (float)rand() / (float)RAND_MAX;  /* [0, 1) */
        t->data[i] = (2.0f * u - 1.0f) * limit;     /* [-limit, limit) */
    }
}

/* Kaiming (He) initialization for ReLU: N(0, sqrt(2/fan_in)) */
static void kaiming_init(cg_tensor* t, int fan_in) {
    float std = sqrtf(2.0f / (float)fan_in);
    
    for (int i = 0; i < t->size; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        
        float mag = sqrtf(-2.0f * logf(u1));
        t->data[i] = mag * cosf(2.0f * (float)M_PI * u2) * std;
        if (i + 1 < t->size) {
            t->data[i + 1] = mag * sinf(2.0f * (float)M_PI * u2) * std;
        }
    }
}

/* Linear forward: y = x @ W^T + b */
static cg_tensor* linear_forward(cg_layer* self, cg_tensor* input) {
    cg_linear* linear = (cg_linear*)self;
    
    assert(input->ndim == 2);
    assert(input->shape[1] == linear->in_features);
    
    int batch_size = input->shape[0];
    
    /* Save input for backward */
    if (self->input) {
        cg_tensor_release(self->input);
    }
    self->input = input;
    cg_tensor_retain(input);
    
    /* Create output tensor: [batch_size, out_features] */
    int out_shape[] = {batch_size, linear->out_features};
    cg_tensor* output = cg_tensor_new(out_shape, 2, true);
    
    /* Compute x @ W^T
     * input: [batch_size, in_features]
     * weights: [out_features, in_features]
     * We need: [batch_size, out_features]
     * 
     * Actually store weights as [in_features, out_features] for easier matmul
     */
    cg_tensor_matmul(input, self->weights, output);
    
    /* Add bias if present (broadcast over batch) */
    if (self->bias) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < linear->out_features; j++) {
                output->data[i * linear->out_features + j] += self->bias->data[j];
            }
        }
    }
    
    /* Save output for potential layer chaining */
    if (self->output) {
        cg_tensor_release(self->output);
    }
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

/* Linear backward:
 * dW = x^T @ grad_output
 * db = sum(grad_output, axis=0)
 * dx = grad_output @ W (for previous layer)
 */
static void linear_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_linear* linear = (cg_linear*)self;
    cg_tensor* input = self->input;
    
    assert(input != NULL);
    assert(grad_output->ndim == 2);
    
    int batch_size = input->shape[0];
    
    /* Compute weight gradients: dW = x^T @ grad_output */
    /* input: [batch_size, in_features]
     * grad_output: [batch_size, out_features]
     * dW: [in_features, out_features]
     */
    if (self->weights->grad) {
        /* Compute x^T @ grad_output */
        for (int i = 0; i < linear->in_features; i++) {
            for (int j = 0; j < linear->out_features; j++) {
                float sum = 0.0f;
                for (int b = 0; b < batch_size; b++) {
                    sum += input->data[b * linear->in_features + i] *
                           grad_output->grad[b * linear->out_features + j];
                }
                self->weights->grad[i * linear->out_features + j] += sum;
            }
        }
    }
    
    /* Compute bias gradients */
    if (self->bias && self->bias->grad) {
        for (int j = 0; j < linear->out_features; j++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                sum += grad_output->grad[b * linear->out_features + j];
            }
            self->bias->grad[j] += sum;
        }
    }
    
    /* Compute input gradients: dx = grad_output @ W^T */
    if (input->requires_grad && input->grad) {
        /* grad_output: [batch_size, out_features]
         * W: [in_features, out_features]
         * W^T: [out_features, in_features]
         * dx: [batch_size, in_features]
         */
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < linear->in_features; i++) {
                float sum = 0.0f;
                for (int j = 0; j < linear->out_features; j++) {
                    sum += grad_output->grad[b * linear->out_features + j] *
                           self->weights->data[i * linear->out_features + j];
                }
                input->grad[b * linear->in_features + i] += sum;
            }
        }
    }
}

static void linear_free(cg_layer* self) {
    if (!self) return;
    
    if (self->weights) cg_tensor_release(self->weights);
    if (self->bias) cg_tensor_release(self->bias);
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    
    free(self);
}

static int linear_num_params(cg_layer* self) {
    return self->bias ? 2 : 1;
}

static cg_tensor** linear_get_params(cg_layer* self) {
    cg_linear* linear = (cg_linear*)self;
    
    linear->param_ptrs[0] = self->weights;
    linear->param_ptrs[1] = self->bias; // Can be NULL, which is fine as num_params handles it
    
    return linear->param_ptrs;
}

cg_linear* cg_linear_new(int in_features, int out_features, bool use_bias) {
    cg_linear* linear = (cg_linear*)calloc(1, sizeof(cg_linear));
    if (!linear) return NULL;
    
    linear->in_features = in_features;
    linear->out_features = out_features;
    
    /* Initialize base layer */
    cg_layer* base = (cg_layer*)linear;
    base->name = "Linear";
    base->forward = linear_forward;
    base->backward = linear_backward;
    base->free = linear_free;
    base->num_params = linear_num_params;
    base->get_params = linear_get_params;
    
    /* Create weight tensor: [in_features, out_features] */
    int weight_shape[] = {in_features, out_features};
    base->weights = cg_tensor_new(weight_shape, 2, true);
    if (!base->weights) {
        free(linear);
        return NULL;
    }
    
    /* Xavier initialization */
    srand(42);  /* Deterministic initialization */
    xavier_init(base->weights, in_features, out_features);
    
    /* Create bias tensor if needed */
    if (use_bias) {
        int bias_shape[] = {out_features};
        base->bias = cg_tensor_zeros(bias_shape, 1, true);
        if (!base->bias) {
            cg_tensor_free(base->weights);
            free(linear);
            return NULL;
        }
    }
    
    base->input = NULL;
    base->output = NULL;
    
    return linear;
}

/*============================================================================
 * LAYER UTILITIES
 *============================================================================*/

void cg_layer_free(cg_layer* layer) {
    if (layer && layer->free) {
        layer->free(layer);
    }
}

void cg_layer_zero_grad(cg_layer* layer) {
    if (!layer) return;
    
    if (layer->weights && layer->weights->grad) {
        memset(layer->weights->grad, 0, layer->weights->size * sizeof(float));
    }
    if (layer->bias && layer->bias->grad) {
        memset(layer->bias->grad, 0, layer->bias->size * sizeof(float));
    }
    if (layer->input && layer->input->grad) {
        memset(layer->input->grad, 0, layer->input->size * sizeof(float));
    }
}
