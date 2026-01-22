/**
 * Activation Layers Implementation
 * 
 * ReLU, Sigmoid, Tanh, Softmax, Dropout, BatchNorm
 */

#include "cg_layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/*============================================================================
 * SwiGLU ACTIVATION
 *============================================================================*/

static cg_tensor* swiglu_forward(cg_layer* self, cg_tensor* input) {
    /* Expects input to be output of [Gate | Value] projection */
    /* Input shape: [batch, 2 * dim] */
    
    return cg_swiglu_forward_split(input);
}

static void swiglu_backward(cg_layer* self, cg_tensor* grad_output) {
    /* Explicit backward not implemented yet, relies on autograd */
}

static void swiglu_free(cg_layer* self) {
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(self);
}

cg_tensor* cg_swiglu_forward_split(cg_tensor* input) {
    int last_dim = input->shape[input->ndim - 1];
    int half_dim = last_dim / 2;
    int batch = input->size / last_dim;
    
    int out_shape[CG_MAX_DIMS];
    memcpy(out_shape, input->shape, input->ndim * sizeof(int));
    out_shape[input->ndim - 1] = half_dim;
    
    cg_tensor* out = cg_tensor_new(out_shape, input->ndim, input->requires_grad);
    
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < half_dim; i++) {
            float gate = input->data[b * last_dim + i];
            float val  = input->data[b * last_dim + half_dim + i];
            float swish_gate = gate / (1.0f + expf(-gate));
            out->data[b * half_dim + i] = swish_gate * val;
        }
    }
    return out;
}

cg_swiglu* cg_swiglu_new(void) {
    cg_swiglu* l = (cg_swiglu*)calloc(1, sizeof(cg_swiglu));
    l->base.name = "SwiGLU";
    l->base.forward = swiglu_forward;
    l->base.free = swiglu_free;
    return l;
}

/*============================================================================
 * RELU ACTIVATION
 *============================================================================*/

static cg_tensor* relu_forward(cg_layer* self, cg_tensor* input) {
    cg_relu* relu = (cg_relu*)self;
    
    /* Create output tensor */
    cg_tensor* output = cg_tensor_new(input->shape, input->ndim, input->requires_grad);
    if (!output) return NULL;
    
    /* Create mask for backward */
    if (relu->mask) cg_tensor_free(relu->mask);
    relu->mask = cg_tensor_new(input->shape, input->ndim, false);
    
    /* Forward: max(0, x) and save mask */
    for (int i = 0; i < input->size; i++) {
        if (input->data[i] > 0) {
            output->data[i] = input->data[i];
            relu->mask->data[i] = 1.0f;
        } else {
            output->data[i] = 0.0f;
            relu->mask->data[i] = 0.0f;
        }
    }
    
    /* Save input reference */
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void relu_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_relu* relu = (cg_relu*)self;
    cg_tensor* input = self->input;
    
    if (!input || !input->requires_grad || !input->grad) return;
    
    /* Backward: grad_input = grad_output * mask */
    for (int i = 0; i < grad_output->size; i++) {
        input->grad[i] += grad_output->grad[i] * relu->mask->data[i];
    }
}

static void relu_free(cg_layer* self) {
    cg_relu* relu = (cg_relu*)self;
    if (relu->mask) cg_tensor_free(relu->mask);
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(relu);
}

cg_relu* cg_relu_new(void) {
    cg_relu* relu = (cg_relu*)calloc(1, sizeof(cg_relu));
    if (!relu) return NULL;
    
    cg_layer* base = (cg_layer*)relu;
    base->name = "ReLU";
    base->forward = relu_forward;
    base->backward = relu_backward;
    base->free = relu_free;
    base->num_params = NULL;
    base->get_params = NULL;
    base->weights = NULL;
    base->bias = NULL;
    
    relu->mask = NULL;
    
    return relu;
}

/*============================================================================
 * SIGMOID ACTIVATION
 *============================================================================*/

static cg_tensor* sigmoid_forward(cg_layer* self, cg_tensor* input) {
    cg_tensor* output = cg_tensor_new(input->shape, input->ndim, input->requires_grad);
    if (!output) return NULL;
    
    /* Forward: sigmoid(x) = 1 / (1 + exp(-x)) */
    for (int i = 0; i < input->size; i++) {
        output->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
    }
    
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void sigmoid_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_tensor* output = self->output;
    cg_tensor* input = self->input;
    
    if (!input || !input->requires_grad || !input->grad) return;
    
    /* Backward: grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x)) */
    for (int i = 0; i < grad_output->size; i++) {
        float s = output->data[i];
        input->grad[i] += grad_output->grad[i] * s * (1.0f - s);
    }
}

static void sigmoid_free(cg_layer* self) {
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(self);
}

cg_sigmoid* cg_sigmoid_new(void) {
    cg_sigmoid* sig = (cg_sigmoid*)calloc(1, sizeof(cg_sigmoid));
    if (!sig) return NULL;
    
    cg_layer* base = (cg_layer*)sig;
    base->name = "Sigmoid";
    base->forward = sigmoid_forward;
    base->backward = sigmoid_backward;
    base->free = sigmoid_free;
    base->weights = NULL;
    base->bias = NULL;
    
    return sig;
}

/*============================================================================
 * TANH ACTIVATION
 *============================================================================*/

static cg_tensor* tanh_forward(cg_layer* self, cg_tensor* input) {
    cg_tensor* output = cg_tensor_new(input->shape, input->ndim, input->requires_grad);
    if (!output) return NULL;
    
    for (int i = 0; i < input->size; i++) {
        output->data[i] = tanhf(input->data[i]);
    }
    
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void tanh_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_tensor* output = self->output;
    cg_tensor* input = self->input;
    
    if (!input || !input->requires_grad || !input->grad) return;
    
    /* Backward: grad_input = grad_output * (1 - tanh(x)^2) */
    for (int i = 0; i < grad_output->size; i++) {
        float t = output->data[i];
        input->grad[i] += grad_output->grad[i] * (1.0f - t * t);
    }
}

static void tanh_free(cg_layer* self) {
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(self);
}

cg_tanh_layer* cg_tanh_new(void) {
    cg_tanh_layer* t = (cg_tanh_layer*)calloc(1, sizeof(cg_tanh_layer));
    if (!t) return NULL;
    
    cg_layer* base = (cg_layer*)t;
    base->name = "Tanh";
    base->forward = tanh_forward;
    base->backward = tanh_backward;
    base->free = tanh_free;
    base->weights = NULL;
    base->bias = NULL;
    
    return t;
}

/*============================================================================
 * SOFTMAX ACTIVATION
 *============================================================================*/

static cg_tensor* softmax_forward(cg_layer* self, cg_tensor* input) {
    assert(input->ndim == 2);  /* [batch_size, num_classes] */
    
    cg_tensor* output = cg_tensor_new(input->shape, input->ndim, input->requires_grad);
    if (!output) return NULL;
    
    int batch_size = input->shape[0];
    int num_classes = input->shape[1];
    
    /* Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x))) */
    for (int b = 0; b < batch_size; b++) {
        float* in_row = input->data + b * num_classes;
        float* out_row = output->data + b * num_classes;
        
        /* Find max for numerical stability */
        float max_val = in_row[0];
        for (int c = 1; c < num_classes; c++) {
            if (in_row[c] > max_val) max_val = in_row[c];
        }
        
        /* Compute exp and sum */
        float sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            out_row[c] = expf(in_row[c] - max_val);
            sum += out_row[c];
        }
        
        /* Normalize */
        for (int c = 0; c < num_classes; c++) {
            out_row[c] /= sum;
        }
    }
    
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void softmax_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_tensor* output = self->output;
    cg_tensor* input = self->input;
    
    if (!input || !input->requires_grad || !input->grad) return;
    
    int batch_size = output->shape[0];
    int num_classes = output->shape[1];
    
    /* Jacobian of softmax: diag(s) - s @ s^T
     * For cross-entropy loss, this simplifies to: softmax - one_hot
     * But for general backward, we need the full Jacobian.
     */
    for (int b = 0; b < batch_size; b++) {
        float* s = output->data + b * num_classes;
        float* g = grad_output->grad + b * num_classes;
        float* dx = input->grad + b * num_classes;
        
        /* Compute dot product: sum(grad * softmax) */
        float dot = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            dot += g[c] * s[c];
        }
        
        /* dx = softmax * (grad - dot) */
        for (int c = 0; c < num_classes; c++) {
            dx[c] += s[c] * (g[c] - dot);
        }
    }
}

static void softmax_free(cg_layer* self) {
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(self);
}

cg_softmax* cg_softmax_new(void) {
    cg_softmax* sm = (cg_softmax*)calloc(1, sizeof(cg_softmax));
    if (!sm) return NULL;
    
    cg_layer* base = (cg_layer*)sm;
    base->name = "Softmax";
    base->forward = softmax_forward;
    base->backward = softmax_backward;
    base->free = softmax_free;
    base->weights = NULL;
    base->bias = NULL;
    
    return sm;
}

/*============================================================================
 * DROPOUT LAYER
 *============================================================================*/

static cg_tensor* dropout_forward(cg_layer* self, cg_tensor* input) {
    cg_dropout* dropout = (cg_dropout*)self;
    
    cg_tensor* output = cg_tensor_new(input->shape, input->ndim, input->requires_grad);
    if (!output) return NULL;
    
    if (dropout->training && dropout->p > 0.0f) {
        /* Create dropout mask */
        if (dropout->mask) cg_tensor_free(dropout->mask);
        dropout->mask = cg_tensor_new(input->shape, input->ndim, false);
        
        float scale = 1.0f / (1.0f - dropout->p);
        
        for (int i = 0; i < input->size; i++) {
            float r = (float)rand() / (float)RAND_MAX;
            if (r < dropout->p) {
                output->data[i] = 0.0f;
                dropout->mask->data[i] = 0.0f;
            } else {
                output->data[i] = input->data[i] * scale;
                dropout->mask->data[i] = scale;
            }
        }
    } else {
        /* Inference: just copy */
        memcpy(output->data, input->data, input->size * sizeof(float));
    }
    
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void dropout_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_dropout* dropout = (cg_dropout*)self;
    cg_tensor* input = self->input;
    
    if (!input || !input->requires_grad || !input->grad) return;
    
    if (dropout->training && dropout->mask) {
        for (int i = 0; i < grad_output->size; i++) {
            input->grad[i] += grad_output->grad[i] * dropout->mask->data[i];
        }
    } else {
        for (int i = 0; i < grad_output->size; i++) {
            input->grad[i] += grad_output->grad[i];
        }
    }
}

static void dropout_free(cg_layer* self) {
    cg_dropout* dropout = (cg_dropout*)self;
    if (dropout->mask) cg_tensor_free(dropout->mask);
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(dropout);
}

cg_dropout* cg_dropout_new(float p) {
    cg_dropout* dropout = (cg_dropout*)calloc(1, sizeof(cg_dropout));
    if (!dropout) return NULL;
    
    cg_layer* base = (cg_layer*)dropout;
    base->name = "Dropout";
    base->forward = dropout_forward;
    base->backward = dropout_backward;
    base->free = dropout_free;
    base->weights = NULL;
    base->bias = NULL;
    
    dropout->p = p;
    dropout->training = true;
    dropout->mask = NULL;
    
    return dropout;
}

void cg_dropout_set_training(cg_dropout* layer, bool training) {
    if (layer) layer->training = training;
}

/*============================================================================
 * BATCH NORMALIZATION
 *============================================================================*/

static cg_tensor* batchnorm_forward(cg_layer* self, cg_tensor* input) {
    cg_batchnorm* bn = (cg_batchnorm*)self;
    
    assert(input->ndim == 2);  /* [batch_size, num_features] */
    assert(input->shape[1] == bn->num_features);
    
    int batch_size = input->shape[0];
    int num_features = bn->num_features;
    
    cg_tensor* output = cg_tensor_new(input->shape, input->ndim, input->requires_grad);
    if (!output) return NULL;
    
    /* Free previous saved tensors */
    if (bn->input_normalized) cg_tensor_free(bn->input_normalized);
    if (bn->batch_mean) cg_tensor_free(bn->batch_mean);
    if (bn->batch_var) cg_tensor_free(bn->batch_var);
    
    int stat_shape[] = {num_features};
    bn->batch_mean = cg_tensor_zeros(stat_shape, 1, false);
    bn->batch_var = cg_tensor_zeros(stat_shape, 1, false);
    bn->input_normalized = cg_tensor_new(input->shape, input->ndim, false);
    
    if (bn->training) {
        /* Compute batch mean */
        for (int f = 0; f < num_features; f++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                sum += input->data[b * num_features + f];
            }
            bn->batch_mean->data[f] = sum / (float)batch_size;
        }
        
        /* Compute batch variance */
        for (int f = 0; f < num_features; f++) {
            float sum = 0.0f;
            float mean = bn->batch_mean->data[f];
            for (int b = 0; b < batch_size; b++) {
                float diff = input->data[b * num_features + f] - mean;
                sum += diff * diff;
            }
            bn->batch_var->data[f] = sum / (float)batch_size;
        }
        
        /* Update running statistics */
        for (int f = 0; f < num_features; f++) {
            bn->running_mean->data[f] = bn->momentum * bn->running_mean->data[f] +
                                         (1.0f - bn->momentum) * bn->batch_mean->data[f];
            bn->running_var->data[f] = bn->momentum * bn->running_var->data[f] +
                                        (1.0f - bn->momentum) * bn->batch_var->data[f];
        }
    }
    
    /* Normalize and scale */
    float* mean = bn->training ? bn->batch_mean->data : bn->running_mean->data;
    float* var = bn->training ? bn->batch_var->data : bn->running_var->data;
    
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < num_features; f++) {
            int idx = b * num_features + f;
            float x_norm = (input->data[idx] - mean[f]) / sqrtf(var[f] + bn->epsilon);
            bn->input_normalized->data[idx] = x_norm;
            output->data[idx] = self->weights->data[f] * x_norm + self->bias->data[f];
        }
    }
    
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void batchnorm_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_batchnorm* bn = (cg_batchnorm*)self;
    cg_tensor* input = self->input;
    
    if (!input) return;
    
    int batch_size = input->shape[0];
    int num_features = bn->num_features;
    
    /* Compute gradients for gamma (weights) and beta (bias) */
    if (self->weights->grad) {
        for (int f = 0; f < num_features; f++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                sum += grad_output->grad[b * num_features + f] *
                       bn->input_normalized->data[b * num_features + f];
            }
            self->weights->grad[f] += sum;
        }
    }
    
    if (self->bias->grad) {
        for (int f = 0; f < num_features; f++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                sum += grad_output->grad[b * num_features + f];
            }
            self->bias->grad[f] += sum;
        }
    }
    
    /* Compute input gradients */
    if (input->requires_grad && input->grad) {
        float n = (float)batch_size;
        
        for (int f = 0; f < num_features; f++) {
            float gamma = self->weights->data[f];
            float var = bn->batch_var->data[f];
            float std_inv = 1.0f / sqrtf(var + bn->epsilon);
            
            /* Sum of gradients */
            float sum_dout = 0.0f;
            float sum_dout_xnorm = 0.0f;
            
            for (int b = 0; b < batch_size; b++) {
                int idx = b * num_features + f;
                sum_dout += grad_output->grad[idx];
                sum_dout_xnorm += grad_output->grad[idx] * bn->input_normalized->data[idx];
            }
            
            /* dx = gamma * std_inv * (dout - mean(dout) - x_norm * mean(dout * x_norm)) / n */
            for (int b = 0; b < batch_size; b++) {
                int idx = b * num_features + f;
                float dout = grad_output->grad[idx];
                float x_norm = bn->input_normalized->data[idx];
                
                input->grad[idx] += gamma * std_inv * 
                    (dout - sum_dout / n - x_norm * sum_dout_xnorm / n);
            }
        }
    }
}

static void batchnorm_free(cg_layer* self) {
    cg_batchnorm* bn = (cg_batchnorm*)self;
    
    if (bn->running_mean) cg_tensor_free(bn->running_mean);
    if (bn->running_var) cg_tensor_free(bn->running_var);
    if (bn->input_normalized) cg_tensor_free(bn->input_normalized);
    if (bn->batch_mean) cg_tensor_free(bn->batch_mean);
    if (bn->batch_var) cg_tensor_free(bn->batch_var);
    
    if (self->weights) cg_tensor_release(self->weights);
    if (self->bias) cg_tensor_release(self->bias);
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    
    free(bn);
}

cg_batchnorm* cg_batchnorm_new(int num_features, float epsilon, float momentum) {
    cg_batchnorm* bn = (cg_batchnorm*)calloc(1, sizeof(cg_batchnorm));
    if (!bn) return NULL;
    
    cg_layer* base = (cg_layer*)bn;
    base->name = "BatchNorm";
    base->forward = batchnorm_forward;
    base->backward = batchnorm_backward;
    base->free = batchnorm_free;
    
    bn->num_features = num_features;
    bn->epsilon = epsilon > 0 ? epsilon : 1e-5f;
    bn->momentum = momentum > 0 ? momentum : 0.1f;
    bn->training = true;
    
    int shape[] = {num_features};
    
    /* Gamma (scale) - initialized to 1 */
    base->weights = cg_tensor_ones(shape, 1, true);
    /* Beta (shift) - initialized to 0 */
    base->bias = cg_tensor_zeros(shape, 1, true);
    
    /* Running statistics */
    bn->running_mean = cg_tensor_zeros(shape, 1, false);
    bn->running_var = cg_tensor_ones(shape, 1, false);
    
    bn->input_normalized = NULL;
    bn->batch_mean = NULL;
    bn->batch_var = NULL;
    
    return bn;
}

void cg_batchnorm_set_training(cg_batchnorm* layer, bool training) {
    if (layer) layer->training = training;
}

/*============================================================================
 * SEQUENTIAL CONTAINER
 *============================================================================*/

#define INITIAL_CAPACITY 8

cg_sequential* cg_sequential_new(void) {
    cg_sequential* seq = (cg_sequential*)calloc(1, sizeof(cg_sequential));
    if (!seq) return NULL;
    
    seq->layers = (cg_layer**)malloc(INITIAL_CAPACITY * sizeof(cg_layer*));
    seq->num_layers = 0;
    seq->capacity = INITIAL_CAPACITY;
    
    return seq;
}

void cg_sequential_add(cg_sequential* seq, cg_layer* layer) {
    assert(seq != NULL && layer != NULL);
    
    if (seq->num_layers >= seq->capacity) {
        seq->capacity *= 2;
        seq->layers = (cg_layer**)realloc(seq->layers, 
                                           seq->capacity * sizeof(cg_layer*));
    }
    
    seq->layers[seq->num_layers++] = layer;
}

cg_tensor* cg_sequential_forward(cg_sequential* seq, cg_tensor* input) {
    assert(seq != NULL && input != NULL);
    
    cg_tensor* x = input;
    
    for (int i = 0; i < seq->num_layers; i++) {
        x = seq->layers[i]->forward(seq->layers[i], x);
        if (!x) return NULL;
    }
    
    return x;
}

void cg_sequential_backward(cg_sequential* seq, cg_tensor* grad_output) {
    assert(seq != NULL);
    
    /* Backward in reverse order */
    for (int i = seq->num_layers - 1; i >= 0; i--) {
        cg_layer* layer = seq->layers[i];
        
        /* Get input gradient from this layer's output */
        cg_tensor* grad_in;
        if (i == seq->num_layers - 1) {
            grad_in = grad_output;
        } else {
            /* Use the saved input of the next layer, which has gradients */
            grad_in = seq->layers[i + 1]->input;
        }
        
        layer->backward(layer, grad_in);
    }
}

void cg_sequential_free(cg_sequential* seq) {
    if (!seq) return;
    
    for (int i = 0; i < seq->num_layers; i++) {
        cg_layer_free(seq->layers[i]);
    }
    
    free(seq->layers);
    free(seq);
}

void cg_sequential_zero_grad(cg_sequential* seq) {
    if (!seq) return;
    
    for (int i = 0; i < seq->num_layers; i++) {
        cg_layer_zero_grad(seq->layers[i]);
    }
}

int cg_sequential_num_params(cg_sequential* seq) {
    if (!seq) return 0;
    
    int count = 0;
    for (int i = 0; i < seq->num_layers; i++) {
        if (seq->layers[i]->weights) count++;
        if (seq->layers[i]->bias) count++;
    }
    
    return count;
}

cg_tensor** cg_sequential_get_params(cg_sequential* seq) {
    if (!seq) return NULL;
    
    int num_params = cg_sequential_num_params(seq);
    static cg_tensor* params[256];  /* Max 256 parameters */
    
    int idx = 0;
    for (int i = 0; i < seq->num_layers; i++) {
        if (seq->layers[i]->weights) {
            params[idx++] = seq->layers[i]->weights;
        }
        if (seq->layers[i]->bias) {
            params[idx++] = seq->layers[i]->bias;
        }
    }
    
    return params;
}
