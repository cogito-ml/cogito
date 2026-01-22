/**
 * Normalization Layers
 */

#include "cg_layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * RMS NORM
 *============================================================================*/

static cg_tensor* rmsnorm_forward(cg_layer* layer, cg_tensor* input) {
    cg_rmsnorm* norm = (cg_rmsnorm*)layer;
    
    /* 1. Calculate RMS: sqrt(mean(x^2) + eps) */
    /* Implementation note: We compute on last dim */
    int batch = input->size / norm->normalized_shape;
    int dim = norm->normalized_shape;
    
    cg_tensor* inv_rms = cg_tensor_new(NULL, 0, false); /* Shape logic needed but storing flattened */
    /* Real implementation would construct proper shape [batch, 1] */
    int rms_shape[] = {batch, 1};
    cg_tensor_reshape(inv_rms, rms_shape, 2);
    
    for (int b = 0; b < batch; b++) {
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            float val = input->data[b * dim + i];
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / dim + norm->epsilon);
        inv_rms->data[b] = 1.0f / rms;
    }
    
    /* Store for backward */
    norm->inv_rms = inv_rms;
    norm->base.input = input; /* Retain input (should increase ref count) */
    cg_tensor_retain(input);
    
    /* 2. Scale: x * inv_rms * weight */
    cg_tensor* output = cg_tensor_new(input->shape, input->ndim, input->requires_grad);
    
    for (int b = 0; b < batch; b++) {
        float scale = inv_rms->data[b];
        for (int i = 0; i < dim; i++) {
            float w = norm->base.weights->data[i];
            output->data[b * dim + i] = input->data[b * dim + i] * scale * w;
        }
    }
    
    norm->base.output = output;
    return output;
}

static void rmsnorm_backward(cg_layer* layer, cg_tensor* grad_output) {
    cg_rmsnorm* norm = (cg_rmsnorm*)layer;
    cg_tensor* input = norm->base.input;
    cg_tensor* inv_rms = norm->inv_rms;
    
    int batch = input->size / norm->normalized_shape;
    int dim = norm->normalized_shape;
    
    /* Gradients for Weight */
    if (norm->base.weights->requires_grad) {
        cg_tensor_zero_grad(norm->base.weights);
        for (int b = 0; b < batch; b++) {
            float scale = inv_rms->data[b];
            for (int i = 0; i < dim; i++) {
                float val = input->data[b * dim + i];
                float go = grad_output->data[b * dim + i];
                /* dL/dw += dL/dy * x * inv_rms */
                norm->base.weights->grad[i] += go * val * scale;
            }
        }
    }
    
    /* Gradeint for Input (Simplified RMSProp Gradient) */
    /* dx = inv_rms * (dy - x * sum(dy * x) / (dim * E[x^2])) ... formula complex */
    if (input->requires_grad) {
        for (int b = 0; b < batch; b++) {
            float scale = inv_rms->data[b];
            float sum_grad_x = 0.0f;
            
            /* First pass: sum(dL/dy * x * w) */
            for (int i = 0; i < dim; i++) {
                float w = norm->base.weights->data[i];
                float go = grad_output->data[b * dim + i];
                float x = input->data[b * dim + i];
                sum_grad_x += go * w * x;
            }
            
            float factor = sum_grad_x * scale * scale / dim;
            
            for (int i = 0; i < dim; i++) {
                float w = norm->base.weights->data[i];
                float go = grad_output->data[b * dim + i];
                float x = input->data[b * dim + i];
                
                input->grad[b * dim + i] += (go * w - x * factor) * scale;
            }
        }
    }
}

static void rmsnorm_free(cg_layer* layer) {
    cg_rmsnorm* norm = (cg_rmsnorm*)layer;
    cg_tensor_free(norm->base.weights);
    cg_tensor_free(norm->inv_rms);
    if (norm->base.input) cg_tensor_release(norm->base.input);
    if (norm->base.output) cg_tensor_release(norm->base.output);
    free(norm);
}

cg_rmsnorm* cg_rmsnorm_new(int normalized_shape, float epsilon) {
    cg_rmsnorm* l = (cg_rmsnorm*)calloc(1, sizeof(cg_rmsnorm));
    l->base.name = "RMSNorm";
    l->normalized_shape = normalized_shape;
    l->epsilon = epsilon;
    
    l->base.forward = rmsnorm_forward;
    l->base.backward = rmsnorm_backward;
    l->base.free = rmsnorm_free;
    
    int w_shape[] = {normalized_shape};
    l->base.weights = cg_tensor_ones(w_shape, 1, true);
    
    return l;
}
