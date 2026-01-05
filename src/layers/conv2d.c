/**
 * Conv2D Layer with im2col for efficient convolution
 *
 * Converts convolution to matrix multiplication via image-to-column transform.
 */

#include "cg_layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/*============================================================================
 * IM2COL AND COL2IM TRANSFORMS
 *============================================================================*/

/**
 * im2col: Convert image patches to column matrix
 * 
 * Input:  (N, C, H, W)
 * Output: (C*kh*kw, out_h*out_w*N) for efficient GEMM
 */
static void im2col(float* input, int N, int C, int H, int W,
                   int kernel_h, int kernel_w, int stride, int padding,
                   float* output) {
    int out_h = (H + 2 * padding - kernel_h) / stride + 1;
    int out_w = (W + 2 * padding - kernel_w) / stride + 1;
    
    int col_idx = 0;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                for (int n = 0; n < N; n++) {
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            int h_pad = oh * stride + kh - padding;
                            int w_pad = ow * stride + kw - padding;
                            
                            if (h_pad >= 0 && h_pad < H && w_pad >= 0 && w_pad < W) {
                                int img_idx = ((n * C + c) * H + h_pad) * W + w_pad;
                                output[col_idx] = input[img_idx];
                            } else {
                                output[col_idx] = 0.0f;  /* Zero padding */
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
}

/**
 * col2im: Reverse of im2col, accumulates gradients back to image format
 */
static void col2im(float* col, int N, int C, int H, int W,
                   int kernel_h, int kernel_w, int stride, int padding,
                   float* grad_input) {
    int out_h = (H + 2 * padding - kernel_h) / stride + 1;
    int out_w = (W + 2 * padding - kernel_w) / stride + 1;
    
    memset(grad_input, 0, N * C * H * W * sizeof(float));
    
    int col_idx = 0;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                for (int n = 0; n < N; n++) {
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            int h_pad = oh * stride + kh - padding;
                            int w_pad = ow * stride + kw - padding;
                            
                            if (h_pad >= 0 && h_pad < H && w_pad >= 0 && w_pad < W) {
                                int img_idx = ((n * C + c) * H + h_pad) * W + w_pad;
                                grad_input[img_idx] += col[col_idx];
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
}

/*============================================================================
 * CONV2D LAYER
 *============================================================================*/

struct cg_conv2d {
    cg_layer base;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    float* col_buffer;           /* im2col buffer */
    int col_buffer_size;
};

static cg_tensor* conv2d_forward(cg_layer* self, cg_tensor* input) {
    cg_conv2d* conv = (cg_conv2d*)self;
    
    assert(input->ndim == 4);  /* (N, C, H, W) */
    assert(input->shape[1] == conv->in_channels);
    
    int N = input->shape[0];
    int C = conv->in_channels;
    int H = input->shape[2];
    int W = input->shape[3];
    int K = conv->out_channels;
    int kh = conv->kernel_size;
    int kw = conv->kernel_size;
    
    int out_h = (H + 2 * conv->padding - kh) / conv->stride + 1;
    int out_w = (W + 2 * conv->padding - kw) / conv->stride + 1;
    
    /* Allocate im2col buffer */
    int col_size = C * kh * kw * N * out_h * out_w;
    if (conv->col_buffer_size < col_size) {
        free(conv->col_buffer);
        conv->col_buffer = (float*)malloc(col_size * sizeof(float));
        conv->col_buffer_size = col_size;
    }
    
    /* im2col transform */
    im2col(input->data, N, C, H, W, kh, kw, conv->stride, conv->padding, conv->col_buffer);
    
    /* Create output: (N, K, out_h, out_w) */
    int out_shape[] = {N, K, out_h, out_w};
    cg_tensor* output = cg_tensor_new(out_shape, 4, input->requires_grad);
    
    /* GEMM: weight(K, C*kh*kw) @ col(C*kh*kw, N*out_h*out_w) -> (K, N*out_h*out_w) */
    int M = K;
    int inner = C * kh * kw;
    int Nu = N * out_h * out_w;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Nu; j++) {
            float sum = conv->base.bias ? conv->base.bias->data[i] : 0.0f;
            for (int k = 0; k < inner; k++) {
                sum += conv->base.weights->data[i * inner + k] * conv->col_buffer[k * Nu + j];
            }
            /* Rearrange to NCHW */
            int n = j / (out_h * out_w);
            int hw = j % (out_h * out_w);
            int h = hw / out_w;
            int w = hw % out_w;
            output->data[((n * K + i) * out_h + h) * out_w + w] = sum;
        }
    }
    
    /* Cache for backward */
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void conv2d_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_conv2d* conv = (cg_conv2d*)self;
    cg_tensor* input = self->input;
    
    if (!input) return;
    
    int N = input->shape[0];
    int C = conv->in_channels;
    int H = input->shape[2];
    int W = input->shape[3];
    int K = conv->out_channels;
    int kh = conv->kernel_size;
    int kw = conv->kernel_size;
    
    int out_h = (H + 2 * conv->padding - kh) / conv->stride + 1;
    int out_w = (W + 2 * conv->padding - kw) / conv->stride + 1;
    
    int inner = C * kh * kw;
    int Nu = N * out_h * out_w;
    
    /* Rearrange grad_output from NCHW to (K, N*out_h*out_w) */
    float* grad_col = (float*)malloc(K * Nu * sizeof(float));
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int src = ((n * K + k) * out_h + h) * out_w + w;
                    int dst = k * Nu + n * out_h * out_w + h * out_w + w;
                    grad_col[dst] = grad_output->data[src];
                }
            }
        }
    }
    
    /* grad_weights = grad_output @ col^T */
    if (self->weights->grad) {
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < inner; j++) {
                float sum = 0.0f;
                for (int k = 0; k < Nu; k++) {
                    sum += grad_col[i * Nu + k] * conv->col_buffer[j * Nu + k];
                }
                self->weights->grad[i * inner + j] += sum;
            }
        }
    }
    
    /* grad_bias = sum(grad_output) over spatial dims */
    if (self->bias && self->bias->grad) {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int j = 0; j < Nu; j++) {
                sum += grad_col[k * Nu + j];
            }
            self->bias->grad[k] += sum;
        }
    }
    
    /* grad_input via col2im */
    if (input->requires_grad && input->grad) {
        /* grad_col_input = weight^T @ grad_output: (inner, Nu) */
        float* grad_col_input = (float*)malloc(inner * Nu * sizeof(float));
        
        for (int i = 0; i < inner; i++) {
            for (int j = 0; j < Nu; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += self->weights->data[k * inner + i] * grad_col[k * Nu + j];
                }
                grad_col_input[i * Nu + j] = sum;
            }
        }
        
        col2im(grad_col_input, N, C, H, W, kh, kw, conv->stride, conv->padding, input->grad);
        free(grad_col_input);
    }
    
    free(grad_col);
}

static void conv2d_free(cg_layer* self) {
    cg_conv2d* conv = (cg_conv2d*)self;
    free(conv->col_buffer);
    if (self->weights) cg_tensor_release(self->weights);
    if (self->bias) cg_tensor_release(self->bias);
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(conv);
}

static int conv2d_num_params(cg_layer* self) {
    return self->bias ? 2 : 1;
}

static cg_tensor** conv2d_get_params(cg_layer* self) {
    static cg_tensor* params[2];
    int idx = 0;
    if (self->weights) params[idx++] = self->weights;
    if (self->bias) params[idx++] = self->bias;
    return params;
}

cg_conv2d* cg_conv2d_new(int in_channels, int out_channels, 
                         int kernel_size, int stride, int padding, bool use_bias) {
    cg_conv2d* conv = (cg_conv2d*)calloc(1, sizeof(cg_conv2d));
    if (!conv) return NULL;
    
    cg_layer* base = (cg_layer*)conv;
    base->name = "Conv2D";
    base->forward = conv2d_forward;
    base->backward = conv2d_backward;
    base->free = conv2d_free;
    base->num_params = conv2d_num_params;
    base->get_params = conv2d_get_params;
    
    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->col_buffer = NULL;
    conv->col_buffer_size = 0;
    
    /* Kaiming initialization for ReLU */
    int weight_shape[] = {out_channels, in_channels * kernel_size * kernel_size};
    base->weights = cg_tensor_randn(weight_shape, 2, 42, true);
    
    float std = sqrtf(2.0f / (float)(in_channels * kernel_size * kernel_size));
    for (int i = 0; i < base->weights->size; i++) {
        base->weights->data[i] *= std;
    }
    
    if (use_bias) {
        int bias_shape[] = {out_channels};
        base->bias = cg_tensor_zeros(bias_shape, 1, true);
    }
    
    return conv;
}

/*============================================================================
 * MAXPOOL2D LAYER
 *============================================================================*/

struct cg_maxpool2d {
    cg_layer base;
    int kernel_size;
    int stride;
    int* max_indices;  /* Store argmax for backward */
    int indices_size;
};

static cg_tensor* maxpool2d_forward(cg_layer* self, cg_tensor* input) {
    cg_maxpool2d* pool = (cg_maxpool2d*)self;
    
    assert(input->ndim == 4);
    
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int k = pool->kernel_size;
    int s = pool->stride;
    
    int out_h = (H - k) / s + 1;
    int out_w = (W - k) / s + 1;
    
    int out_shape[] = {N, C, out_h, out_w};
    cg_tensor* output = cg_tensor_new(out_shape, 4, input->requires_grad);
    
    /* Allocate max indices */
    int indices_size = N * C * out_h * out_w;
    if (pool->indices_size < indices_size) {
        free(pool->max_indices);
        pool->max_indices = (int*)malloc(indices_size * sizeof(int));
        pool->indices_size = indices_size;
    }
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float max_val = -INFINITY;
                    int max_idx = 0;
                    
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int h = oh * s + kh;
                            int w = ow * s + kw;
                            int idx = ((n * C + c) * H + h) * W + w;
                            if (input->data[idx] > max_val) {
                                max_val = input->data[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    
                    int out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
                    output->data[out_idx] = max_val;
                    pool->max_indices[out_idx] = max_idx;
                }
            }
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

static void maxpool2d_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_maxpool2d* pool = (cg_maxpool2d*)self;
    cg_tensor* input = self->input;
    
    if (!input || !input->requires_grad || !input->grad) return;
    
    for (int i = 0; i < grad_output->size; i++) {
        int max_idx = pool->max_indices[i];
        input->grad[max_idx] += grad_output->data[i];
    }
}

static void maxpool2d_free(cg_layer* self) {
    cg_maxpool2d* pool = (cg_maxpool2d*)self;
    free(pool->max_indices);
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(pool);
}

cg_maxpool2d* cg_maxpool2d_new(int kernel_size, int stride) {
    cg_maxpool2d* pool = (cg_maxpool2d*)calloc(1, sizeof(cg_maxpool2d));
    if (!pool) return NULL;
    
    cg_layer* base = (cg_layer*)pool;
    base->name = "MaxPool2D";
    base->forward = maxpool2d_forward;
    base->backward = maxpool2d_backward;
    base->free = maxpool2d_free;
    base->weights = NULL;
    base->bias = NULL;
    
    pool->kernel_size = kernel_size;
    pool->stride = stride > 0 ? stride : kernel_size;
    pool->max_indices = NULL;
    pool->indices_size = 0;
    
    return pool;
}

/*============================================================================
 * FLATTEN LAYER
 *============================================================================*/

struct cg_flatten {
    cg_layer base;
    int original_shape[CG_MAX_DIMS];
    int original_ndim;
};

static cg_tensor* flatten_forward(cg_layer* self, cg_tensor* input) {
    cg_flatten* flat = (cg_flatten*)self;
    
    /* Save original shape */
    flat->original_ndim = input->ndim;
    memcpy(flat->original_shape, input->shape, input->ndim * sizeof(int));
    
    /* Flatten to (N, features) */
    int N = input->shape[0];
    int features = input->size / N;
    int out_shape[] = {N, features};
    
    cg_tensor* output = cg_tensor_new(out_shape, 2, input->requires_grad);
    memcpy(output->data, input->data, input->size * sizeof(float));
    
    if (self->input) cg_tensor_release(self->input);
    self->input = input;
    cg_tensor_retain(input);
    
    if (self->output) cg_tensor_release(self->output);
    self->output = output;
    cg_tensor_retain(output);
    
    return output;
}

static void flatten_backward(cg_layer* self, cg_tensor* grad_output) {
    cg_tensor* input = self->input;
    
    if (!input || !input->requires_grad || !input->grad) return;
    
    /* Just copy gradients (reshape is a view operation) */
    for (int i = 0; i < grad_output->size; i++) {
        input->grad[i] += grad_output->data[i];
    }
}

static void flatten_free(cg_layer* self) {
    if (self->input) cg_tensor_release(self->input);
    if (self->output) cg_tensor_release(self->output);
    free(self);
}

cg_flatten* cg_flatten_new(void) {
    cg_flatten* flat = (cg_flatten*)calloc(1, sizeof(cg_flatten));
    if (!flat) return NULL;
    
    cg_layer* base = (cg_layer*)flat;
    base->name = "Flatten";
    base->forward = flatten_forward;
    base->backward = flatten_backward;
    base->free = flatten_free;
    base->weights = NULL;
    base->bias = NULL;
    
    return flat;
}
