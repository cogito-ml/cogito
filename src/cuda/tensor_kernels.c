/**
 * CUDA Tensor Kernels Implementation
 * 
 * GPU implementations of core tensor operations.
 * Includes simulation fallback using CPU when CUDA is unavailable.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cg_tensor_kernels.h"
#include "cg_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef CG_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*============================================================================
 * CUDA KERNELS
 *============================================================================*/

__global__ void ker_add(float* a, float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void ker_sub(float* a, float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void ker_mul(float* a, float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void ker_div(float* a, float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void ker_scale(float* a, float scalar, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void ker_exp(float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(a[idx]);
    }
}

__global__ void ker_log(float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = logf(a[idx]);
    }
}

__global__ void ker_pow(float* a, float power, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(a[idx], power);
    }
}

__global__ void ker_sqrt(float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(a[idx]);
    }
}

__global__ void ker_abs(float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(a[idx]);
    }
}

/* Parallel reduction for sum */
__global__ void ker_sum_reduce(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    /* Load and reduce in shared memory */
    float val = 0.0f;
    if (idx < n) val += input[idx];
    if (idx + blockDim.x < n) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();
    
    /* Reduce in shared memory */
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    /* Warp-level reduction */
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/* Activation kernels */
__global__ void ker_relu_forward(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void ker_relu_backward(float* input, float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = input[idx] > 0.0f ? grad_out[idx] : 0.0f;
    }
}

__global__ void ker_sigmoid_forward(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void ker_sigmoid_backward(float* output, float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = output[idx];
        grad_in[idx] = grad_out[idx] * s * (1.0f - s);
    }
}

__global__ void ker_tanh_forward(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void ker_tanh_backward(float* output, float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = output[idx];
        grad_in[idx] = grad_out[idx] * (1.0f - t * t);
    }
}

__global__ void ker_gelu_forward(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
        float c = 0.7978845608f;  /* sqrt(2/pi) */
        float inner = c * (x + 0.044715f * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

#endif /* CG_USE_CUDA */

/*============================================================================
 * ELEMENT-WISE OPERATIONS (Simulation fallback)
 *============================================================================*/

#ifndef CG_USE_CUDA

/* Simulation implementations using CPU */
static void sim_add(float* a, float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

static void sim_sub(float* a, float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

static void sim_mul(float* a, float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

static void sim_div(float* a, float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
}

static void sim_scale(float* a, float scalar, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * scalar;
    }
}

static void sim_exp(float* a, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = expf(a[i]);
    }
}

static void sim_log(float* a, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = logf(a[i]);
    }
}

static void sim_pow(float* a, float power, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = powf(a[i], power);
    }
}

static void sim_sqrt(float* a, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = sqrtf(a[i]);
    }
}

static void sim_abs(float* a, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = fabsf(a[i]);
    }
}

#endif

/*============================================================================
 * PUBLIC API IMPLEMENTATIONS
 *============================================================================*/

void cg_cuda_tensor_add(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_add<<<grid, block>>>(a->data, b->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_add(a->data, b->data, out->data, n);
#endif
}

void cg_cuda_tensor_sub(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_sub<<<grid, block>>>(a->data, b->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_sub(a->data, b->data, out->data, n);
#endif
}

void cg_cuda_tensor_mul(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_mul<<<grid, block>>>(a->data, b->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_mul(a->data, b->data, out->data, n);
#endif
}

void cg_cuda_tensor_div(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_div<<<grid, block>>>(a->data, b->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_div(a->data, b->data, out->data, n);
#endif
}

void cg_cuda_tensor_scale(cg_tensor* a, float scalar, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_scale<<<grid, block>>>(a->data, scalar, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_scale(a->data, scalar, out->data, n);
#endif
}

void cg_cuda_tensor_exp(cg_tensor* a, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_exp<<<grid, block>>>(a->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_exp(a->data, out->data, n);
#endif
}

void cg_cuda_tensor_log(cg_tensor* a, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_log<<<grid, block>>>(a->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_log(a->data, out->data, n);
#endif
}

void cg_cuda_tensor_pow(cg_tensor* a, float power, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_pow<<<grid, block>>>(a->data, power, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_pow(a->data, power, out->data, n);
#endif
}

void cg_cuda_tensor_sqrt(cg_tensor* a, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_sqrt<<<grid, block>>>(a->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_sqrt(a->data, out->data, n);
#endif
}

void cg_cuda_tensor_abs(cg_tensor* a, cg_tensor* out) {
    int n = out->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_abs<<<grid, block>>>(a->data, out->data, n);
    cg_cuda_device_synchronize();
#else
    sim_abs(a->data, out->data, n);
#endif
}

/*============================================================================
 * REDUCTION OPERATIONS
 *============================================================================*/

void cg_cuda_tensor_sum(cg_tensor* a, int axis, cg_tensor* out) {
    if (axis < 0) {
        /* Sum all elements */
        float sum = 0.0f;
        for (int i = 0; i < a->size; i++) {
            sum += a->data[i];
        }
        out->data[0] = sum;
        return;
    }
    
    /* Axis-specific sum - use CPU implementation for now */
    memset(out->data, 0, out->size * sizeof(float));
    
    for (int i = 0; i < a->size; i++) {
        int remaining = i;
        int out_idx = 0;
        int out_dim = 0;
        
        for (int d = 0; d < a->ndim; d++) {
            int coord = remaining / a->strides[d];
            remaining = remaining % a->strides[d];
            
            if (d != axis) {
                out_idx += coord * out->strides[out_dim];
                out_dim++;
            }
        }
        
        out->data[out_idx] += a->data[i];
    }
}

void cg_cuda_tensor_mean(cg_tensor* a, int axis, cg_tensor* out) {
    cg_cuda_tensor_sum(a, axis, out);
    
    float divisor;
    if (axis < 0) {
        divisor = (float)a->size;
    } else {
        divisor = (float)a->shape[axis];
    }
    
    for (int i = 0; i < out->size; i++) {
        out->data[i] /= divisor;
    }
}

void cg_cuda_tensor_max(cg_tensor* a, int axis, cg_tensor* out) {
    if (axis < 0) {
        float max_val = a->data[0];
        for (int i = 1; i < a->size; i++) {
            if (a->data[i] > max_val) max_val = a->data[i];
        }
        out->data[0] = max_val;
        return;
    }
    
    /* Initialize to -inf */
    for (int i = 0; i < out->size; i++) {
        out->data[i] = -INFINITY;
    }
    
    for (int i = 0; i < a->size; i++) {
        int remaining = i;
        int out_idx = 0;
        int out_dim = 0;
        
        for (int d = 0; d < a->ndim; d++) {
            int coord = remaining / a->strides[d];
            remaining = remaining % a->strides[d];
            
            if (d != axis) {
                out_idx += coord * out->strides[out_dim];
                out_dim++;
            }
        }
        
        if (a->data[i] > out->data[out_idx]) {
            out->data[out_idx] = a->data[i];
        }
    }
}

void cg_cuda_tensor_min(cg_tensor* a, int axis, cg_tensor* out) {
    if (axis < 0) {
        float min_val = a->data[0];
        for (int i = 1; i < a->size; i++) {
            if (a->data[i] < min_val) min_val = a->data[i];
        }
        out->data[0] = min_val;
        return;
    }
    
    /* Initialize to +inf */
    for (int i = 0; i < out->size; i++) {
        out->data[i] = INFINITY;
    }
    
    for (int i = 0; i < a->size; i++) {
        int remaining = i;
        int out_idx = 0;
        int out_dim = 0;
        
        for (int d = 0; d < a->ndim; d++) {
            int coord = remaining / a->strides[d];
            remaining = remaining % a->strides[d];
            
            if (d != axis) {
                out_idx += coord * out->strides[out_dim];
                out_dim++;
            }
        }
        
        if (a->data[i] < out->data[out_idx]) {
            out->data[out_idx] = a->data[i];
        }
    }
}

/*============================================================================
 * MATRIX OPERATIONS
 *============================================================================*/

void cg_cuda_tensor_matmul(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];
    
#ifdef CG_USE_CUDA
    cg_cuda_context* ctx = cg_cuda_get_context();
    cublasHandle_t handle = (cublasHandle_t)ctx->cublas_handle;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    /* cuBLAS uses column-major, we use row-major */
    /* Compute C = A @ B as C^T = B^T @ A^T */
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                b->data, n,
                a->data, k,
                &beta,
                out->data, n);
    
    cg_cuda_device_synchronize();
#else
    /* CPU fallback - naive matmul */
    memset(out->data, 0, m * n * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a->data[i * k + p] * b->data[p * n + j];
            }
            out->data[i * n + j] = sum;
        }
    }
#endif
}

void cg_cuda_tensor_transpose(cg_tensor* a, cg_tensor* out) {
    int rows = a->shape[0];
    int cols = a->shape[1];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[j * rows + i] = a->data[i * cols + j];
        }
    }
}

void cg_cuda_tensor_bmm(cg_tensor* a, cg_tensor* b, cg_tensor* out) {
    /* Batch matrix multiplication */
    int batch_size = a->shape[0];
    int m = a->shape[a->ndim - 2];
    int k = a->shape[a->ndim - 1];
    int n = b->shape[b->ndim - 1];
    
    int a_stride = m * k;
    int b_stride = k * n;
    int out_stride = m * n;
    
    for (int batch = 0; batch < batch_size; batch++) {
        float* a_batch = a->data + batch * a_stride;
        float* b_batch = b->data + batch * b_stride;
        float* out_batch = out->data + batch * out_stride;
        
        /* Simple matmul for each batch */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += a_batch[i * k + p] * b_batch[p * n + j];
                }
                out_batch[i * n + j] = sum;
            }
        }
    }
}

/*============================================================================
 * ACTIVATION FUNCTIONS
 *============================================================================*/

void cg_cuda_relu_forward(cg_tensor* input, cg_tensor* output) {
    int n = input->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_relu_forward<<<grid, block>>>(input->data, output->data, n);
    cg_cuda_device_synchronize();
#else
    for (int i = 0; i < n; i++) {
        output->data[i] = fmaxf(0.0f, input->data[i]);
    }
#endif
}

void cg_cuda_relu_backward(cg_tensor* input, cg_tensor* grad_output, cg_tensor* grad_input) {
    int n = input->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_relu_backward<<<grid, block>>>(input->data, grad_output->data, grad_input->data, n);
    cg_cuda_device_synchronize();
#else
    for (int i = 0; i < n; i++) {
        grad_input->data[i] = input->data[i] > 0.0f ? grad_output->data[i] : 0.0f;
    }
#endif
}

void cg_cuda_sigmoid_forward(cg_tensor* input, cg_tensor* output) {
    int n = input->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_sigmoid_forward<<<grid, block>>>(input->data, output->data, n);
    cg_cuda_device_synchronize();
#else
    for (int i = 0; i < n; i++) {
        output->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
    }
#endif
}

void cg_cuda_sigmoid_backward(cg_tensor* output, cg_tensor* grad_output, cg_tensor* grad_input) {
    int n = output->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_sigmoid_backward<<<grid, block>>>(output->data, grad_output->data, grad_input->data, n);
    cg_cuda_device_synchronize();
#else
    for (int i = 0; i < n; i++) {
        float s = output->data[i];
        grad_input->data[i] = grad_output->data[i] * s * (1.0f - s);
    }
#endif
}

void cg_cuda_tanh_forward(cg_tensor* input, cg_tensor* output) {
    int n = input->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_tanh_forward<<<grid, block>>>(input->data, output->data, n);
    cg_cuda_device_synchronize();
#else
    for (int i = 0; i < n; i++) {
        output->data[i] = tanhf(input->data[i]);
    }
#endif
}

void cg_cuda_tanh_backward(cg_tensor* output, cg_tensor* grad_output, cg_tensor* grad_input) {
    int n = output->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_tanh_backward<<<grid, block>>>(output->data, grad_output->data, grad_input->data, n);
    cg_cuda_device_synchronize();
#else
    for (int i = 0; i < n; i++) {
        float t = output->data[i];
        grad_input->data[i] = grad_output->data[i] * (1.0f - t * t);
    }
#endif
}

void cg_cuda_softmax_forward(cg_tensor* input, cg_tensor* output, int axis) {
    /* Numerically stable softmax */
    int n = input->size;
    int axis_size = input->shape[axis];
    int outer_size = 1;
    int inner_size = 1;
    
    for (int i = 0; i < axis; i++) {
        outer_size *= input->shape[i];
    }
    for (int i = axis + 1; i < input->ndim; i++) {
        inner_size *= input->shape[i];
    }
    
    for (int outer = 0; outer < outer_size; outer++) {
        for (int inner = 0; inner < inner_size; inner++) {
            /* Find max for numerical stability */
            float max_val = -INFINITY;
            for (int i = 0; i < axis_size; i++) {
                int idx = outer * (axis_size * inner_size) + i * inner_size + inner;
                if (input->data[idx] > max_val) {
                    max_val = input->data[idx];
                }
            }
            
            /* Compute exp and sum */
            float sum = 0.0f;
            for (int i = 0; i < axis_size; i++) {
                int idx = outer * (axis_size * inner_size) + i * inner_size + inner;
                float val = expf(input->data[idx] - max_val);
                output->data[idx] = val;
                sum += val;
            }
            
            /* Normalize */
            for (int i = 0; i < axis_size; i++) {
                int idx = outer * (axis_size * inner_size) + i * inner_size + inner;
                output->data[idx] /= sum;
            }
        }
    }
}

void cg_cuda_softmax_backward(cg_tensor* output, cg_tensor* grad_output, 
                               cg_tensor* grad_input, int axis) {
    int axis_size = output->shape[axis];
    int outer_size = 1;
    int inner_size = 1;
    
    for (int i = 0; i < axis; i++) {
        outer_size *= output->shape[i];
    }
    for (int i = axis + 1; i < output->ndim; i++) {
        inner_size *= output->shape[i];
    }
    
    for (int outer = 0; outer < outer_size; outer++) {
        for (int inner = 0; inner < inner_size; inner++) {
            /* Compute sum(output * grad_output) for this row */
            float dot = 0.0f;
            for (int i = 0; i < axis_size; i++) {
                int idx = outer * (axis_size * inner_size) + i * inner_size + inner;
                dot += output->data[idx] * grad_output->data[idx];
            }
            
            /* grad_input = output * (grad_output - dot) */
            for (int i = 0; i < axis_size; i++) {
                int idx = outer * (axis_size * inner_size) + i * inner_size + inner;
                grad_input->data[idx] = output->data[idx] * (grad_output->data[idx] - dot);
            }
        }
    }
}

void cg_cuda_gelu_forward(cg_tensor* input, cg_tensor* output) {
    int n = input->size;
    
#ifdef CG_USE_CUDA
    int grid, block;
    cg_cuda_calculate_launch_dims(n, &grid, &block);
    ker_gelu_forward<<<grid, block>>>(input->data, output->data, n);
    cg_cuda_device_synchronize();
#else
    float c = 0.7978845608f;  /* sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float x = input->data[i];
        float inner = c * (x + 0.044715f * x * x * x);
        output->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#endif
}

void cg_cuda_gelu_backward(cg_tensor* input, cg_tensor* grad_output, cg_tensor* grad_input) {
    int n = input->size;
    float c = 0.7978845608f;
    
    for (int i = 0; i < n; i++) {
        float x = input->data[i];
        float x3 = x * x * x;
        float inner = c * (x + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float d_inner = c * (1.0f + 3.0f * 0.044715f * x * x);
        
        float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
        grad_input->data[i] = grad_output->data[i] * grad;
    }
}

/*============================================================================
 * NORMALIZATION
 *============================================================================*/

void cg_cuda_layernorm_forward(cg_tensor* input, cg_tensor* gamma, cg_tensor* beta,
                                cg_tensor* output, cg_tensor* mean, cg_tensor* rstd,
                                float eps, int normalized_shape) {
    int batch_size = input->size / normalized_shape;
    
    for (int b = 0; b < batch_size; b++) {
        float* x = input->data + b * normalized_shape;
        float* y = output->data + b * normalized_shape;
        
        /* Compute mean */
        float m = 0.0f;
        for (int i = 0; i < normalized_shape; i++) {
            m += x[i];
        }
        m /= normalized_shape;
        mean->data[b] = m;
        
        /* Compute variance */
        float var = 0.0f;
        for (int i = 0; i < normalized_shape; i++) {
            float diff = x[i] - m;
            var += diff * diff;
        }
        var /= normalized_shape;
        
        float r = 1.0f / sqrtf(var + eps);
        rstd->data[b] = r;
        
        /* Normalize and apply affine transform */
        for (int i = 0; i < normalized_shape; i++) {
            float normalized = (x[i] - m) * r;
            y[i] = gamma->data[i] * normalized + beta->data[i];
        }
    }
}

void cg_cuda_layernorm_backward(cg_tensor* grad_output, cg_tensor* input,
                                 cg_tensor* gamma, cg_tensor* mean, cg_tensor* rstd,
                                 cg_tensor* grad_input, cg_tensor* grad_gamma,
                                 cg_tensor* grad_beta, int normalized_shape) {
    int batch_size = input->size / normalized_shape;
    
    /* Zero grad_gamma and grad_beta */
    memset(grad_gamma->data, 0, normalized_shape * sizeof(float));
    memset(grad_beta->data, 0, normalized_shape * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        float* x = input->data + b * normalized_shape;
        float* dy = grad_output->data + b * normalized_shape;
        float* dx = grad_input->data + b * normalized_shape;
        float m = mean->data[b];
        float r = rstd->data[b];
        
        /* Accumulate grad_gamma and grad_beta */
        for (int i = 0; i < normalized_shape; i++) {
            float normalized = (x[i] - m) * r;
            grad_gamma->data[i] += dy[i] * normalized;
            grad_beta->data[i] += dy[i];
        }
        
        /* Compute grad_input */
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int i = 0; i < normalized_shape; i++) {
            sum1 += gamma->data[i] * dy[i];
            sum2 += gamma->data[i] * dy[i] * (x[i] - m);
        }
        
        for (int i = 0; i < normalized_shape; i++) {
            float normalized = (x[i] - m) * r;
            dx[i] = r * gamma->data[i] * dy[i];
            dx[i] -= r / normalized_shape * sum1;
            dx[i] -= normalized * r / normalized_shape * sum2 * r;
        }
    }
}

void cg_cuda_rmsnorm_forward(cg_tensor* input, cg_tensor* weight,
                              cg_tensor* output, float eps) {
    int normalized_shape = weight->size;
    int batch_size = input->size / normalized_shape;
    
    for (int b = 0; b < batch_size; b++) {
        float* x = input->data + b * normalized_shape;
        float* y = output->data + b * normalized_shape;
        
        /* Compute RMS */
        float rms = 0.0f;
        for (int i = 0; i < normalized_shape; i++) {
            rms += x[i] * x[i];
        }
        rms = sqrtf(rms / normalized_shape + eps);
        
        /* Normalize and scale */
        for (int i = 0; i < normalized_shape; i++) {
            y[i] = weight->data[i] * x[i] / rms;
        }
    }
}

#ifdef __cplusplus
}
#endif
