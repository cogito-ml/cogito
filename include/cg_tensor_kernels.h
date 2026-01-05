/**
 * CUDA Tensor Kernels
 * 
 * GPU implementations of core tensor operations.
 * Uses simulation stubs when compiled without CUDA.
 */

#ifndef CG_TENSOR_KERNELS_H
#define CG_TENSOR_KERNELS_H

#include "cg_tensor.h"
#include "cg_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * ELEMENT-WISE KERNEL LAUNCHERS
 *============================================================================*/

/**
 * GPU element-wise addition: out = a + b
 */
void cg_cuda_tensor_add(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * GPU element-wise subtraction: out = a - b
 */
void cg_cuda_tensor_sub(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * GPU element-wise multiplication: out = a * b
 */
void cg_cuda_tensor_mul(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * GPU element-wise division: out = a / b
 */
void cg_cuda_tensor_div(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * GPU scalar multiplication: out = a * scalar
 */
void cg_cuda_tensor_scale(cg_tensor* a, float scalar, cg_tensor* out);

/*============================================================================
 * UNARY KERNEL LAUNCHERS
 *============================================================================*/

/**
 * GPU exponential: out = exp(a)
 */
void cg_cuda_tensor_exp(cg_tensor* a, cg_tensor* out);

/**
 * GPU logarithm: out = log(a)
 */
void cg_cuda_tensor_log(cg_tensor* a, cg_tensor* out);

/**
 * GPU power: out = a^power
 */
void cg_cuda_tensor_pow(cg_tensor* a, float power, cg_tensor* out);

/**
 * GPU square root: out = sqrt(a)
 */
void cg_cuda_tensor_sqrt(cg_tensor* a, cg_tensor* out);

/**
 * GPU absolute value: out = |a|
 */
void cg_cuda_tensor_abs(cg_tensor* a, cg_tensor* out);

/*============================================================================
 * REDUCTION KERNEL LAUNCHERS
 *============================================================================*/

/**
 * GPU sum reduction.
 */
void cg_cuda_tensor_sum(cg_tensor* a, int axis, cg_tensor* out);

/**
 * GPU mean reduction.
 */
void cg_cuda_tensor_mean(cg_tensor* a, int axis, cg_tensor* out);

/**
 * GPU max reduction.
 */
void cg_cuda_tensor_max(cg_tensor* a, int axis, cg_tensor* out);

/**
 * GPU min reduction.
 */
void cg_cuda_tensor_min(cg_tensor* a, int axis, cg_tensor* out);

/*============================================================================
 * MATRIX OPERATIONS
 *============================================================================*/

/**
 * GPU matrix multiplication using cuBLAS.
 * out = a @ b
 */
void cg_cuda_tensor_matmul(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/**
 * GPU transpose.
 */
void cg_cuda_tensor_transpose(cg_tensor* a, cg_tensor* out);

/**
 * GPU batch matrix multiplication.
 * For tensors of shape [..., M, K] @ [..., K, N] -> [..., M, N]
 */
void cg_cuda_tensor_bmm(cg_tensor* a, cg_tensor* b, cg_tensor* out);

/*============================================================================
 * ACTIVATION KERNEL LAUNCHERS
 *============================================================================*/

/**
 * GPU ReLU: out = max(0, a)
 */
void cg_cuda_relu_forward(cg_tensor* input, cg_tensor* output);
void cg_cuda_relu_backward(cg_tensor* input, cg_tensor* grad_output, cg_tensor* grad_input);

/**
 * GPU Sigmoid: out = 1 / (1 + exp(-a))
 */
void cg_cuda_sigmoid_forward(cg_tensor* input, cg_tensor* output);
void cg_cuda_sigmoid_backward(cg_tensor* output, cg_tensor* grad_output, cg_tensor* grad_input);

/**
 * GPU Tanh: out = tanh(a)
 */
void cg_cuda_tanh_forward(cg_tensor* input, cg_tensor* output);
void cg_cuda_tanh_backward(cg_tensor* output, cg_tensor* grad_output, cg_tensor* grad_input);

/**
 * GPU Softmax (numerically stable).
 */
void cg_cuda_softmax_forward(cg_tensor* input, cg_tensor* output, int axis);
void cg_cuda_softmax_backward(cg_tensor* output, cg_tensor* grad_output, 
                               cg_tensor* grad_input, int axis);

/**
 * GPU GELU activation.
 */
void cg_cuda_gelu_forward(cg_tensor* input, cg_tensor* output);
void cg_cuda_gelu_backward(cg_tensor* input, cg_tensor* grad_output, cg_tensor* grad_input);

/*============================================================================
 * NORMALIZATION KERNELS
 *============================================================================*/

/**
 * GPU Layer Normalization.
 */
void cg_cuda_layernorm_forward(cg_tensor* input, cg_tensor* gamma, cg_tensor* beta,
                                cg_tensor* output, cg_tensor* mean, cg_tensor* rstd,
                                float eps, int normalized_shape);

void cg_cuda_layernorm_backward(cg_tensor* grad_output, cg_tensor* input,
                                 cg_tensor* gamma, cg_tensor* mean, cg_tensor* rstd,
                                 cg_tensor* grad_input, cg_tensor* grad_gamma,
                                 cg_tensor* grad_beta, int normalized_shape);

/**
 * GPU RMS Normalization.
 */
void cg_cuda_rmsnorm_forward(cg_tensor* input, cg_tensor* weight,
                              cg_tensor* output, float eps);

#ifdef __cplusplus
}
#endif

#endif /* CG_TENSOR_KERNELS_H */
