/**
 * CUDA Kernel Unit Tests
 * 
 * Tests for GPU kernels using simulation mode.
 * Verifies correctness of tensor operations, Flash Attention, and activations.
 */

#include "cogito.h"
#include "cg_cuda.h"
#include "cg_tensor_kernels.h"
#include "cg_flash_attn_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else { printf("PASS: %s\n", msg); passes++; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabsf((a) - (b)) < (tol), msg)

static int passes = 0, failures = 0;

/*============================================================================
 * CUDA INITIALIZATION TESTS
 *============================================================================*/

void test_cuda_init(void) {
    printf("\n=== CUDA Initialization ===\n");
    
    bool initialized = cg_cuda_init();
    ASSERT(initialized, "CUDA init succeeds (simulation mode)");
    ASSERT(cg_cuda_is_available(), "CUDA is available after init");
    
    int device_count = cg_cuda_device_count();
    ASSERT(device_count >= 1, "At least one device available");
    
    cg_cuda_device_info info = cg_cuda_get_device_info(0);
    ASSERT(info.is_available, "Device 0 is available");
    printf("  Device: %s\n", info.device_name);
    printf("  Compute: %d.%d\n", info.compute_capability_major, info.compute_capability_minor);
    printf("  Memory: %zu MB\n", info.total_memory / (1024 * 1024));
}

/*============================================================================
 * MEMORY OPERATION TESTS
 *============================================================================*/

void test_cuda_memory(void) {
    printf("\n=== CUDA Memory Operations ===\n");
    
    /* Allocate GPU memory */
    size_t size = 1024 * sizeof(float);
    float* gpu_mem = (float*)cg_cuda_malloc(size);
    ASSERT(gpu_mem != NULL, "GPU malloc succeeds");
    
    /* Set memory */
    cg_cuda_memset(gpu_mem, 0, size);
    
    /* Host memory */
    float* host_mem = (float*)malloc(size);
    for (int i = 0; i < 1024; i++) {
        host_mem[i] = (float)i;
    }
    
    /* Copy to device */
    cg_cuda_memcpy(gpu_mem, host_mem, size, CG_MEMCPY_HOST_TO_DEVICE);
    
    /* Zero host memory */
    memset(host_mem, 0, size);
    
    /* Copy back */
    cg_cuda_memcpy(host_mem, gpu_mem, size, CG_MEMCPY_DEVICE_TO_HOST);
    
    /* Verify */
    ASSERT_NEAR(host_mem[0], 0.0f, 0.001f, "Memcpy round-trip [0]");
    ASSERT_NEAR(host_mem[512], 512.0f, 0.001f, "Memcpy round-trip [512]");
    ASSERT_NEAR(host_mem[1023], 1023.0f, 0.001f, "Memcpy round-trip [1023]");
    
    cg_cuda_free(gpu_mem);
    free(host_mem);
}

/*============================================================================
 * TENSOR GPU OPERATION TESTS
 *============================================================================*/

void test_tensor_cuda_ops(void) {
    printf("\n=== CUDA Tensor Operations ===\n");
    
    int shape[] = {4, 4};
    
    /* Create CPU tensors */
    float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float b_data[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    
    cg_tensor* a = cg_tensor_from_data(a_data, shape, 2, false);
    cg_tensor* b = cg_tensor_from_data(b_data, shape, 2, false);
    cg_tensor* out = cg_tensor_zeros(shape, 2, false);
    
    /* Test addition */
    cg_cuda_tensor_add(a, b, out);
    ASSERT_NEAR(out->data[0], 17.0f, 0.001f, "CUDA add [0]");
    ASSERT_NEAR(out->data[15], 17.0f, 0.001f, "CUDA add [15]");
    
    /* Test multiplication */
    cg_cuda_tensor_mul(a, b, out);
    ASSERT_NEAR(out->data[0], 16.0f, 0.001f, "CUDA mul [0]");
    ASSERT_NEAR(out->data[15], 16.0f, 0.001f, "CUDA mul [15]");
    
    /* Test scale */
    cg_cuda_tensor_scale(a, 2.0f, out);
    ASSERT_NEAR(out->data[0], 2.0f, 0.001f, "CUDA scale [0]");
    ASSERT_NEAR(out->data[15], 32.0f, 0.001f, "CUDA scale [15]");
    
    /* Test exp */
    float exp_data[] = {0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    cg_tensor* exp_in = cg_tensor_from_data(exp_data, shape, 2, false);
    cg_cuda_tensor_exp(exp_in, out);
    ASSERT_NEAR(out->data[0], 1.0f, 0.001f, "CUDA exp(0)");
    ASSERT_NEAR(out->data[1], expf(1.0f), 0.001f, "CUDA exp(1)");
    ASSERT_NEAR(out->data[2], expf(2.0f), 0.001f, "CUDA exp(2)");
    
    cg_tensor_free(a);
    cg_tensor_free(b);
    cg_tensor_free(out);
    cg_tensor_free(exp_in);
}

/*============================================================================
 * REDUCTION TESTS
 *============================================================================*/

void test_cuda_reductions(void) {
    printf("\n=== CUDA Reductions ===\n");
    
    int shape[] = {4, 4};
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    
    cg_tensor* t = cg_tensor_from_data(data, shape, 2, false);
    
    /* Sum all */
    int scalar_shape[] = {1};
    cg_tensor* sum_out = cg_tensor_zeros(scalar_shape, 1, false);
    cg_cuda_tensor_sum(t, -1, sum_out);
    ASSERT_NEAR(sum_out->data[0], 136.0f, 0.001f, "CUDA sum all");
    
    /* Mean all */
    cg_cuda_tensor_mean(t, -1, sum_out);
    ASSERT_NEAR(sum_out->data[0], 8.5f, 0.001f, "CUDA mean all");
    
    /* Max all */
    cg_cuda_tensor_max(t, -1, sum_out);
    ASSERT_NEAR(sum_out->data[0], 16.0f, 0.001f, "CUDA max all");
    
    /* Min all */
    cg_cuda_tensor_min(t, -1, sum_out);
    ASSERT_NEAR(sum_out->data[0], 1.0f, 0.001f, "CUDA min all");
    
    cg_tensor_free(t);
    cg_tensor_free(sum_out);
}

/*============================================================================
 * MATMUL TESTS
 *============================================================================*/

void test_cuda_matmul(void) {
    printf("\n=== CUDA Matrix Multiplication ===\n");
    
    int shape_a[] = {2, 3};
    int shape_b[] = {3, 2};
    int shape_c[] = {2, 2};
    
    /* A = [[1, 2, 3], [4, 5, 6]] */
    float a_data[] = {1, 2, 3, 4, 5, 6};
    
    /* B = [[7, 8], [9, 10], [11, 12]] */
    float b_data[] = {7, 8, 9, 10, 11, 12};
    
    cg_tensor* a = cg_tensor_from_data(a_data, shape_a, 2, false);
    cg_tensor* b = cg_tensor_from_data(b_data, shape_b, 2, false);
    cg_tensor* c = cg_tensor_zeros(shape_c, 2, false);
    
    /* C = A @ B = [[58, 64], [139, 154]] */
    cg_cuda_tensor_matmul(a, b, c);
    
    ASSERT_NEAR(c->data[0], 58.0f, 0.001f, "CUDA matmul [0,0]");
    ASSERT_NEAR(c->data[1], 64.0f, 0.001f, "CUDA matmul [0,1]");
    ASSERT_NEAR(c->data[2], 139.0f, 0.001f, "CUDA matmul [1,0]");
    ASSERT_NEAR(c->data[3], 154.0f, 0.001f, "CUDA matmul [1,1]");
    
    cg_tensor_free(a);
    cg_tensor_free(b);
    cg_tensor_free(c);
}

/*============================================================================
 * ACTIVATION TESTS
 *============================================================================*/

void test_cuda_activations(void) {
    printf("\n=== CUDA Activations ===\n");
    
    int shape[] = {8};
    float data[] = {-2, -1, -0.5, 0, 0.5, 1, 2, 3};
    
    cg_tensor* input = cg_tensor_from_data(data, shape, 1, false);
    cg_tensor* output = cg_tensor_zeros(shape, 1, false);
    
    /* ReLU */
    cg_cuda_relu_forward(input, output);
    ASSERT_NEAR(output->data[0], 0.0f, 0.001f, "CUDA ReLU(-2)");
    ASSERT_NEAR(output->data[3], 0.0f, 0.001f, "CUDA ReLU(0)");
    ASSERT_NEAR(output->data[6], 2.0f, 0.001f, "CUDA ReLU(2)");
    
    /* Sigmoid */
    cg_cuda_sigmoid_forward(input, output);
    ASSERT_NEAR(output->data[3], 0.5f, 0.001f, "CUDA Sigmoid(0)");
    ASSERT(output->data[0] < 0.2f, "CUDA Sigmoid(-2) < 0.2");
    ASSERT(output->data[7] > 0.9f, "CUDA Sigmoid(3) > 0.9");
    
    /* Tanh */
    cg_cuda_tanh_forward(input, output);
    ASSERT_NEAR(output->data[3], 0.0f, 0.001f, "CUDA Tanh(0)");
    ASSERT(output->data[0] < -0.9f, "CUDA Tanh(-2) < -0.9");
    ASSERT(output->data[6] > 0.9f, "CUDA Tanh(2) > 0.9");
    
    /* Softmax */
    int sm_shape[] = {2, 4};
    cg_tensor* sm_in = cg_tensor_from_data(data, sm_shape, 2, false);
    cg_tensor* sm_out = cg_tensor_zeros(sm_shape, 2, false);
    cg_cuda_softmax_forward(sm_in, sm_out, 1);
    
    /* Check that rows sum to 1 */
    float row0_sum = sm_out->data[0] + sm_out->data[1] + sm_out->data[2] + sm_out->data[3];
    float row1_sum = sm_out->data[4] + sm_out->data[5] + sm_out->data[6] + sm_out->data[7];
    ASSERT_NEAR(row0_sum, 1.0f, 0.001f, "CUDA Softmax row0 sum = 1");
    ASSERT_NEAR(row1_sum, 1.0f, 0.001f, "CUDA Softmax row1 sum = 1");
    
    cg_tensor_free(input);
    cg_tensor_free(output);
    cg_tensor_free(sm_in);
    cg_tensor_free(sm_out);
}

/*============================================================================
 * FLASH ATTENTION TESTS
 *============================================================================*/

void test_flash_attention(void) {
    printf("\n=== Flash Attention (Simulation) ===\n");
    
    /* Small test case: B=1, H=1, N=4, D=8 */
    int B = 1, H = 1, N = 4, D = 8;
    
    int q_shape[] = {B, N, H, D};
    int k_shape[] = {B, N, H, D};
    int v_shape[] = {B, N, H, D};
    int o_shape[] = {B, N, H, D};
    int lse_shape[] = {B, H, N};
    
    cg_tensor* Q = cg_tensor_randn(q_shape, 4, 42, false);
    cg_tensor* K = cg_tensor_randn(k_shape, 4, 43, false);
    cg_tensor* V = cg_tensor_randn(v_shape, 4, 44, false);
    cg_tensor* O = cg_tensor_zeros(o_shape, 4, false);
    cg_tensor* lse = cg_tensor_zeros(lse_shape, 3, false);
    
    /* Create params */
    cg_flash_attn_params params = {0};
    params.Q = Q->data;
    params.K = K->data;
    params.V = V->data;
    params.O = O->data;
    params.softmax_lse = lse->data;
    params.batch_size = B;
    params.seqlen_q = N;
    params.seqlen_k = N;
    params.n_heads = H;
    params.n_heads_k = H;
    params.d_head = D;
    params.scale = 1.0f / sqrtf((float)D);
    params.mask_type = ATTN_MASK_NONE;
    
    /* Run Flash Attention */
    cg_flash_attention_forward_cuda(&params);
    
    /* Basic sanity checks */
    ASSERT(O->data[0] != 0.0f, "Flash Attention produces non-zero output");
    ASSERT(lse->data[0] != 0.0f, "Flash Attention computes LSE");
    
    /* Check output is finite */
    bool all_finite = true;
    for (int i = 0; i < B * N * H * D; i++) {
        if (!isfinite(O->data[i])) {
            all_finite = false;
            break;
        }
    }
    ASSERT(all_finite, "Flash Attention output is finite");
    
    /* Test causal attention */
    params.mask_type = ATTN_MASK_CAUSAL;
    cg_tensor* O_causal = cg_tensor_zeros(o_shape, 4, false);
    params.O = O_causal->data;
    cg_flash_attention_forward_cuda(&params);
    ASSERT(O_causal->data[0] != 0.0f, "Causal Flash Attention works");
    
    cg_tensor_free(Q);
    cg_tensor_free(K);
    cg_tensor_free(V);
    cg_tensor_free(O);
    cg_tensor_free(O_causal);
    cg_tensor_free(lse);
}

/*============================================================================
 * ROPE TESTS
 *============================================================================*/

void test_rope(void) {
    printf("\n=== RoPE (Rotary Positional Embedding) ===\n");
    
    int batch = 1, seqlen = 4, n_heads = 2, d_head = 8;
    int half_d = d_head / 2;
    int max_seqlen = 16;
    
    /* Precompute frequencies */
    float* cos_cache = (float*)malloc(max_seqlen * half_d * sizeof(float));
    float* sin_cache = (float*)malloc(max_seqlen * half_d * sizeof(float));
    cg_rope_precompute_freqs(cos_cache, sin_cache, max_seqlen, d_head, 10000.0f);
    
    ASSERT(cos_cache[0] == 1.0f, "RoPE cos(0) = 1");
    ASSERT_NEAR(sin_cache[0], 0.0f, 0.001f, "RoPE sin(0) = 0");
    
    /* Create input tensor */
    int x_shape[] = {batch, seqlen, n_heads, d_head};
    cg_tensor* x = cg_tensor_ones(x_shape, 4, false);
    
    /* Apply RoPE */
    float* x_orig = (float*)malloc(x->size * sizeof(float));
    memcpy(x_orig, x->data, x->size * sizeof(float));
    
    cg_rope_forward_cuda(x->data, batch, seqlen, n_heads, d_head, 
                          cos_cache, sin_cache, 0);
    
    /* Check that values changed (rotation applied) */
    bool values_changed = false;
    for (int i = 0; i < x->size; i++) {
        if (fabsf(x->data[i] - x_orig[i]) > 0.001f) {
            values_changed = true;
            break;
        }
    }
    ASSERT(values_changed, "RoPE modifies input");
    
    /* All values should still be finite */
    bool all_finite = true;
    for (int i = 0; i < x->size; i++) {
        if (!isfinite(x->data[i])) {
            all_finite = false;
            break;
        }
    }
    ASSERT(all_finite, "RoPE output is finite");
    
    free(cos_cache);
    free(sin_cache);
    free(x_orig);
    cg_tensor_free(x);
}

/*============================================================================
 * LAYER NORMALIZATION TESTS
 *============================================================================*/

void test_layernorm(void) {
    printf("\n=== Layer Normalization ===\n");
    
    int batch = 2, features = 4;
    int x_shape[] = {batch, features};
    int stat_shape[] = {batch};
    int param_shape[] = {features};
    
    float x_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    
    cg_tensor* x = cg_tensor_from_data(x_data, x_shape, 2, false);
    cg_tensor* gamma = cg_tensor_ones(param_shape, 1, false);
    cg_tensor* beta = cg_tensor_zeros(param_shape, 1, false);
    cg_tensor* y = cg_tensor_zeros(x_shape, 2, false);
    cg_tensor* mean = cg_tensor_zeros(stat_shape, 1, false);
    cg_tensor* rstd = cg_tensor_zeros(stat_shape, 1, false);
    
    cg_cuda_layernorm_forward(x, gamma, beta, y, mean, rstd, 1e-5f, features);
    
    /* Check that output is normalized (mean ~0, var ~1) */
    float row0_mean = (y->data[0] + y->data[1] + y->data[2] + y->data[3]) / 4;
    float row0_var = 0;
    for (int i = 0; i < 4; i++) {
        row0_var += (y->data[i] - row0_mean) * (y->data[i] - row0_mean);
    }
    row0_var /= 4;
    
    ASSERT_NEAR(row0_mean, 0.0f, 0.01f, "LayerNorm output mean ~0");
    ASSERT_NEAR(row0_var, 1.0f, 0.1f, "LayerNorm output var ~1");
    
    cg_tensor_free(x);
    cg_tensor_free(gamma);
    cg_tensor_free(beta);
    cg_tensor_free(y);
    cg_tensor_free(mean);
    cg_tensor_free(rstd);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void) {
    printf("Cogito CUDA Kernel Tests\n");
    printf("========================\n");
    
    test_cuda_init();
    test_cuda_memory();
    test_tensor_cuda_ops();
    test_cuda_reductions();
    test_cuda_matmul();
    test_cuda_activations();
    test_flash_attention();
    test_rope();
    test_layernorm();
    
    /* Cleanup */
    cg_cuda_shutdown();
    
    printf("\n========================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);
    
    return failures > 0 ? 1 : 0;
}
