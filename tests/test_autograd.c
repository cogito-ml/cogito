/**
 * Autograd Tests - Verify gradients using numerical differentiation
 */

#include "cogito.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else { printf("PASS: %s\n", msg); passes++; } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabsf((a) - (b)) < (tol), msg)

int passes = 0, failures = 0;

/* Numerical gradient using central differences */
float numerical_gradient(cg_tensor* (*forward_fn)(cg_tensor*), cg_tensor* x, int idx, float eps) {
    float orig = x->data[idx];
    
    x->data[idx] = orig + eps;
    cg_tensor* f_plus = forward_fn(x);
    float y_plus = f_plus->data[0];
    cg_tensor_free(f_plus);
    
    x->data[idx] = orig - eps;
    cg_tensor* f_minus = forward_fn(x);
    float y_minus = f_minus->data[0];
    cg_tensor_free(f_minus);
    
    x->data[idx] = orig;
    
    return (y_plus - y_minus) / (2.0f * eps);
}

/* Test functions */
static cg_tensor* sum_of_squares(cg_tensor* x) {
    int shape[] = {1};
    cg_tensor* out = cg_tensor_zeros(shape, 1, false);
    float sum = 0;
    for (int i = 0; i < x->size; i++) {
        sum += x->data[i] * x->data[i];
    }
    out->data[0] = sum;
    return out;
}

void test_mse_gradient(void) {
    printf("\n=== MSE Loss Gradient ===\n");
    
    int shape[] = {4};
    float pred_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float targ_data[] = {1.5f, 2.5f, 2.5f, 3.5f};
    
    cg_tensor* pred = cg_tensor_from_data(pred_data, shape, 1, true);
    cg_tensor* target = cg_tensor_from_data(targ_data, shape, 1, false);
    
    cg_tensor* loss = cg_mse_loss(pred, target, CG_REDUCTION_MEAN);
    
    /* Initialize loss gradient */
    loss->grad = (float*)calloc(1, sizeof(float));
    loss->grad[0] = 1.0f;
    
    /* Backward */
    if (loss->backward_fn) loss->backward_fn(loss);
    
    /* Expected: d/d_pred = 2 * (pred - target) / n */
    float expected_grad[4];
    for (int i = 0; i < 4; i++) {
        expected_grad[i] = 2.0f * (pred_data[i] - targ_data[i]) / 4.0f;
    }
    
    for (int i = 0; i < 4; i++) {
        char msg[64];
        sprintf(msg, "MSE grad[%d]", i);
        ASSERT_NEAR(pred->grad[i], expected_grad[i], 0.001f, msg);
    }
    
    cg_tensor_release(loss);
    cg_tensor_free(pred);
    cg_tensor_free(target);
}

void test_ce_gradient(void) {
    printf("\n=== Cross-Entropy Gradient ===\n");
    
    int logit_shape[] = {2, 3};  /* batch=2, classes=3 */
    float logit_data[] = {1.0f, 2.0f, 0.5f, 0.5f, 1.0f, 1.5f};
    
    int label_shape[] = {2};
    float label_data[] = {1, 2};  /* class indices */
    
    cg_tensor* logits = cg_tensor_from_data(logit_data, logit_shape, 2, true);
    cg_tensor* labels = cg_tensor_from_data(label_data, label_shape, 1, false);
    
    cg_tensor* loss = cg_softmax_cross_entropy_loss(logits, labels, CG_REDUCTION_MEAN);
    
    loss->grad = (float*)calloc(1, sizeof(float));
    loss->grad[0] = 1.0f;
    
    if (loss->backward_fn) loss->backward_fn(loss);
    
    /* For softmax cross-entropy, gradient is (softmax - one_hot) / batch_size */
    /* Verify gradients sum to ~0 for each sample (property of softmax grad) */
    float sum0 = logits->grad[0] + logits->grad[1] + logits->grad[2];
    float sum1 = logits->grad[3] + logits->grad[4] + logits->grad[5];
    
    ASSERT_NEAR(sum0, 0.0f, 0.01f, "CE grad sum sample 0");
    ASSERT_NEAR(sum1, 0.0f, 0.01f, "CE grad sum sample 1");
    
    /* True class should have negative gradient */
    ASSERT(logits->grad[1] < 0, "CE grad[0,1] negative (true class)");
    ASSERT(logits->grad[5] < 0, "CE grad[1,2] negative (true class)");
    
    cg_tensor_release(loss);
    cg_tensor_free(logits);
    cg_tensor_free(labels);
}

void test_linear_gradient(void) {
    printf("\n=== Linear Layer Gradient ===\n");
    
    cg_linear* linear = cg_linear_new(2, 2, true);
    
    /* Set known weights for testing */
    linear->base.weights->data[0] = 1.0f;
    linear->base.weights->data[1] = 0.0f;
    linear->base.weights->data[2] = 0.0f;
    linear->base.weights->data[3] = 1.0f;
    linear->base.bias->data[0] = 0.0f;
    linear->base.bias->data[1] = 0.0f;
    
    int in_shape[] = {2, 2};
    float in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    cg_tensor* input = cg_tensor_from_data(in_data, in_shape, 2, true);
    
    cg_tensor* output = linear->base.forward((cg_layer*)linear, input);
    
    /* With identity weights, output should equal input */
    ASSERT_NEAR(output->data[0], 1.0f, 0.01f, "Linear fwd [0]");
    ASSERT_NEAR(output->data[3], 4.0f, 0.01f, "Linear fwd [3]");
    
    /* Backward with unit gradient */
    int grad_shape[] = {2, 2};
    cg_tensor* grad_out = cg_tensor_ones(grad_shape, 2, false);
    
    cg_layer_zero_grad((cg_layer*)linear);
    linear->base.backward((cg_layer*)linear, grad_out);
    
    /* dW should be sum over batch of x */
    ASSERT_NEAR(linear->base.weights->grad[0], 4.0f, 0.01f, "Linear dW[0,0]");
    ASSERT_NEAR(linear->base.weights->grad[1], 4.0f, 0.01f, "Linear dW[0,1]");
    
    /* db should be sum over batch (= 2) */
    ASSERT_NEAR(linear->base.bias->grad[0], 2.0f, 0.01f, "Linear db[0]");
    
    cg_tensor_free(grad_out);
    cg_tensor_release(input);
    cg_tensor_release(output);
    cg_layer_free((cg_layer*)linear);
}

void test_numerical_gradient(void) {
    printf("\n=== Numerical Gradient Check ===\n");
    
    int shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};
    cg_tensor* x = cg_tensor_from_data(data, shape, 1, false);
    
    /* f(x) = sum(x^2), df/dx_i = 2*x_i */
    for (int i = 0; i < 3; i++) {
        float num_grad = numerical_gradient(sum_of_squares, x, i, 1e-4f);
        float expected = 2.0f * x->data[i];
        char msg[64];
        sprintf(msg, "Numerical grad[%d]", i);
        ASSERT_NEAR(num_grad, expected, 0.01f, msg);
    }
    
    cg_tensor_free(x);
}

int main(void) {
    printf("Cogito Autograd Tests\n");
    printf("=====================\n");
    
    test_mse_gradient();
    test_ce_gradient();
    test_linear_gradient();
    test_numerical_gradient();
    
    printf("\n=====================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);
    
    return failures > 0 ? 1 : 0;
}
