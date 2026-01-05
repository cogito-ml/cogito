/**
 * Tensor Unit Tests
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

void test_tensor_creation(void) {
    printf("\n=== Tensor Creation ===\n");
    
    int shape[] = {2, 3};
    
    cg_tensor* zeros = cg_tensor_zeros(shape, 2, false);
    ASSERT(zeros != NULL, "zeros creation");
    ASSERT(zeros->size == 6, "zeros size");
    ASSERT(zeros->data[0] == 0 && zeros->data[5] == 0, "zeros values");
    cg_tensor_free(zeros);
    
    cg_tensor* ones = cg_tensor_ones(shape, 2, false);
    ASSERT(ones != NULL, "ones creation");
    ASSERT(ones->data[0] == 1 && ones->data[5] == 1, "ones values");
    cg_tensor_free(ones);
    
    cg_tensor* full = cg_tensor_full(shape, 2, 3.14f, false);
    ASSERT(full != NULL, "full creation");
    ASSERT_NEAR(full->data[0], 3.14f, 0.001f, "full value");
    cg_tensor_free(full);
    
    cg_tensor* randn = cg_tensor_randn(shape, 2, 42, false);
    ASSERT(randn != NULL, "randn creation");
    ASSERT(randn->data[0] != randn->data[1], "randn has variance");
    cg_tensor_free(randn);
}

void test_tensor_operations(void) {
    printf("\n=== Tensor Operations ===\n");
    
    int shape[] = {2, 2};
    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    
    cg_tensor* a = cg_tensor_from_data(a_data, shape, 2, false);
    cg_tensor* b = cg_tensor_from_data(b_data, shape, 2, false);
    cg_tensor* c = cg_tensor_zeros(shape, 2, false);
    
    /* Add */
    cg_tensor_add(a, b, c);
    ASSERT_NEAR(c->data[0], 6.0f, 0.001f, "add [0]");
    ASSERT_NEAR(c->data[3], 12.0f, 0.001f, "add [3]");
    
    /* Mul */
    cg_tensor_mul(a, b, c);
    ASSERT_NEAR(c->data[0], 5.0f, 0.001f, "mul [0]");
    ASSERT_NEAR(c->data[3], 32.0f, 0.001f, "mul [3]");
    
    /* Matmul: [1,2;3,4] @ [5,6;7,8] = [19,22;43,50] */
    cg_tensor_matmul(a, b, c);
    ASSERT_NEAR(c->data[0], 19.0f, 0.001f, "matmul [0,0]");
    ASSERT_NEAR(c->data[1], 22.0f, 0.001f, "matmul [0,1]");
    ASSERT_NEAR(c->data[2], 43.0f, 0.001f, "matmul [1,0]");
    ASSERT_NEAR(c->data[3], 50.0f, 0.001f, "matmul [1,1]");
    
    cg_tensor_free(a);
    cg_tensor_free(b);
    cg_tensor_free(c);
}

void test_tensor_reductions(void) {
    printf("\n=== Tensor Reductions ===\n");
    
    int shape[] = {2, 3};
    float data[] = {1, 2, 3, 4, 5, 6};
    
    cg_tensor* t = cg_tensor_from_data(data, shape, 2, false);
    
    int scalar_shape[] = {1};
    cg_tensor* sum_all = cg_tensor_zeros(scalar_shape, 1, false);
    cg_tensor_sum(t, -1, sum_all);
    ASSERT_NEAR(sum_all->data[0], 21.0f, 0.001f, "sum all");
    
    cg_tensor* mean_all = cg_tensor_zeros(scalar_shape, 1, false);
    cg_tensor_mean(t, -1, mean_all);
    ASSERT_NEAR(mean_all->data[0], 3.5f, 0.001f, "mean all");
    
    cg_tensor_free(t);
    cg_tensor_free(sum_all);
    cg_tensor_free(mean_all);
}

void test_arena(void) {
    printf("\n=== Arena Allocator ===\n");
    
    cg_arena* arena = cg_arena_new(1024 * 1024);  /* 1MB */
    ASSERT(arena != NULL, "arena creation");
    
    void* ptr1 = cg_arena_alloc(arena, 1024, 16);
    ASSERT(ptr1 != NULL, "arena alloc 1KB");
    
    void* ptr2 = cg_arena_alloc(arena, 4096, 16);
    ASSERT(ptr2 != NULL, "arena alloc 4KB");
    
    size_t used = cg_arena_total_used(arena);
    ASSERT(used >= 5120, "arena tracks usage");
    
    cg_arena_clear(arena);
    ASSERT(cg_arena_total_used(arena) == 0, "arena clear");
    
    cg_arena_free(arena);
}

void test_layers(void) {
    printf("\n=== Neural Network Layers ===\n");
    
    /* Linear layer */
    cg_linear* linear = cg_linear_new(4, 2, true);
    ASSERT(linear != NULL, "linear creation");
    ASSERT(linear->base.weights != NULL, "linear has weights");
    ASSERT(linear->base.bias != NULL, "linear has bias");
    
    int in_shape[] = {3, 4};
    cg_tensor* input = cg_tensor_ones(in_shape, 2, true);
    cg_tensor* output = linear->base.forward((cg_layer*)linear, input);
    
    ASSERT(output != NULL, "linear forward");
    ASSERT(output->shape[0] == 3 && output->shape[1] == 2, "linear output shape");
    
    cg_tensor_release(input);
    cg_tensor_release(output);
    cg_layer_free((cg_layer*)linear);
    
    /* ReLU */
    cg_relu* relu = cg_relu_new();
    int relu_shape[] = {4};
    float relu_data[] = {-2, -1, 0, 1};
    cg_tensor* relu_in = cg_tensor_from_data(relu_data, relu_shape, 1, false);
    cg_tensor* relu_out = relu->base.forward((cg_layer*)relu, relu_in);
    
    ASSERT(relu_out->data[0] == 0, "relu(-2) = 0");
    ASSERT(relu_out->data[3] == 1, "relu(1) = 1");
    
    cg_tensor_release(relu_in);
    cg_tensor_release(relu_out);
    cg_layer_free((cg_layer*)relu);
}

int main(void) {
    printf("Cogito Tensor Tests\n");
    printf("===================\n");
    
    test_tensor_creation();
    test_tensor_operations();
    test_tensor_reductions();
    test_arena();
    test_layers();
    
    printf("\n===================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);
    
    return failures > 0 ? 1 : 0;
}
