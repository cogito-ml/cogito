/**
 * XOR Example - Learn XOR function with a 2-layer neural network
 */

#include "cogito.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(void) {
    printf("Cogito XOR Example\n");
    printf("==================\n\n");
    
    srand(42);
    
    /* XOR dataset */
    float X_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float y_data[] = {0, 1, 1, 0};
    
    int X_shape[] = {4, 2};
    int y_shape[] = {4, 1};
    
    cg_tensor* X = cg_tensor_from_data(X_data, X_shape, 2, true);
    cg_tensor* y = cg_tensor_from_data(y_data, y_shape, 2, false);
    
    /* Build layers with Tanh (better than ReLU for XOR) */
    cg_linear* fc1 = cg_linear_new(2, 8, true);
    cg_tanh_layer* tanh1 = cg_tanh_new();
    cg_linear* fc2 = cg_linear_new(8, 1, true);
    cg_sigmoid* sigmoid = cg_sigmoid_new();
    
    /* Collect parameters */
    cg_tensor* params[] = {fc1->base.weights, fc1->base.bias, 
                           fc2->base.weights, fc2->base.bias};
    cg_sgd* optimizer = cg_sgd_new(params, 4, 0.5f, 0.0f, 0.0f, false);
    
    int epochs = 5000;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        /* Forward pass */
        cg_tensor* h1 = fc1->base.forward((cg_layer*)fc1, X);
        cg_tensor* a1 = tanh1->base.forward((cg_layer*)tanh1, h1);
        cg_tensor* h2 = fc2->base.forward((cg_layer*)fc2, a1);
        cg_tensor* pred = sigmoid->base.forward((cg_layer*)sigmoid, h2);
        
        /* Compute MSE loss manually */
        float loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            float diff = pred->data[i] - y->data[i];
            loss += diff * diff;
        }
        loss /= 4.0f;
        
        if (epoch % 500 == 0 || epoch == epochs - 1) {
            printf("Epoch %4d | Loss: %.6f\n", epoch, loss);
        }
        
        /* Zero gradients */
        cg_layer_zero_grad((cg_layer*)fc1);
        cg_layer_zero_grad((cg_layer*)fc2);
        
        /* Create gradient tensor for pred (output of sigmoid) */
        cg_tensor* grad_pred = cg_tensor_new(pred->shape, pred->ndim, false);
        for (int i = 0; i < 4; i++) {
            grad_pred->data[i] = 2.0f * (pred->data[i] - y->data[i]) / 4.0f;
        }
        
        /* Set gradients on output tensors for backward */
        for (int i = 0; i < pred->size; i++) pred->grad[i] = grad_pred->data[i];
        
        /* Backward through each layer - pass the OUTPUT tensor (with grad set) */
        sigmoid->base.backward((cg_layer*)sigmoid, pred);
        
        /* Now h2->grad should have gradients - copy to grad tensor for fc2 */
        cg_tensor* grad_h2 = cg_tensor_new(h2->shape, h2->ndim, false);
        memcpy(grad_h2->data, h2->grad, h2->size * sizeof(float));
        
        fc2->base.backward((cg_layer*)fc2, grad_h2);
        
        /* a1->grad now has gradients */
        cg_tensor* grad_a1 = cg_tensor_new(a1->shape, a1->ndim, false);
        memcpy(grad_a1->data, a1->grad, a1->size * sizeof(float));
        
        tanh1->base.backward((cg_layer*)tanh1, grad_a1);
        
        /* h1->grad now has gradients */
        cg_tensor* grad_h1 = cg_tensor_new(h1->shape, h1->ndim, false);
        memcpy(grad_h1->data, h1->grad, h1->size * sizeof(float));
        
        fc1->base.backward((cg_layer*)fc1, grad_h1);
        
        /* Update */
        cg_optimizer_step((cg_optimizer*)optimizer);
        
        /* Cleanup */
        cg_tensor_free(grad_pred);
        cg_tensor_free(grad_h2);
        cg_tensor_free(grad_a1);
        cg_tensor_free(grad_h1);
        cg_tensor_release(h1);
        cg_tensor_release(a1);
        cg_tensor_release(h2);
        cg_tensor_release(pred);
    }
    
    /* Final predictions */
    printf("\nFinal Predictions:\n");
    cg_tensor* h1 = fc1->base.forward((cg_layer*)fc1, X);
    cg_tensor* a1 = tanh1->base.forward((cg_layer*)tanh1, h1);
    cg_tensor* h2 = fc2->base.forward((cg_layer*)fc2, a1);
    cg_tensor* final_pred = sigmoid->base.forward((cg_layer*)sigmoid, h2);
    
    for (int i = 0; i < 4; i++) {
        printf("  [%d, %d] -> %.4f (expected: %.0f)\n",
               (int)X->data[i * 2], (int)X->data[i * 2 + 1],
               final_pred->data[i], y->data[i]);
    }
    
    /* Cleanup */
    cg_tensor_release(h1);
    cg_tensor_release(a1);
    cg_tensor_release(h2);
    cg_tensor_release(final_pred);
    cg_tensor_free(X);
    cg_tensor_free(y);
    cg_layer_free((cg_layer*)fc1);
    cg_layer_free((cg_layer*)tanh1);
    cg_layer_free((cg_layer*)fc2);
    cg_layer_free((cg_layer*)sigmoid);
    cg_optimizer_free((cg_optimizer*)optimizer);
    
    printf("\nDone!\n");
    return 0;
}
