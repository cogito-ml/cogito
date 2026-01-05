/**
 * MNIST Example - Train a neural network on MNIST digits
 * 
 * Network: 784 -> 128 -> 64 -> 10
 * Requires MNIST dataset files in current directory.
 */

#include "cogito.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BATCH_SIZE 64
#define EPOCHS 5
#define LEARNING_RATE 0.001f

int compute_accuracy(cg_tensor* pred, cg_tensor* labels) {
    int correct = 0;
    int batch_size = pred->shape[0];
    int num_classes = pred->shape[1];
    
    for (int b = 0; b < batch_size; b++) {
        int pred_class = 0;
        float max_val = pred->data[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (pred->data[b * num_classes + c] > max_val) {
                max_val = pred->data[b * num_classes + c];
                pred_class = c;
            }
        }
        if (pred_class == (int)labels->data[b]) correct++;
    }
    
    return correct;
}

int main(int argc, char** argv) {
    printf("Cogito MNIST Example\n");
    printf("====================\n\n");
    
    /* Default paths */
    const char* train_images = "train-images.idx3-ubyte";
    const char* train_labels = "train-labels.idx1-ubyte";
    const char* test_images = "t10k-images.idx3-ubyte";
    const char* test_labels = "t10k-labels.idx1-ubyte";
    
    if (argc >= 5) {
        train_images = argv[1];
        train_labels = argv[2];
        test_images = argv[3];
        test_labels = argv[4];
    }
    
    /* Load datasets */
    printf("Loading MNIST dataset...\n");
    cg_mnist* train = cg_mnist_load(train_images, train_labels);
    cg_mnist* test = cg_mnist_load(test_images, test_labels);
    
    if (!train || !test) {
        printf("Error: Could not load MNIST dataset.\n");
        printf("Please download MNIST files from:\n");
        printf("  http://yann.lecun.com/exdb/mnist/\n");
        printf("And place them in the current directory.\n");
        if (train) cg_mnist_free(train);
        if (test) cg_mnist_free(test);
        return 1;
    }
    
    printf("  Train: %d samples\n", train->num_samples);
    printf("  Test:  %d samples\n\n", test->num_samples);
    
    /* Build model: 784 -> 128 -> 64 -> 10 */
    printf("Building model...\n");
    cg_sequential* model = cg_sequential_new();
    cg_sequential_add(model, (cg_layer*)cg_linear_new(784, 128, true));
    cg_sequential_add(model, (cg_layer*)cg_relu_new());
    cg_sequential_add(model, (cg_layer*)cg_linear_new(128, 64, true));
    cg_sequential_add(model, (cg_layer*)cg_relu_new());
    cg_sequential_add(model, (cg_layer*)cg_linear_new(64, 10, true));
    
    printf("  Parameters: %d tensors\n\n", cg_sequential_num_params(model));
    
    /* Adam optimizer */
    cg_adam* optimizer = cg_adam_new_for_sequential(
        model, LEARNING_RATE, 0.9f, 0.999f, 1e-8f, 0.0f
    );
    
    /* Training loop */
    printf("Training for %d epochs...\n\n", EPOCHS);
    
    clock_t start_time = clock();
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int total_correct = 0;
        int num_batches = train->num_samples / BATCH_SIZE;
        
        cg_data_iter* iter = cg_data_iter_new(
            train->images, train->labels, BATCH_SIZE, true, epoch
        );
        
        cg_tensor* batch_x;
        cg_tensor* batch_y;
        int batch_idx = 0;
        
        while (cg_data_iter_next(iter, &batch_x, &batch_y)) {
            /* Forward */
            batch_x->requires_grad = true;
            cg_tensor* pred = cg_sequential_forward(model, batch_x);
            
            /* Loss */
            cg_tensor* loss = cg_softmax_cross_entropy_loss(pred, batch_y, CG_REDUCTION_MEAN);
            total_loss += loss->data[0];
            total_correct += compute_accuracy(pred, batch_y);
            
            /* Zero grads */
            cg_optimizer_zero_grad((cg_optimizer*)optimizer);
            cg_sequential_zero_grad(model);
            
            /* Backward */
            if (!loss->grad) loss->grad = (float*)calloc(1, sizeof(float));
            loss->grad[0] = 1.0f;
            if (loss->backward_fn) loss->backward_fn(loss);
            cg_sequential_backward(model, pred);
            
            /* Update */
            cg_optimizer_step((cg_optimizer*)optimizer);
            
            /* Cleanup */
            cg_tensor_free(batch_x);
            cg_tensor_free(batch_y);
            cg_tensor_release(pred);
            cg_tensor_release(loss);
            
            batch_idx++;
        }
        
        float avg_loss = total_loss / num_batches;
        float train_acc = 100.0f * total_correct / (num_batches * BATCH_SIZE);
        
        printf("Epoch %d/%d | Loss: %.4f | Train Acc: %.2f%%\n",
               epoch + 1, EPOCHS, avg_loss, train_acc);
        
        cg_data_iter_free(iter);
    }
    
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("\nTraining completed in %.2f seconds.\n", elapsed);
    
    /* Evaluate on test set */
    printf("\nEvaluating on test set...\n");
    
    int test_correct = 0;
    int test_batches = test->num_samples / BATCH_SIZE;
    
    cg_data_iter* test_iter = cg_data_iter_new(
        test->images, test->labels, BATCH_SIZE, false, 0
    );
    
    cg_tensor* batch_x;
    cg_tensor* batch_y;
    
    while (cg_data_iter_next(test_iter, &batch_x, &batch_y)) {
        cg_tensor* pred = cg_sequential_forward(model, batch_x);
        test_correct += compute_accuracy(pred, batch_y);
        
        cg_tensor_free(batch_x);
        cg_tensor_free(batch_y);
        cg_tensor_release(pred);
    }
    
    float test_acc = 100.0f * test_correct / (test_batches * BATCH_SIZE);
    printf("Test Accuracy: %.2f%%\n", test_acc);
    
    /* Cleanup */
    cg_data_iter_free(test_iter);
    cg_mnist_free(train);
    cg_mnist_free(test);
    cg_sequential_free(model);
    cg_optimizer_free((cg_optimizer*)optimizer);
    
    printf("\nDone!\n");
    return 0;
}
