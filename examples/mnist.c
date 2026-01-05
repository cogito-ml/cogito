/**
 * MNIST MLP Example - Train a simple MLP on MNIST digits (or synthetic data)
 * 
 * Architecture: 784 -> 128 -> 64 -> 10
 */

#include "cogito.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ReLU in-place */
static void relu_inplace(float* data, int size, float* mask) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
            if (mask) mask[i] = 0.0f;
        } else {
            if (mask) mask[i] = 1.0f;
        }
    }
}

/* Softmax cross-entropy loss and gradient */
static float cross_entropy_loss(float* logits, int* labels, int batch_size, 
                                int num_classes, float* grad) {
    float loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        float max_val = logits[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (logits[b * num_classes + c] > max_val) {
                max_val = logits[b * num_classes + c];
            }
        }
        
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(logits[b * num_classes + c] - max_val);
        }
        
        for (int c = 0; c < num_classes; c++) {
            float softmax = expf(logits[b * num_classes + c] - max_val) / sum_exp;
            grad[b * num_classes + c] = softmax;
            if (c == labels[b]) {
                loss -= logf(softmax + 1e-10f);
                grad[b * num_classes + c] -= 1.0f;
            }
        }
    }
    
    loss /= batch_size;
    for (int i = 0; i < batch_size * num_classes; i++) {
        grad[i] /= batch_size;
    }
    
    return loss;
}

int main(int argc, char* argv[]) {
    printf("Cogito MNIST MLP Example\n");
    printf("========================\n\n");
    
    /* Load or create data */
    const char* train_images = "data/train-images.idx3-ubyte";
    const char* train_labels = "data/train-labels.idx1-ubyte";
    
    printf("Loading MNIST from %s...\n", train_images);
    cg_mnist* mnist = cg_mnist_load(train_images, train_labels);
    
    if (!mnist) {
        printf("MNIST not found. Using synthetic data (100 samples)...\n\n");
        
        int img_shape[] = {100, 784};
        int lbl_shape[] = {100};
        mnist = (cg_mnist*)calloc(1, sizeof(cg_mnist));
        mnist->images = cg_tensor_randn(img_shape, 2, 42, false);
        mnist->labels = cg_tensor_zeros(lbl_shape, 1, false);
        mnist->num_samples = 100;
        
        for (int i = 0; i < 100; i++) {
            mnist->labels->data[i] = (float)(i % 10);
        }
        
        /* Normalize */
        for (int i = 0; i < mnist->images->size; i++) {
            mnist->images->data[i] = (mnist->images->data[i] + 3.0f) / 6.0f;
            if (mnist->images->data[i] < 0) mnist->images->data[i] = 0;
            if (mnist->images->data[i] > 1) mnist->images->data[i] = 1;
        }
    }
    
    printf("Loaded %d samples\n\n", mnist->num_samples);
    
    /* Hyperparams */
    int batch_size = 32;
    int epochs = 5;
    int num_classes = 10;
    float lr = 0.01f;
    
    /* Weight shapes */
    int in1 = 784, out1 = 128;
    int in2 = 128, out2 = 64;
    int in3 = 64, out3 = 10;
    
    /* Allocate weights and biases */
    float* W1 = (float*)malloc(in1 * out1 * sizeof(float));
    float* b1 = (float*)calloc(out1, sizeof(float));
    float* W2 = (float*)malloc(in2 * out2 * sizeof(float));
    float* b2 = (float*)calloc(out2, sizeof(float));
    float* W3 = (float*)malloc(in3 * out3 * sizeof(float));
    float* b3 = (float*)calloc(out3, sizeof(float));
    
    float* dW1 = (float*)malloc(in1 * out1 * sizeof(float));
    float* db1 = (float*)malloc(out1 * sizeof(float));
    float* dW2 = (float*)malloc(in2 * out2 * sizeof(float));
    float* db2 = (float*)malloc(out2 * sizeof(float));
    float* dW3 = (float*)malloc(in3 * out3 * sizeof(float));
    float* db3 = (float*)malloc(out3 * sizeof(float));
    
    /* Xavier init */
    srand(42);
    for (int i = 0; i < in1 * out1; i++) 
        W1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrtf(6.0f / (in1 + out1));
    for (int i = 0; i < in2 * out2; i++) 
        W2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrtf(6.0f / (in2 + out2));
    for (int i = 0; i < in3 * out3; i++) 
        W3[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrtf(6.0f / (in3 + out3));
    
    /* Allocate intermediate buffers */
    int max_bs = batch_size;
    float* h1 = (float*)malloc(max_bs * out1 * sizeof(float));
    float* mask1 = (float*)malloc(max_bs * out1 * sizeof(float));
    float* h2 = (float*)malloc(max_bs * out2 * sizeof(float));
    float* mask2 = (float*)malloc(max_bs * out2 * sizeof(float));
    float* logits = (float*)malloc(max_bs * out3 * sizeof(float));
    float* dlogits = (float*)malloc(max_bs * out3 * sizeof(float));
    float* dh2 = (float*)malloc(max_bs * out2 * sizeof(float));
    float* dh1 = (float*)malloc(max_bs * out1 * sizeof(float));
    
    int num_batches = (mnist->num_samples + batch_size - 1) / batch_size;
    printf("Training: %d epochs, batch=%d, %d batches/epoch\n\n", epochs, batch_size, num_batches);
    
    clock_t start = clock();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct = 0, total = 0;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int bs_start = batch * batch_size;
            int bs = (bs_start + batch_size > mnist->num_samples) ? 
                     (mnist->num_samples - bs_start) : batch_size;
            
            float* X = mnist->images->data + bs_start * 784;
            float* Y = mnist->labels->data + bs_start;
            
            /* Zero grads */
            memset(dW1, 0, in1 * out1 * sizeof(float));
            memset(db1, 0, out1 * sizeof(float));
            memset(dW2, 0, in2 * out2 * sizeof(float));
            memset(db2, 0, out2 * sizeof(float));
            memset(dW3, 0, in3 * out3 * sizeof(float));
            memset(db3, 0, out3 * sizeof(float));
            
            /* Forward: h1 = ReLU(X @ W1 + b1) */
            for (int b = 0; b < bs; b++) {
                for (int j = 0; j < out1; j++) {
                    float sum = b1[j];
                    for (int k = 0; k < 784; k++) {
                        sum += X[b * 784 + k] * W1[k * out1 + j];
                    }
                    h1[b * out1 + j] = sum;
                }
            }
            relu_inplace(h1, bs * out1, mask1);
            
            /* Forward: h2 = ReLU(h1 @ W2 + b2) */
            for (int b = 0; b < bs; b++) {
                for (int j = 0; j < out2; j++) {
                    float sum = b2[j];
                    for (int k = 0; k < out1; k++) {
                        sum += h1[b * out1 + k] * W2[k * out2 + j];
                    }
                    h2[b * out2 + j] = sum;
                }
            }
            relu_inplace(h2, bs * out2, mask2);
            
            /* Forward: logits = h2 @ W3 + b3 */
            for (int b = 0; b < bs; b++) {
                for (int j = 0; j < out3; j++) {
                    float sum = b3[j];
                    for (int k = 0; k < out2; k++) {
                        sum += h2[b * out2 + k] * W3[k * out3 + j];
                    }
                    logits[b * out3 + j] = sum;
                }
            }
            
            /* Loss */
            int labels_int[64];
            for (int i = 0; i < bs; i++) labels_int[i] = (int)Y[i];
            float loss = cross_entropy_loss(logits, labels_int, bs, out3, dlogits);
            epoch_loss += loss;
            
            /* Accuracy */
            for (int b = 0; b < bs; b++) {
                int pred = 0;
                float max_val = logits[b * out3];
                for (int c = 1; c < out3; c++) {
                    if (logits[b * out3 + c] > max_val) {
                        max_val = logits[b * out3 + c];
                        pred = c;
                    }
                }
                if (pred == labels_int[b]) correct++;
                total++;
            }
            
            /* Backward: dW3 = h2^T @ dlogits, db3 = sum(dlogits), dh2 = dlogits @ W3^T */
            for (int k = 0; k < out2; k++) {
                for (int j = 0; j < out3; j++) {
                    float sum = 0;
                    for (int b = 0; b < bs; b++) {
                        sum += h2[b * out2 + k] * dlogits[b * out3 + j];
                    }
                    dW3[k * out3 + j] = sum;
                }
            }
            for (int j = 0; j < out3; j++) {
                float sum = 0;
                for (int b = 0; b < bs; b++) sum += dlogits[b * out3 + j];
                db3[j] = sum;
            }
            for (int b = 0; b < bs; b++) {
                for (int k = 0; k < out2; k++) {
                    float sum = 0;
                    for (int j = 0; j < out3; j++) {
                        sum += dlogits[b * out3 + j] * W3[k * out3 + j];
                    }
                    dh2[b * out2 + k] = sum * mask2[b * out2 + k];
                }
            }
            
            /* Backward: dW2, db2, dh1 */
            for (int k = 0; k < out1; k++) {
                for (int j = 0; j < out2; j++) {
                    float sum = 0;
                    for (int b = 0; b < bs; b++) {
                        sum += h1[b * out1 + k] * dh2[b * out2 + j];
                    }
                    dW2[k * out2 + j] = sum;
                }
            }
            for (int j = 0; j < out2; j++) {
                float sum = 0;
                for (int b = 0; b < bs; b++) sum += dh2[b * out2 + j];
                db2[j] = sum;
            }
            for (int b = 0; b < bs; b++) {
                for (int k = 0; k < out1; k++) {
                    float sum = 0;
                    for (int j = 0; j < out2; j++) {
                        sum += dh2[b * out2 + j] * W2[k * out2 + j];
                    }
                    dh1[b * out1 + k] = sum * mask1[b * out1 + k];
                }
            }
            
            /* Backward: dW1, db1 */
            for (int k = 0; k < in1; k++) {
                for (int j = 0; j < out1; j++) {
                    float sum = 0;
                    for (int b = 0; b < bs; b++) {
                        sum += X[b * in1 + k] * dh1[b * out1 + j];
                    }
                    dW1[k * out1 + j] = sum;
                }
            }
            for (int j = 0; j < out1; j++) {
                float sum = 0;
                for (int b = 0; b < bs; b++) sum += dh1[b * out1 + j];
                db1[j] = sum;
            }
            
            /* SGD update */
            for (int i = 0; i < in1 * out1; i++) W1[i] -= lr * dW1[i];
            for (int i = 0; i < out1; i++) b1[i] -= lr * db1[i];
            for (int i = 0; i < in2 * out2; i++) W2[i] -= lr * dW2[i];
            for (int i = 0; i < out2; i++) b2[i] -= lr * db2[i];
            for (int i = 0; i < in3 * out3; i++) W3[i] -= lr * dW3[i];
            for (int i = 0; i < out3; i++) b3[i] -= lr * db3[i];
        }
        
        printf("Epoch %d/%d | Loss: %.4f | Accuracy: %.2f%%\n",
               epoch + 1, epochs, epoch_loss / num_batches, 100.0f * correct / total);
    }
    
    printf("\nTraining time: %.2f sec\n", (double)(clock() - start) / CLOCKS_PER_SEC);
    
    /* Cleanup */
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
    free(dW1); free(db1); free(dW2); free(db2); free(dW3); free(db3);
    free(h1); free(mask1); free(h2); free(mask2);
    free(logits); free(dlogits); free(dh2); free(dh1);
    cg_mnist_free(mnist);
    
    printf("Done!\n");
    return 0;
}
