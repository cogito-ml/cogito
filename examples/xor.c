/**
 * XOR Example - Basic neural network training with manual gradients
 * 
 * Network: 2 -> 8 -> 1 with Tanh and Sigmoid activations
 */

#include "cogito.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("Cogito XOR Example\n");
    printf("==================\n\n");
    
    srand(42);
    
    /* XOR dataset: 4 samples, 2 features each */
    float X_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float y_data[] = {0, 1, 1, 0};
    
    /* Initialize weights */
    /* Layer 1: (2, 8) weights + (8) bias */
    /* Layer 2: (8, 1) weights + (1) bias */
    float W1[2 * 8], b1[8], W2[8], b2[1];
    float dW1[2 * 8], db1[8], dW2[8], db2[1];
    
    /* Xavier init */
    for (int i = 0; i < 16; i++) {
        W1[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / 2.0f);
    }
    for (int i = 0; i < 8; i++) {
        b1[i] = 0.0f;
        W2[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / 8.0f);
    }
    b2[0] = 0.0f;
    
    float lr = 1.0f;
    int epochs = 5000;
    
    /* Intermediate buffers */
    float h1[4 * 8], a1[4 * 8], h2[4], pred[4];
    float dh2[4], da1[4 * 8], dh1[4 * 8];
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        /* Zero gradients */
        memset(dW1, 0, sizeof(dW1));
        memset(db1, 0, sizeof(db1));
        memset(dW2, 0, sizeof(dW2));
        memset(db2, 0, sizeof(db2));
        
        /* === FORWARD PASS === */
        
        /* h1 = X @ W1 + b1 : (4,2) @ (2,8) -> (4,8) */
        for (int b = 0; b < 4; b++) {
            for (int j = 0; j < 8; j++) {
                float sum = b1[j];
                for (int k = 0; k < 2; k++) {
                    sum += X_data[b * 2 + k] * W1[k * 8 + j];
                }
                h1[b * 8 + j] = sum;
            }
        }
        
        /* a1 = tanh(h1) */
        for (int i = 0; i < 32; i++) {
            a1[i] = tanhf(h1[i]);
        }
        
        /* h2 = a1 @ W2 + b2 : (4,8) @ (8,1) -> (4,1) */
        for (int b = 0; b < 4; b++) {
            float sum = b2[0];
            for (int k = 0; k < 8; k++) {
                sum += a1[b * 8 + k] * W2[k];
            }
            h2[b] = sum;
        }
        
        /* pred = sigmoid(h2) */
        for (int i = 0; i < 4; i++) {
            pred[i] = 1.0f / (1.0f + expf(-h2[i]));
        }
        
        /* === LOSS === */
        float loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            float diff = pred[i] - y_data[i];
            loss += diff * diff;
        }
        loss /= 4.0f;
        
        if (epoch % 500 == 0 || epoch == epochs - 1) {
            printf("Epoch %4d | Loss: %.6f\n", epoch, loss);
        }
        
        /* === BACKWARD PASS === */
        
        /* d_loss/d_pred = 2 * (pred - y) / n */
        float dpred[4];
        for (int i = 0; i < 4; i++) {
            dpred[i] = 2.0f * (pred[i] - y_data[i]) / 4.0f;
        }
        
        /* dh2 = dpred * sigmoid'(h2) = dpred * pred * (1 - pred) */
        for (int i = 0; i < 4; i++) {
            dh2[i] = dpred[i] * pred[i] * (1.0f - pred[i]);
        }
        
        /* db2 = sum(dh2) */
        for (int b = 0; b < 4; b++) {
            db2[0] += dh2[b];
        }
        
        /* dW2 = a1^T @ dh2 : (8,4) @ (4,1) -> (8,1) */
        for (int i = 0; i < 8; i++) {
            for (int b = 0; b < 4; b++) {
                dW2[i] += a1[b * 8 + i] * dh2[b];
            }
        }
        
        /* da1 = dh2 @ W2^T : (4,1) @ (1,8) -> (4,8) */
        for (int b = 0; b < 4; b++) {
            for (int j = 0; j < 8; j++) {
                da1[b * 8 + j] = dh2[b] * W2[j];
            }
        }
        
        /* dh1 = da1 * tanh'(h1) = da1 * (1 - a1^2) */
        for (int i = 0; i < 32; i++) {
            dh1[i] = da1[i] * (1.0f - a1[i] * a1[i]);
        }
        
        /* db1 = sum over batch */
        for (int b = 0; b < 4; b++) {
            for (int j = 0; j < 8; j++) {
                db1[j] += dh1[b * 8 + j];
            }
        }
        
        /* dW1 = X^T @ dh1 : (2,4) @ (4,8) -> (2,8) */
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 8; j++) {
                for (int b = 0; b < 4; b++) {
                    dW1[i * 8 + j] += X_data[b * 2 + i] * dh1[b * 8 + j];
                }
            }
        }
        
        /* === SGD UPDATE === */
        for (int i = 0; i < 16; i++) W1[i] -= lr * dW1[i];
        for (int i = 0; i < 8; i++) b1[i] -= lr * db1[i];
        for (int i = 0; i < 8; i++) W2[i] -= lr * dW2[i];
        b2[0] -= lr * db2[0];
    }
    
    /* Final forward pass for predictions */
    for (int b = 0; b < 4; b++) {
        for (int j = 0; j < 8; j++) {
            float sum = b1[j];
            for (int k = 0; k < 2; k++) {
                sum += X_data[b * 2 + k] * W1[k * 8 + j];
            }
            h1[b * 8 + j] = tanhf(sum);
        }
    }
    for (int b = 0; b < 4; b++) {
        float sum = b2[0];
        for (int k = 0; k < 8; k++) {
            sum += h1[b * 8 + k] * W2[k];
        }
        pred[b] = 1.0f / (1.0f + expf(-sum));
    }
    
    printf("\nFinal Predictions:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d, %d] -> %.4f (expected: %.0f)\n",
               (int)X_data[i * 2], (int)X_data[i * 2 + 1],
               pred[i], y_data[i]);
    }
    
    printf("\nDone!\n");
    return 0;
}
