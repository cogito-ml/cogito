/**
 * Dataset Utilities for Cogito
 * 
 * Data loading and preprocessing utilities.
 */

#ifndef CG_DATASETS_H
#define CG_DATASETS_H

#include "cg_tensor.h"
#include <stdbool.h>

/*============================================================================
 * MNIST DATASET
 *============================================================================*/

/**
 * MNIST dataset structure.
 * 
 * Images: 28x28 grayscale, normalized to [0, 1]
 * Labels: 0-9 class indices
 */
typedef struct {
    cg_tensor* images;             /* [N, 784] flattened images */
    cg_tensor* labels;             /* [N] integer labels */
    int num_samples;
} cg_mnist;

/**
 * Load MNIST dataset from IDX files.
 * 
 * @param images_path Path to images file (e.g., "train-images.idx3-ubyte")
 * @param labels_path Path to labels file (e.g., "train-labels.idx1-ubyte")
 * @return MNIST dataset or NULL on failure
 */
cg_mnist* cg_mnist_load(const char* images_path, const char* labels_path);

/**
 * Free MNIST dataset.
 */
void cg_mnist_free(cg_mnist* mnist);

/**
 * Get a batch from MNIST dataset.
 * 
 * @param mnist The dataset
 * @param start_idx Starting index
 * @param batch_size Number of samples
 * @param images_out Output tensor for images [batch_size, 784]
 * @param labels_out Output tensor for labels [batch_size]
 */
void cg_mnist_get_batch(cg_mnist* mnist, int start_idx, int batch_size,
                        cg_tensor* images_out, cg_tensor* labels_out);

/**
 * Convert labels to one-hot encoding.
 * 
 * @param labels Label tensor [N]
 * @param num_classes Number of classes
 * @return One-hot tensor [N, num_classes]
 */
cg_tensor* cg_labels_to_onehot(cg_tensor* labels, int num_classes);

/*============================================================================
 * DATA AUGMENTATION
 *============================================================================*/

/**
 * Add Gaussian noise to tensor.
 */
void cg_augment_gaussian_noise(cg_tensor* t, float std, unsigned int seed);

/**
 * Random horizontal flip (for images).
 */
void cg_augment_horizontal_flip(cg_tensor* t, float prob, unsigned int seed);

/*============================================================================
 * DATA ITERATOR
 *============================================================================*/

/**
 * Simple data iterator for batching.
 */
typedef struct {
    cg_tensor* data;
    cg_tensor* labels;
    int batch_size;
    int current_idx;
    int num_samples;
    bool shuffle;
    int* indices;
} cg_data_iter;

/**
 * Create a data iterator.
 */
cg_data_iter* cg_data_iter_new(cg_tensor* data, cg_tensor* labels, 
                                int batch_size, bool shuffle, unsigned int seed);

/**
 * Get next batch from iterator.
 * Returns false if no more batches (end of epoch).
 */
bool cg_data_iter_next(cg_data_iter* iter, cg_tensor** batch_data, cg_tensor** batch_labels);

/**
 * Reset iterator for new epoch.
 */
void cg_data_iter_reset(cg_data_iter* iter, unsigned int seed);

/**
 * Free iterator.
 */
void cg_data_iter_free(cg_data_iter* iter);

/*============================================================================
 * DATALOADER
 *============================================================================*/

typedef struct cg_dataloader cg_dataloader;

/**
 * Create a new DataLoader for batched iteration.
 */
cg_dataloader* cg_dataloader_new(cg_tensor* data, cg_tensor* labels,
                                  int batch_size, bool shuffle, unsigned int seed);

/**
 * Get next batch from DataLoader.
 * Allocates new tensors for batch_data and batch_labels.
 * Returns false when epoch is complete.
 */
bool cg_dataloader_next(cg_dataloader* loader, cg_tensor** batch_data, 
                         cg_tensor** batch_labels, int* actual_batch_size);

/**
 * Reset DataLoader for new epoch.
 */
void cg_dataloader_reset(cg_dataloader* loader);

/**
 * Get number of batches in DataLoader.
 */
int cg_dataloader_num_batches(cg_dataloader* loader);

/**
 * Free DataLoader (does not free underlying data).
 */
void cg_dataloader_free(cg_dataloader* loader);

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

/**
 * One-hot encode labels.
 */
cg_tensor* cg_tensor_onehot(cg_tensor* labels, int num_classes);

/**
 * Get argmax for a sample in a batch.
 */
int cg_tensor_argmax(cg_tensor* t, int sample_idx, int num_classes);

/**
 * Compute classification accuracy.
 */
float cg_compute_accuracy(cg_tensor* logits, cg_tensor* labels, int num_classes);

#endif /* CG_DATASETS_H */
