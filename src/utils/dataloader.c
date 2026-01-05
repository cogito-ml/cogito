/**
 * DataLoader - Efficient batched data iteration with shuffling
 */

#include "cg_datasets.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*============================================================================
 * DATALOADER IMPLEMENTATION
 *============================================================================*/

struct cg_dataloader {
    cg_tensor* data;       /* Full dataset: (N, features...) */
    cg_tensor* labels;     /* Labels: (N,) or (N, num_classes) */
    int* indices;          /* Shuffled indices */
    int num_samples;
    int batch_size;
    int current_idx;
    bool shuffle;
    unsigned int seed;
};

static void shuffle_array(int* arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

cg_dataloader* cg_dataloader_new(cg_tensor* data, cg_tensor* labels,
                                  int batch_size, bool shuffle, unsigned int seed) {
    if (!data) return NULL;
    
    cg_dataloader* loader = (cg_dataloader*)calloc(1, sizeof(cg_dataloader));
    loader->data = data;
    loader->labels = labels;
    loader->num_samples = data->shape[0];
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->seed = seed;
    loader->current_idx = 0;
    
    /* Create indices array */
    loader->indices = (int*)malloc(loader->num_samples * sizeof(int));
    for (int i = 0; i < loader->num_samples; i++) {
        loader->indices[i] = i;
    }
    
    if (shuffle) {
        shuffle_array(loader->indices, loader->num_samples, seed);
    }
    
    return loader;
}

bool cg_dataloader_next(cg_dataloader* loader, cg_tensor** batch_data, 
                         cg_tensor** batch_labels, int* actual_batch_size) {
    if (!loader || loader->current_idx >= loader->num_samples) {
        return false;
    }
    
    /* Compute actual batch size */
    int remaining = loader->num_samples - loader->current_idx;
    int bs = (remaining < loader->batch_size) ? remaining : loader->batch_size;
    *actual_batch_size = bs;
    
    /* Compute feature size per sample */
    int feat_size = loader->data->size / loader->num_samples;
    
    /* Create batch tensors */
    int batch_shape[CG_MAX_DIMS];
    batch_shape[0] = bs;
    for (int i = 1; i < loader->data->ndim; i++) {
        batch_shape[i] = loader->data->shape[i];
    }
    
    *batch_data = cg_tensor_new(batch_shape, loader->data->ndim, true);
    
    if (loader->labels) {
        int label_shape[] = {bs};
        *batch_labels = cg_tensor_new(label_shape, 1, false);
    } else {
        *batch_labels = NULL;
    }
    
    /* Copy data for this batch */
    for (int i = 0; i < bs; i++) {
        int idx = loader->indices[loader->current_idx + i];
        
        /* Copy features */
        memcpy((*batch_data)->data + i * feat_size,
               loader->data->data + idx * feat_size,
               feat_size * sizeof(float));
        
        /* Copy label */
        if (loader->labels && *batch_labels) {
            (*batch_labels)->data[i] = loader->labels->data[idx];
        }
    }
    
    loader->current_idx += bs;
    return true;
}

void cg_dataloader_reset(cg_dataloader* loader) {
    if (!loader) return;
    loader->current_idx = 0;
    
    if (loader->shuffle) {
        loader->seed++;
        shuffle_array(loader->indices, loader->num_samples, loader->seed);
    }
}

int cg_dataloader_num_batches(cg_dataloader* loader) {
    if (!loader || loader->batch_size == 0) return 0;
    return (loader->num_samples + loader->batch_size - 1) / loader->batch_size;
}

void cg_dataloader_free(cg_dataloader* loader) {
    if (!loader) return;
    free(loader->indices);
    free(loader);
}

/*============================================================================
 * UTILITY: ONE-HOT ENCODING
 *============================================================================*/

cg_tensor* cg_tensor_onehot(cg_tensor* labels, int num_classes) {
    if (!labels || labels->ndim != 1) return NULL;
    
    int n = labels->shape[0];
    int shape[] = {n, num_classes};
    cg_tensor* onehot = cg_tensor_zeros(shape, 2, false);
    
    for (int i = 0; i < n; i++) {
        int label = (int)labels->data[i];
        if (label >= 0 && label < num_classes) {
            onehot->data[i * num_classes + label] = 1.0f;
        }
    }
    
    return onehot;
}

/*============================================================================
 * UTILITY: ARGMAX
 *============================================================================*/

int cg_tensor_argmax(cg_tensor* t, int sample_idx, int num_classes) {
    int offset = sample_idx * num_classes;
    int max_idx = 0;
    float max_val = t->data[offset];
    
    for (int i = 1; i < num_classes; i++) {
        if (t->data[offset + i] > max_val) {
            max_val = t->data[offset + i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/**
 * Compute classification accuracy.
 */
float cg_compute_accuracy(cg_tensor* logits, cg_tensor* labels, int num_classes) {
    int batch_size = logits->shape[0];
    int correct = 0;
    
    for (int b = 0; b < batch_size; b++) {
        int pred = cg_tensor_argmax(logits, b, num_classes);
        int target = (int)labels->data[b];
        if (pred == target) correct++;
    }
    
    return (float)correct / (float)batch_size;
}
