/**
 * MNIST Dataset Loader
 * 
 * Parses IDX file format for MNIST handwritten digits.
 */

#include "cg_datasets.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* Read big-endian 32-bit integer */
static int read_int32_be(FILE* f) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, f) != 4) return -1;
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

cg_mnist* cg_mnist_load(const char* images_path, const char* labels_path) {
    FILE* img_file = fopen(images_path, "rb");
    FILE* lbl_file = fopen(labels_path, "rb");
    
    if (!img_file || !lbl_file) {
        if (img_file) fclose(img_file);
        if (lbl_file) fclose(lbl_file);
        return NULL;
    }
    
    /* Read image file header */
    int img_magic = read_int32_be(img_file);
    int num_images = read_int32_be(img_file);
    int rows = read_int32_be(img_file);
    int cols = read_int32_be(img_file);
    
    if (img_magic != 2051) { fclose(img_file); fclose(lbl_file); return NULL; }
    
    /* Read label file header */
    int lbl_magic = read_int32_be(lbl_file);
    int num_labels = read_int32_be(lbl_file);
    
    if (lbl_magic != 2049 || num_labels != num_images) {
        fclose(img_file); fclose(lbl_file); return NULL;
    }
    
    cg_mnist* mnist = (cg_mnist*)malloc(sizeof(cg_mnist));
    mnist->num_samples = num_images;
    
    /* Create image tensor: [N, 784] */
    int img_shape[] = {num_images, rows * cols};
    mnist->images = cg_tensor_new(img_shape, 2, false);
    
    /* Create label tensor: [N] */
    int lbl_shape[] = {num_images};
    mnist->labels = cg_tensor_new(lbl_shape, 1, false);
    
    /* Read and normalize images */
    unsigned char* pixel_buf = (unsigned char*)malloc(rows * cols);
    for (int i = 0; i < num_images; i++) {
        if (fread(pixel_buf, 1, rows * cols, img_file) != (size_t)(rows * cols)) {
            free(pixel_buf); cg_mnist_free(mnist);
            fclose(img_file); fclose(lbl_file); return NULL;
        }
        for (int j = 0; j < rows * cols; j++) {
            mnist->images->data[i * (rows * cols) + j] = pixel_buf[j] / 255.0f;
        }
    }
    free(pixel_buf);
    
    /* Read labels */
    unsigned char* label_buf = (unsigned char*)malloc(num_images);
    if (fread(label_buf, 1, num_images, lbl_file) != (size_t)num_images) {
        free(label_buf); cg_mnist_free(mnist);
        fclose(img_file); fclose(lbl_file); return NULL;
    }
    for (int i = 0; i < num_images; i++) {
        mnist->labels->data[i] = (float)label_buf[i];
    }
    free(label_buf);
    
    fclose(img_file);
    fclose(lbl_file);
    
    return mnist;
}

void cg_mnist_free(cg_mnist* mnist) {
    if (!mnist) return;
    if (mnist->images) cg_tensor_free(mnist->images);
    if (mnist->labels) cg_tensor_free(mnist->labels);
    free(mnist);
}

void cg_mnist_get_batch(cg_mnist* mnist, int start_idx, int batch_size,
                        cg_tensor* images_out, cg_tensor* labels_out) {
    assert(mnist && images_out && labels_out);
    assert(start_idx + batch_size <= mnist->num_samples);
    
    int img_size = mnist->images->shape[1];
    
    for (int i = 0; i < batch_size; i++) {
        int src_idx = start_idx + i;
        memcpy(images_out->data + i * img_size,
               mnist->images->data + src_idx * img_size,
               img_size * sizeof(float));
        labels_out->data[i] = mnist->labels->data[src_idx];
    }
}

cg_tensor* cg_labels_to_onehot(cg_tensor* labels, int num_classes) {
    assert(labels && labels->ndim == 1);
    
    int n = labels->size;
    int shape[] = {n, num_classes};
    cg_tensor* onehot = cg_tensor_zeros(shape, 2, false);
    
    for (int i = 0; i < n; i++) {
        int label = (int)labels->data[i];
        assert(label >= 0 && label < num_classes);
        onehot->data[i * num_classes + label] = 1.0f;
    }
    
    return onehot;
}

/* Data augmentation */
void cg_augment_gaussian_noise(cg_tensor* t, float std, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < t->size; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        float mag = sqrtf(-2.0f * logf(u1));
        t->data[i] += mag * cosf(2.0f * 3.14159f * u2) * std;
        if (i + 1 < t->size) {
            t->data[i + 1] += mag * sinf(2.0f * 3.14159f * u2) * std;
        }
    }
}

void cg_augment_horizontal_flip(cg_tensor* t, float prob, unsigned int seed) {
    srand(seed);
    if ((float)rand() / RAND_MAX < prob) {
        /* Assumes 2D image in flattened form - would need width info */
        /* Simplified: just reverse the array */
        for (int i = 0; i < t->size / 2; i++) {
            float tmp = t->data[i];
            t->data[i] = t->data[t->size - 1 - i];
            t->data[t->size - 1 - i] = tmp;
        }
    }
}

/* Data iterator */
static void shuffle_indices(int* indices, int n, unsigned int seed) {
    srand(seed);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

cg_data_iter* cg_data_iter_new(cg_tensor* data, cg_tensor* labels,
                                int batch_size, bool shuffle, unsigned int seed) {
    cg_data_iter* iter = (cg_data_iter*)malloc(sizeof(cg_data_iter));
    iter->data = data;
    iter->labels = labels;
    iter->batch_size = batch_size;
    iter->num_samples = data->shape[0];
    iter->shuffle = shuffle;
    iter->current_idx = 0;
    
    iter->indices = (int*)malloc(iter->num_samples * sizeof(int));
    for (int i = 0; i < iter->num_samples; i++) iter->indices[i] = i;
    
    if (shuffle) shuffle_indices(iter->indices, iter->num_samples, seed);
    
    return iter;
}

bool cg_data_iter_next(cg_data_iter* iter, cg_tensor** batch_data, cg_tensor** batch_labels) {
    if (iter->current_idx >= iter->num_samples) return false;
    
    int actual_batch = iter->batch_size;
    if (iter->current_idx + actual_batch > iter->num_samples) {
        actual_batch = iter->num_samples - iter->current_idx;
    }
    
    int feat_dim = iter->data->size / iter->num_samples;
    int batch_shape[] = {actual_batch, feat_dim};
    int label_shape[] = {actual_batch};
    
    *batch_data = cg_tensor_new(batch_shape, 2, false);
    *batch_labels = cg_tensor_new(label_shape, 1, false);
    
    for (int i = 0; i < actual_batch; i++) {
        int idx = iter->indices[iter->current_idx + i];
        memcpy((*batch_data)->data + i * feat_dim,
               iter->data->data + idx * feat_dim,
               feat_dim * sizeof(float));
        (*batch_labels)->data[i] = iter->labels->data[idx];
    }
    
    iter->current_idx += actual_batch;
    return true;
}

void cg_data_iter_reset(cg_data_iter* iter, unsigned int seed) {
    iter->current_idx = 0;
    if (iter->shuffle) shuffle_indices(iter->indices, iter->num_samples, seed);
}

void cg_data_iter_free(cg_data_iter* iter) {
    if (iter) { free(iter->indices); free(iter); }
}
