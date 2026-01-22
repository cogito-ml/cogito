/**
 * Model Serialization - Zero-copy binary format
 */

#ifndef CG_SERIALIZATION_H
#define CG_SERIALIZATION_H

#include "cg_tensor.h"
#include <stdio.h>

#define CG_MODEL_MAGIC 0x434F4731  /* "COG1" */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_tensors;
    uint64_t total_size;
} cg_model_header;

typedef struct {
    char name[64];
    int ndim;
    int shape[8];
    int dtype; /* 0=FP32, 1=BF16, 2=INT4 */
    uint64_t offset;
    uint64_t size_bytes;
} cg_tensor_entry;

/* API */
void cg_save_model(const char* filename, cg_tensor** tensors, const char** names, int count);
void cg_load_model(const char* filename, cg_tensor*** out_tensors, char*** out_names, int* out_count);

#endif /* CG_SERIALIZATION_H */
