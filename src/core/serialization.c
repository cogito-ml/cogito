/**
 * Model Serialization Implementation
 */

#include "cg_serialization.h"
#include <stdlib.h>
#include <string.h>

void cg_save_model(const char* filename, cg_tensor** tensors, const char** names, int count) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    cg_model_header header;
    header.magic = CG_MODEL_MAGIC;
    header.version = 1;
    header.num_tensors = count;
    header.total_size = 0; /* Updated later */
    
    fwrite(&header, sizeof(header), 1, f);
    
    uint64_t current_offset = sizeof(header) + count * sizeof(cg_tensor_entry);
    
    /* Write directory */
    for (int i = 0; i < count; i++) {
        cg_tensor_entry entry;
        memset(&entry, 0, sizeof(entry));
        strncpy(entry.name, names[i], 63);
        entry.ndim = tensors[i]->ndim;
        memcpy(entry.shape, tensors[i]->shape, entry.ndim * sizeof(int));
        entry.dtype = 0; /* FP32 */
        entry.offset = current_offset;
        entry.size_bytes = tensors[i]->size * sizeof(float);
        
        fwrite(&entry, sizeof(entry), 1, f);
        current_offset += entry.size_bytes;
    }
    
    /* Write data */
    for (int i = 0; i < count; i++) {
        fwrite(tensors[i]->data, sizeof(float), tensors[i]->size, f);
    }
    
    fclose(f);
}

void cg_load_model(const char* filename, cg_tensor*** out_tensors, char*** out_names, int* out_count) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    
    cg_model_header header;
    fread(&header, sizeof(header), 1, f);
    
    if (header.magic != CG_MODEL_MAGIC) {
        fclose(f);
        return;
    }
    
    int count = header.num_tensors;
    *out_count = count;
    
    *out_tensors = (cg_tensor**)malloc(count * sizeof(cg_tensor*));
    *out_names = (char**)malloc(count * sizeof(char*));
    
    cg_tensor_entry* entries = (cg_tensor_entry*)malloc(count * sizeof(cg_tensor_entry));
    fread(entries, sizeof(cg_tensor_entry), count, f);
    
    for (int i = 0; i < count; i++) {
        (*out_names)[i] = strdup(entries[i].name);
        (*out_tensors)[i] = cg_tensor_new(entries[i].shape, entries[i].ndim, false);
        
        fseek(f, entries[i].offset, SEEK_SET);
        fread((*out_tensors)[i]->data, 1, entries[i].size_bytes, f);
    }
    
    free(entries);
    fclose(f);
}
