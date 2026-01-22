/**
 * Cogito Profiler Implementation
 */

#include "cg_profiler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Stub for CUDA runtime if not available */
#ifndef __CUDACC__
typedef void* cudaEvent_t;
typedef int cudaError_t;
#define cudaSuccess 0
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventRecord(cudaEvent_t event, void* stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop);
cudaError_t cudaEventDestroy(cudaEvent_t event);
#endif

cg_profiler g_profiler = {0};

void cg_profiler_init(void) {
    if (g_profiler.capacity == 0) {
        g_profiler.capacity = 64;
        g_profiler.layers = (cg_layer_stat*)calloc(g_profiler.capacity, sizeof(cg_layer_stat));
    }
}

void cg_profiler_enable(bool enable) {
    g_profiler.enabled = enable;
}

static cg_layer_stat* get_or_create_layer(const char* name) {
    for (int i = 0; i < g_profiler.num_layers; i++) {
        if (strcmp(g_profiler.layers[i].name, name) == 0) {
            return &g_profiler.layers[i];
        }
    }
    
    if (g_profiler.num_layers >= g_profiler.capacity) {
        g_profiler.capacity *= 2;
        g_profiler.layers = (cg_layer_stat*)realloc(g_profiler.layers, 
                                                    g_profiler.capacity * sizeof(cg_layer_stat));
    }
    
    cg_layer_stat* layer = &g_profiler.layers[g_profiler.num_layers++];
    strncpy(layer->name, name, 63);
    cudaEventCreate((cudaEvent_t*)&layer->start_event);
    cudaEventCreate((cudaEvent_t*)&layer->stop_event);
    return layer;
}

void cg_profiler_begin_layer(const char* name) {
    if (!g_profiler.enabled) return;
    cg_layer_stat* layer = get_or_create_layer(name);
    cudaEventRecord((cudaEvent_t)layer->start_event, NULL);
}

void cg_profiler_end_layer(const char* name) {
    if (!g_profiler.enabled) return;
    cg_layer_stat* layer = get_or_create_layer(name);
    cudaEventRecord((cudaEvent_t)layer->stop_event, NULL);
    
    /* We sync occasionally or just record. For accurate timing, we need to wait or query later.
       For simplicity here, we assume sync at report time or step end. 
       Actually, accumulated time logic usually requires sync. 
       Let's record now and compute later, but structure here is simple. */
    cudaEventSynchronize((cudaEvent_t)layer->stop_event);
    float ms = 0;
    cudaEventElapsedTime(&ms, (cudaEvent_t)layer->start_event, (cudaEvent_t)layer->stop_event);
    layer->total_ms += ms;
    layer->count++;
}

void cg_profiler_report(void) {
    if (!g_profiler.enabled || g_profiler.num_layers == 0) return;
    
    printf("\n=== Profiler Report (Epoch %d) ===\n", g_profiler.current_epoch);
    printf("%-40s | %-10s | %-10s\n", "Layer Name", "Total (ms)", "Avg (ms)");
    printf("----------------------------------------------------------------\n");
    
    /* Simple bubble sort to find slowest */
    cg_layer_stat** sorted = (cg_layer_stat**)malloc(g_profiler.num_layers * sizeof(cg_layer_stat*));
    for (int i = 0; i < g_profiler.num_layers; i++) sorted[i] = &g_profiler.layers[i];
    
    for (int i = 0; i < g_profiler.num_layers - 1; i++) {
        for (int j = 0; j < g_profiler.num_layers - i - 1; j++) {
            if (sorted[j]->total_ms < sorted[j+1]->total_ms) {
                cg_layer_stat* temp = sorted[j];
                sorted[j] = sorted[j+1];
                sorted[j+1] = temp;
            }
        }
    }
    
    for (int i = 0; i < g_profiler.num_layers; i++) {
        float avg = sorted[i]->count > 0 ? sorted[i]->total_ms / sorted[i]->count : 0.0f;
        printf("%-40s | %10.2f | %10.3f\n", 
               sorted[i]->name, sorted[i]->total_ms, avg);
    }
    printf("====================================\n\n");
    free(sorted);
}

void cg_profiler_step_end(void) {
    /* Optional end of step logic */
}

void cg_profiler_reset(void) {
    for (int i = 0; i < g_profiler.num_layers; i++) {
        g_profiler.layers[i].total_ms = 0;
        g_profiler.layers[i].count = 0;
    }
}
