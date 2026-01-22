/**
 * Cogito Built-in Profiler
 * 
 * Tracks per-layer latency using CUDA events and reports bottlenecks.
 */

#ifndef CG_PROFILER_H
#define CG_PROFILER_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char name[64];
    void* start_event;  /* cudaEvent_t */
    void* stop_event;   /* cudaEvent_t */
    float total_ms;
    int count;
} cg_layer_stat;

typedef struct {
    cg_layer_stat* layers;
    int num_layers;
    int capacity;
    bool enabled;
    int current_epoch;
} cg_profiler;

/* Global profiler instance */
extern cg_profiler g_profiler;

void cg_profiler_init(void);
void cg_profiler_enable(bool enable);
void cg_profiler_begin_layer(const char* name);
void cg_profiler_end_layer(const char* name);
void cg_profiler_step_end(void);
void cg_profiler_report(void);
void cg_profiler_reset(void);

#endif /* CG_PROFILER_H */
