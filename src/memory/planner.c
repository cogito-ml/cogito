/**
 * Memory Planner - Binned allocation with lifetime analysis
 */

#include "cg_memory.h"
#include "cg_jit.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CG_MIN_BIN 5
#define CG_MAX_BIN 36

static int size_to_bin(size_t size) {
    if (size == 0) return CG_MIN_BIN;
    int bin = 0;
    size_t s = size - 1;
    while (s > 0) { s >>= 1; bin++; }
    if (bin < CG_MIN_BIN) bin = CG_MIN_BIN;
    if (bin > CG_MAX_BIN) bin = CG_MAX_BIN;
    return bin;
}

static size_t bin_to_size(int bin) { return (size_t)1 << bin; }

cg_memory_pool* cg_memory_pool_new(void) {
    return (cg_memory_pool*)calloc(1, sizeof(cg_memory_pool));
}

void* cg_pool_alloc(cg_memory_pool* pool, size_t size) {
    int bin = size_to_bin(size);
    size_t actual_size = bin_to_size(bin);
    
    if (pool->free_list[bin]) {
        cg_memory_block* block = pool->free_list[bin];
        pool->free_list[bin] = block->next;
        block->is_free = false;
        pool->total_in_use += actual_size;
        pool->cache_hits++;
        return block->ptr;
    }
    
    cg_memory_block* block = (cg_memory_block*)calloc(1, sizeof(cg_memory_block));
    block->ptr = malloc(actual_size);
    block->size = actual_size;
    block->bin = bin;
    block->next = pool->all_blocks;
    pool->all_blocks = block;
    pool->total_allocated += actual_size;
    pool->total_in_use += actual_size;
    pool->cache_misses++;
    pool->num_allocs++;
    if (pool->total_in_use > pool->peak_usage) pool->peak_usage = pool->total_in_use;
    return block->ptr;
}

void cg_pool_free(cg_memory_pool* pool, void* ptr) {
    if (!ptr) return;
    cg_memory_block* block = pool->all_blocks;
    while (block && block->ptr != ptr) block = block->next;
    if (!block) return;
    block->is_free = true;
    pool->total_in_use -= block->size;
    pool->num_frees++;
    block->next = pool->free_list[block->bin];
    pool->free_list[block->bin] = block;
}

void cg_pool_trim(cg_memory_pool* pool) {
    for (int bin = CG_MIN_BIN; bin <= CG_MAX_BIN; bin++) {
        cg_memory_block* block = pool->free_list[bin];
        while (block) {
            cg_memory_block* next = block->next;
            pool->total_allocated -= block->size;
            free(block->ptr);
            free(block);
            block = next;
        }
        pool->free_list[bin] = NULL;
    }
}

void cg_memory_pool_free(cg_memory_pool* pool) {
    if (!pool) return;
    cg_memory_block* block = pool->all_blocks;
    while (block) {
        cg_memory_block* next = block->next;
        free(block->ptr);
        free(block);
        block = next;
    }
    free(pool);
}

cg_memory_planner* cg_memory_planner_new(void) {
    cg_memory_planner* p = (cg_memory_planner*)calloc(1, sizeof(cg_memory_planner));
    p->device_pool = cg_memory_pool_new();
    p->host_pool = cg_memory_pool_new();
    p->frag_threshold = 0.25f;
    p->defrag_enabled = true;
    return p;
}

void cg_memory_planner_free(cg_memory_planner* p) {
    if (!p) return;
    cg_memory_pool_free(p->device_pool);
    cg_memory_pool_free(p->host_pool);
    free(p->tensor_first_use);
    free(p->tensor_last_use);
    free(p);
}

float cg_memory_fragmentation(cg_memory_pool* pool) {
    if (pool->total_allocated == 0) return 0.0f;
    size_t total_free = 0, largest = 0;
    for (int bin = CG_MIN_BIN; bin <= CG_MAX_BIN; bin++) {
        for (cg_memory_block* b = pool->free_list[bin]; b; b = b->next) {
            total_free += b->size;
            if (b->size > largest) largest = b->size;
        }
    }
    return total_free ? (1.0f - (float)largest / total_free) : 0.0f;
}

void cg_memory_defrag(cg_memory_planner* p) { cg_pool_trim(p->device_pool); }

void cg_memory_maybe_defrag(cg_memory_planner* p) {
    if (!p->defrag_enabled) return;
    float frag = cg_memory_fragmentation(p->device_pool);
    if (frag > p->frag_threshold) {
        printf("[Defrag] %.1f%% fragmentation\n", frag * 100);
        cg_memory_defrag(p);
    }
}

void* cg_planned_alloc(cg_memory_planner* p, size_t size, int tid) {
    return cg_pool_alloc(p->device_pool, size);
}

void cg_planned_free(cg_memory_planner* p, void* ptr, int tid) {
    cg_pool_free(p->device_pool, ptr);
}

cg_memory_stats cg_memory_get_stats(cg_memory_planner* p) {
    cg_memory_stats s = {0};
    s.total_allocated = p->device_pool->total_allocated;
    s.total_in_use = p->device_pool->total_in_use;
    s.peak_usage = p->device_pool->peak_usage;
    s.fragmentation = cg_memory_fragmentation(p->device_pool);
    return s;
}

void cg_memory_print_stats(cg_memory_planner* p) {
    cg_memory_stats s = cg_memory_get_stats(p);
    printf("Memory: %.2fMB alloc, %.2fMB use, %.2fMB peak, %.1f%% frag\n",
           s.total_allocated/1e6, s.total_in_use/1e6, s.peak_usage/1e6, s.fragmentation*100);
}

void cg_cuda_graph_capture_begin(cg_memory_planner* p) { p->graph_captured = false; }
void cg_cuda_graph_capture_end(cg_memory_planner* p) {}
void cg_cuda_graph_replay(cg_memory_planner* p) {}
bool cg_cuda_graph_needs_update(cg_memory_planner* p) { return !p->graph_captured; }
