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

/*============================================================================
 * SUB-BIN ALLOCATION
 *============================================================================*/

int cg_subbin_index(size_t size) {
    for (int i = 0; i < CG_NUM_SUBBINS; i++) {
        if (size == CG_SUBBIN_SIZES[i]) return i;
    }
    return -1;  /* Not a sub-bin size */
}

void* cg_pool_alloc_subbin(cg_memory_pool* pool, size_t size) {
    int idx = cg_subbin_index(size);
    if (idx < 0) return NULL;
    
    /* Check sub-bin free list first */
    if (pool->subbin_list[idx]) {
        cg_memory_block* block = pool->subbin_list[idx];
        pool->subbin_list[idx] = block->next;
        block->is_free = false;
        pool->total_in_use += size;
        pool->subbin_hits++;
        pool->cache_hits++;
        return block->ptr;
    }
    
    /* No cached block, allocate exact size */
    cg_memory_block* block = (cg_memory_block*)calloc(1, sizeof(cg_memory_block));
    block->ptr = malloc(size);
    block->size = size;
    block->requested_size = size;
    block->bin = -1;  /* Mark as sub-bin allocation */
    block->next = pool->all_blocks;
    pool->all_blocks = block;
    pool->total_allocated += size;
    pool->total_in_use += size;
    pool->cache_misses++;
    pool->num_allocs++;
    if (pool->total_in_use > pool->peak_usage) pool->peak_usage = pool->total_in_use;
    return block->ptr;
}

cg_memory_pool* cg_memory_pool_new(void) {
    cg_memory_pool* pool = (cg_memory_pool*)calloc(1, sizeof(cg_memory_pool));
    /* Sub-bin lists are already zeroed by calloc */
    return pool;
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

/*============================================================================
 * BUDDY ALLOCATOR
 *============================================================================*/

static int size_to_order(size_t size) {
    int order = 0;
    size_t s = size - 1;
    while (s > 0) { s >>= 1; order++; }
    return order - CG_BUDDY_MIN_ORDER;
}

static size_t order_to_size(int order) {
    return (size_t)1 << (order + CG_BUDDY_MIN_ORDER);
}

cg_buddy_allocator* cg_buddy_new(void* base, size_t size) {
    cg_buddy_allocator* buddy = (cg_buddy_allocator*)calloc(1, sizeof(cg_buddy_allocator));
    buddy->base_ptr = base;
    buddy->total_size = size;
    buddy->allocated = 0;
    
    /* Create initial free block covering entire region */
    int order = size_to_order(size);
    if (order >= 0 && order < CG_BUDDY_NUM_ORDERS) {
        cg_buddy_block* block = (cg_buddy_block*)calloc(1, sizeof(cg_buddy_block));
        block->ptr = base;
        block->size = size;
        block->order = order;
        block->is_free = true;
        buddy->free_lists[order] = block;
    }
    return buddy;
}

void* cg_buddy_alloc(cg_buddy_allocator* buddy, size_t size) {
    if (!buddy || size == 0) return NULL;
    
    int order = size_to_order(size);
    if (order < 0) order = 0;
    if (order >= CG_BUDDY_NUM_ORDERS) return NULL;
    
    /* Find smallest available block */
    int found_order = -1;
    for (int o = order; o < CG_BUDDY_NUM_ORDERS; o++) {
        if (buddy->free_lists[o]) {
            found_order = o;
            break;
        }
    }
    
    if (found_order < 0) return NULL;  /* No block available */
    
    /* Split larger blocks if necessary */
    while (found_order > order) {
        cg_buddy_block* block = buddy->free_lists[found_order];
        buddy->free_lists[found_order] = block->next;
        
        /* Split into two buddies */
        size_t half_size = block->size / 2;
        int new_order = found_order - 1;
        
        cg_buddy_block* left = block;
        left->size = half_size;
        left->order = new_order;
        
        cg_buddy_block* right = (cg_buddy_block*)calloc(1, sizeof(cg_buddy_block));
        right->ptr = (char*)block->ptr + half_size;
        right->size = half_size;
        right->order = new_order;
        right->is_free = true;
        right->buddy = left;
        left->buddy = right;
        
        /* Add both to lower-order free list */
        right->next = buddy->free_lists[new_order];
        left->next = right;
        buddy->free_lists[new_order] = left;
        
        found_order = new_order;
    }
    
    /* Allocate from free list */
    cg_buddy_block* block = buddy->free_lists[order];
    buddy->free_lists[order] = block->next;
    block->is_free = false;
    block->next = NULL;
    buddy->allocated += block->size;
    
    return block->ptr;
}

void cg_buddy_free(cg_buddy_allocator* buddy, void* ptr, size_t size) {
    if (!buddy || !ptr) return;
    
    int order = size_to_order(size);
    if (order < 0) order = 0;
    
    /* Create block for freed memory */
    cg_buddy_block* block = (cg_buddy_block*)calloc(1, sizeof(cg_buddy_block));
    block->ptr = ptr;
    block->size = size;
    block->order = order;
    block->is_free = true;
    
    buddy->allocated -= size;
    
    /* Try to coalesce with buddy */
    while (order < CG_BUDDY_NUM_ORDERS - 1) {
        /* Calculate buddy address */
        size_t block_offset = (char*)block->ptr - (char*)buddy->base_ptr;
        size_t buddy_offset = block_offset ^ order_to_size(order);
        void* buddy_ptr = (char*)buddy->base_ptr + buddy_offset;
        
        /* Search for buddy in free list */
        cg_buddy_block* prev = NULL;
        cg_buddy_block* curr = buddy->free_lists[order];
        cg_buddy_block* found_buddy = NULL;
        
        while (curr) {
            if (curr->ptr == buddy_ptr && curr->is_free) {
                found_buddy = curr;
                break;
            }
            prev = curr;
            curr = curr->next;
        }
        
        if (!found_buddy) break;  /* No buddy to coalesce */
        
        /* Remove buddy from free list */
        if (prev) prev->next = found_buddy->next;
        else buddy->free_lists[order] = found_buddy->next;
        
        /* Merge blocks */
        void* merged_ptr = (block_offset < buddy_offset) ? block->ptr : found_buddy->ptr;
        size_t merged_size = block->size * 2;
        
        free(found_buddy);
        block->ptr = merged_ptr;
        block->size = merged_size;
        block->order = order + 1;
        
        buddy->buddy_coalesces++;
        order++;
    }
    
    /* Add to free list */
    block->next = buddy->free_lists[order];
    buddy->free_lists[order] = block;
}

void cg_buddy_destroy(cg_buddy_allocator* buddy) {
    if (!buddy) return;
    for (int o = 0; o < CG_BUDDY_NUM_ORDERS; o++) {
        cg_buddy_block* block = buddy->free_lists[o];
        while (block) {
            cg_buddy_block* next = block->next;
            free(block);
            block = next;
        }
    }
    free(buddy);
}

/*============================================================================
 * ARENA-AWARE ALLOCATION
 *============================================================================*/

void* cg_pool_alloc_arena(cg_memory_pool* pool, size_t size, cg_arena_type arena) {
    /* Try sub-bin first for exact ML sizes */
    void* ptr = cg_pool_alloc_subbin(pool, size);
    if (ptr) return ptr;
    
    /* Arena hints for future optimization:
     * - PARAM: Prefer buddy allocator for stable long-term allocations
     * - ACTIVATION: Use standard bins, aggressively recycle
     * - GRADIENT: Similar to activation but might be cleared frequently
     */
    (void)arena;  /* Currently unused, reserved for future optimization */
    
    return cg_pool_alloc(pool, size);
}

void cg_pool_free_arena(cg_memory_pool* pool, void* ptr, cg_arena_type arena) {
    (void)arena;  /* Currently unused */
    cg_pool_free(pool, ptr);
}

/*============================================================================
 * SLIDING COMPACTION
 *============================================================================*/

bool cg_memory_needs_compact(cg_memory_pool* pool, float threshold) {
    return cg_memory_fragmentation(pool) > threshold;
}

void cg_memory_sliding_compact(cg_memory_planner* planner) {
    cg_memory_pool* pool = planner->device_pool;
    
    /* Collect all live blocks */
    int num_live = 0;
    for (cg_memory_block* b = pool->all_blocks; b; b = b->next) {
        if (!b->is_free) num_live++;
    }
    
    if (num_live == 0) {
        /* No live blocks, just trim */
        cg_pool_trim(pool);
        return;
    }
    
    /* Sort live blocks by address (simple bubble sort for now) */
    cg_memory_block** live = (cg_memory_block**)malloc(num_live * sizeof(cg_memory_block*));
    int idx = 0;
    for (cg_memory_block* b = pool->all_blocks; b; b = b->next) {
        if (!b->is_free) live[idx++] = b;
    }
    
    for (int i = 0; i < num_live - 1; i++) {
        for (int j = 0; j < num_live - i - 1; j++) {
            if (live[j]->ptr > live[j+1]->ptr) {
                cg_memory_block* tmp = live[j];
                live[j] = live[j+1];
                live[j+1] = tmp;
            }
        }
    }
    
    /* Slide blocks toward the beginning */
    void* compact_ptr = live[0]->ptr;  /* Start from first live block */
    for (int i = 0; i < num_live; i++) {
        if (live[i]->ptr != compact_ptr) {
            /* Move block data */
            memmove(compact_ptr, live[i]->ptr, live[i]->size);
            live[i]->ptr = compact_ptr;
        }
        compact_ptr = (char*)compact_ptr + live[i]->size;
    }
    
    /* Clear all free lists and trim */
    cg_pool_trim(pool);
    
    free(live);
    printf("[Compact] Slid %d live blocks\n", num_live);
}

