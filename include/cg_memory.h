/**
 * Memory Planner - Binned allocator with CUDA graph capture
 */

#ifndef CG_MEMORY_H
#define CG_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*============================================================================
 * MEMORY BLOCK
 *============================================================================*/

typedef struct cg_memory_block {
    void* ptr;                  /* Device pointer */
    size_t size;                /* Allocated size */
    size_t requested_size;      /* Original requested size */
    int bin;                    /* Power-of-2 bin index */
    int ref_count;
    bool is_free;
    struct cg_memory_block* next;
    struct cg_memory_block* prev;
    uint64_t alloc_id;          /* For debugging */
} cg_memory_block;

/*============================================================================
 * SUB-BINS FOR COMMON ML SIZES
 *============================================================================*/

#define CG_SUBBIN_768   768     /* Common embedding sizes */
#define CG_SUBBIN_1024  1024    /* Common hidden sizes */
#define CG_SUBBIN_4096  4096    /* Common FFN sizes */
#define CG_NUM_SUBBINS  3

/* Sub-bin sizes array for lookup */
static const size_t CG_SUBBIN_SIZES[CG_NUM_SUBBINS] = {
    CG_SUBBIN_768,
    CG_SUBBIN_1024,
    CG_SUBBIN_4096
};

/*============================================================================
 * ARENA TYPES FOR SPECIALIZED ALLOCATION
 *============================================================================*/

typedef enum {
    CG_ARENA_DEFAULT = 0,       /* General purpose */
    CG_ARENA_PARAM,             /* Long-lived parameter buffers */
    CG_ARENA_ACTIVATION,        /* Short-lived activation buffers */
    CG_ARENA_GRADIENT           /* Frequently cleared gradient buffers */
} cg_arena_type;

/*============================================================================
 * BUDDY ALLOCATION FOR COALESCING
 *============================================================================*/

typedef struct cg_buddy_block {
    void* ptr;                  /* Base pointer */
    size_t size;                /* Block size */
    int order;                  /* log2(size) - MIN_ORDER */
    bool is_free;
    struct cg_buddy_block* buddy;   /* Pointer to buddy block */
    struct cg_buddy_block* next;    /* Next in free list */
    struct cg_buddy_block* prev;    /* Prev in free list */
} cg_buddy_block;

#define CG_BUDDY_MIN_ORDER  5   /* Minimum: 32 bytes */
#define CG_BUDDY_MAX_ORDER  30  /* Maximum: 1GB */
#define CG_BUDDY_NUM_ORDERS (CG_BUDDY_MAX_ORDER - CG_BUDDY_MIN_ORDER + 1)

typedef struct {
    cg_buddy_block* free_lists[CG_BUDDY_NUM_ORDERS];  /* Free blocks by order */
    void* base_ptr;             /* Base of managed region */
    size_t total_size;          /* Total managed size */
    size_t allocated;           /* Currently allocated */
} cg_buddy_allocator;

/*============================================================================
 * MEMORY POOL
 *============================================================================*/

#define CG_NUM_BINS 32          /* 2^5 to 2^36 (32B to 64GB) */
#define CG_MIN_BIN 5            /* Minimum allocation: 32 bytes */
#define CG_MAX_BIN 36           /* Maximum allocation: 64GB */

typedef struct {
    cg_memory_block* free_list[CG_NUM_BINS];  /* Free blocks by size */
    cg_memory_block* subbin_list[CG_NUM_SUBBINS];  /* Sub-bins for ML sizes */
    cg_memory_block* all_blocks;               /* All allocated blocks */
    
    /* Buddy allocator for coalescing */
    cg_buddy_allocator* buddy;
    
    /* Statistics */
    size_t total_allocated;
    size_t total_in_use;
    size_t peak_usage;
    size_t num_allocs;
    size_t num_frees;
    size_t cache_hits;
    size_t cache_misses;
    size_t subbin_hits;         /* Hits in sub-bins */
    size_t buddy_coalesces;     /* Number of buddy coalesces */
} cg_memory_pool;

/*============================================================================
 * MEMORY PLANNER
 *============================================================================*/

typedef struct {
    cg_memory_pool* device_pool;    /* GPU memory */
    cg_memory_pool* host_pool;      /* Pinned host memory */
    
    /* CUDA graph capture state */
    void* cuda_graph;               /* cudaGraph_t */
    void* cuda_graph_exec;          /* cudaGraphExec_t */
    void* stream;                   /* cudaStream_t */
    bool graph_captured;
    
    /* Defragmentation */
    float frag_threshold;           /* Trigger defrag at this % */
    bool defrag_enabled;
    
    /* Tensor lifetime tracking */
    int* tensor_first_use;          /* First op that uses tensor */
    int* tensor_last_use;           /* Last op that uses tensor */
    int num_tensors;
    
    /* Memory reuse planning */
    int** reuse_map;                /* Tensor i can reuse buffer from tensor j */
} cg_memory_planner;

/*============================================================================
 * ALLOCATION API
 *============================================================================*/

/**
 * Create memory pool.
 */
cg_memory_pool* cg_memory_pool_new(void);

/**
 * Allocate from pool (binned allocation).
 */
void* cg_pool_alloc(cg_memory_pool* pool, size_t size);

/**
 * Free to pool (returns to free list).
 */
void cg_pool_free(cg_memory_pool* pool, void* ptr);

/**
 * Trim unused memory.
 */
void cg_pool_trim(cg_memory_pool* pool);

void cg_memory_pool_free(cg_memory_pool* pool);

/*============================================================================
 * SUB-BIN ALLOCATION API
 *============================================================================*/

/**
 * Allocate from sub-bins for common ML sizes.
 * Returns NULL if size doesn't match a sub-bin.
 */
void* cg_pool_alloc_subbin(cg_memory_pool* pool, size_t size);

/**
 * Check if size matches a sub-bin.
 */
int cg_subbin_index(size_t size);

/*============================================================================
 * ARENA-AWARE ALLOCATION API
 *============================================================================*/

/**
 * Allocate from pool with arena type hint.
 * Uses different allocation strategies based on tensor lifetime patterns.
 */
void* cg_pool_alloc_arena(cg_memory_pool* pool, size_t size, cg_arena_type arena);

/**
 * Free with arena type hint for optimized handling.
 */
void cg_pool_free_arena(cg_memory_pool* pool, void* ptr, cg_arena_type arena);

/*============================================================================
 * BUDDY ALLOCATOR API
 *============================================================================*/

/**
 * Create buddy allocator with given region.
 */
cg_buddy_allocator* cg_buddy_new(void* base, size_t size);

/**
 * Allocate from buddy allocator.
 */
void* cg_buddy_alloc(cg_buddy_allocator* buddy, size_t size);

/**
 * Free to buddy allocator (with automatic coalescing).
 */
void cg_buddy_free(cg_buddy_allocator* buddy, void* ptr, size_t size);

/**
 * Free buddy allocator.
 */
void cg_buddy_destroy(cg_buddy_allocator* buddy);

/*============================================================================
 * SLIDING COMPACTION API
 *============================================================================*/

/**
 * Perform sliding compaction to eliminate fragmentation gaps.
 * Moves live allocations to create contiguous free space.
 */
void cg_memory_sliding_compact(cg_memory_planner* planner);

/**
 * Check if sliding compaction is needed (fragmentation > threshold).
 */
bool cg_memory_needs_compact(cg_memory_pool* pool, float threshold);

/*============================================================================
 * MEMORY PLANNER API
 *============================================================================*/

/**
 * Create memory planner.
 */
cg_memory_planner* cg_memory_planner_new(void);

/**
 * Analyze IR graph for memory reuse opportunities.
 */
void cg_memory_plan_analyze(cg_memory_planner* planner, void* ir_graph);

/**
 * Planned allocation (uses lifetime analysis).
 */
void* cg_planned_alloc(cg_memory_planner* planner, size_t size, int tensor_id);

/**
 * Planned free.
 */
void cg_planned_free(cg_memory_planner* planner, void* ptr, int tensor_id);

void cg_memory_planner_free(cg_memory_planner* planner);

/*============================================================================
 * CUDA GRAPH CAPTURE
 *============================================================================*/

/**
 * Begin capturing operations to CUDA graph.
 */
void cg_cuda_graph_capture_begin(cg_memory_planner* planner);

/**
 * End capture and create executable graph.
 */
void cg_cuda_graph_capture_end(cg_memory_planner* planner);

/**
 * Replay captured graph (zero-overhead kernel launch).
 */
void cg_cuda_graph_replay(cg_memory_planner* planner);

/**
 * Check if graph needs recompilation (e.g., shape changed).
 */
bool cg_cuda_graph_needs_update(cg_memory_planner* planner);

/*============================================================================
 * DEFRAGMENTATION
 *============================================================================*/

/**
 * Check fragmentation level.
 */
float cg_memory_fragmentation(cg_memory_pool* pool);

/**
 * Run compaction pass.
 */
void cg_memory_defrag(cg_memory_planner* planner);

/**
 * Conditionally defrag if above threshold.
 */
void cg_memory_maybe_defrag(cg_memory_planner* planner);

/*============================================================================
 * STATISTICS
 *============================================================================*/

typedef struct {
    size_t total_allocated;
    size_t total_in_use;
    size_t peak_usage;
    float fragmentation;
    size_t cache_hit_rate;
    int num_bins_used;
} cg_memory_stats;

cg_memory_stats cg_memory_get_stats(cg_memory_planner* planner);
void cg_memory_print_stats(cg_memory_planner* planner);

/*============================================================================
 * TENSOR LIFETIME ANALYSIS
 *============================================================================*/

/**
 * Analyze tensor lifetimes for memory reuse.
 */
void cg_analyze_tensor_lifetimes(cg_memory_planner* planner, void* ir_graph);

/**
 * Find tensors that can share memory.
 */
void cg_find_reuse_opportunities(cg_memory_planner* planner);

/**
 * Check if two tensors can share memory.
 */
bool cg_can_reuse_memory(cg_memory_planner* planner, int tensor_a, int tensor_b);

#endif /* CG_MEMORY_H */
