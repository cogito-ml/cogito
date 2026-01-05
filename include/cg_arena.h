/**
 * Arena Allocator for Cogito
 * 
 * Provides fast, bulk memory allocation with O(1) allocations and
 * O(1) mass deallocation. Perfect for training loops where you can
 * free all intermediate tensors at once after each batch.
 */

#ifndef CG_ARENA_H
#define CG_ARENA_H

#include <stddef.h>
#include <stdbool.h>

/* Default alignment for allocations */
#define CG_ARENA_DEFAULT_ALIGNMENT 16

/* Arena block structure - linked list of memory blocks */
typedef struct cg_arena_block {
    unsigned char* data;           /* Raw memory */
    size_t size;                   /* Total size of this block */
    size_t used;                   /* Bytes used in this block */
    struct cg_arena_block* next;   /* Next block in chain */
} cg_arena_block;

/* Arena allocator */
typedef struct cg_arena {
    cg_arena_block* first;         /* First block in chain */
    cg_arena_block* current;       /* Current block for allocations */
    size_t default_block_size;     /* Size for new blocks */
    size_t total_allocated;        /* Total bytes allocated */
    size_t total_used;             /* Total bytes used */
} cg_arena;

/**
 * Create a new arena allocator.
 * 
 * @param initial_size Initial block size in bytes (e.g., 1GB = 1e9)
 * @return New arena or NULL on failure
 */
cg_arena* cg_arena_new(size_t initial_size);

/**
 * Allocate memory from the arena.
 * 
 * @param arena The arena allocator
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement (0 for default)
 * @return Pointer to allocated memory or NULL on failure
 */
void* cg_arena_alloc(cg_arena* arena, size_t size, size_t alignment);

/**
 * Convenience macro for allocating aligned memory with default alignment.
 */
#define cg_arena_alloc_default(arena, size) \
    cg_arena_alloc(arena, size, CG_ARENA_DEFAULT_ALIGNMENT)

/**
 * Reset the arena, marking all memory as free.
 * Does not actually free memory back to the OS.
 * 
 * @param arena The arena allocator
 */
void cg_arena_clear(cg_arena* arena);

/**
 * Destroy the arena and free all memory.
 * 
 * @param arena The arena allocator
 */
void cg_arena_free(cg_arena* arena);

/**
 * Get arena statistics.
 */
size_t cg_arena_total_allocated(cg_arena* arena);
size_t cg_arena_total_used(cg_arena* arena);

#endif /* CG_ARENA_H */
