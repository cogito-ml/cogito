/**
 * Arena Allocator Implementation
 * 
 * Bump-pointer allocator with block chaining for fast, bulk allocations.
 */

#include "cg_arena.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Align a pointer up to the given alignment */
static size_t align_up(size_t ptr, size_t alignment) {
    return (ptr + alignment - 1) & ~(alignment - 1);
}

/* Create a new block with the given size */
static cg_arena_block* cg_arena_block_new(size_t size) {
    cg_arena_block* block = (cg_arena_block*)malloc(sizeof(cg_arena_block));
    if (!block) return NULL;
    
    block->data = (unsigned char*)malloc(size);
    if (!block->data) {
        free(block);
        return NULL;
    }
    
    block->size = size;
    block->used = 0;
    block->next = NULL;
    
    return block;
}

/* Free a block and all following blocks */
static void cg_arena_block_free(cg_arena_block* block) {
    while (block) {
        cg_arena_block* next = block->next;
        free(block->data);
        free(block);
        block = next;
    }
}

cg_arena* cg_arena_new(size_t initial_size) {
    cg_arena* arena = (cg_arena*)malloc(sizeof(cg_arena));
    if (!arena) return NULL;
    
    /* Minimum block size of 4KB */
    if (initial_size < 4096) initial_size = 4096;
    
    arena->first = cg_arena_block_new(initial_size);
    if (!arena->first) {
        free(arena);
        return NULL;
    }
    
    arena->current = arena->first;
    arena->default_block_size = initial_size;
    arena->total_allocated = initial_size;
    arena->total_used = 0;
    
    return arena;
}

void* cg_arena_alloc(cg_arena* arena, size_t size, size_t alignment) {
    assert(arena != NULL);
    
    if (alignment == 0) alignment = CG_ARENA_DEFAULT_ALIGNMENT;
    
    /* Ensure alignment is a power of 2 */
    assert((alignment & (alignment - 1)) == 0);
    
    cg_arena_block* block = arena->current;
    
    /* Calculate aligned position */
    size_t aligned_used = align_up(block->used, alignment);
    
    /* Check if current block has enough space */
    if (aligned_used + size > block->size) {
        /* Need a new block */
        size_t new_block_size = arena->default_block_size;
        if (size > new_block_size) {
            /* Allocate larger block for big allocations */
            new_block_size = align_up(size, 4096);
        }
        
        cg_arena_block* new_block = cg_arena_block_new(new_block_size);
        if (!new_block) return NULL;
        
        /* Chain the new block */
        block->next = new_block;
        arena->current = new_block;
        arena->total_allocated += new_block_size;
        
        block = new_block;
        aligned_used = align_up(0, alignment);
    }
    
    void* ptr = block->data + aligned_used;
    block->used = aligned_used + size;
    arena->total_used += size;
    
    return ptr;
}

void cg_arena_clear(cg_arena* arena) {
    assert(arena != NULL);
    
    /* Reset all blocks' used counters */
    cg_arena_block* block = arena->first;
    while (block) {
        block->used = 0;
        block = block->next;
    }
    
    /* Reset to first block */
    arena->current = arena->first;
    arena->total_used = 0;
}

void cg_arena_free(cg_arena* arena) {
    if (!arena) return;
    
    cg_arena_block_free(arena->first);
    free(arena);
}

size_t cg_arena_total_allocated(cg_arena* arena) {
    return arena ? arena->total_allocated : 0;
}

size_t cg_arena_total_used(cg_arena* arena) {
    return arena ? arena->total_used : 0;
}
