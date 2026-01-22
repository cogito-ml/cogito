/**
 * Extended Memory Subsystem Tests
 * 
 * Verifies:
 * 1. Sub-bin allocation (exact sizes for 768, 1024, 4096)
 * 2. Buddy allocator coalescing
 * 3. Sliding compaction defragmentation
 */

#include "cg_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); exit(1); } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

/*============================================================================
 * SUB-BIN TESTS
 *============================================================================*/

void test_subbin_allocation(void) {
    printf("\n=== Sub-bin Allocation Tests ===\n");
    
    cg_memory_pool* pool = cg_memory_pool_new();
    
    /* Test size 768 (should hit sub-bin 0) */
    void* p1 = cg_pool_alloc_subbin(pool, 768);
    ASSERT(p1 != NULL, "Alloc 768 succeeds");
    ASSERT(pool->total_in_use == 768, "Pool usage 768");
    
    /* Free and re-alloc should reuse */
    cg_pool_free(pool, p1); // Returns to subbin list
    ASSERT(pool->total_in_use == 0, "Pool usage 0 after free");
    
    void* p2 = cg_pool_alloc_subbin(pool, 768);
    ASSERT(p2 == p1, "Reuses same pointer from sub-bin cache");
    ASSERT(pool->subbin_hits == 1, "Sub-bin hit recorded");
    
    /* Allocation of non-subbin size */
    void* p3 = cg_pool_alloc_subbin(pool, 500);
    ASSERT(p3 == NULL, "Alloc 500 returns NULL (not a sub-bin size)");
    
    cg_memory_pool_free(pool);
}

/*============================================================================
 * BUDDY ALLOCATOR TESTS
 *============================================================================*/

void test_buddy_allocator(void) {
    printf("\n=== Buddy Allocator Tests ===\n");
    
    size_t total_size = 1024 * 1024; /* 1MB */
    void* base = malloc(total_size);
    cg_buddy_allocator* buddy = cg_buddy_new(base, total_size);
    
    /* Alloc 256KB */
    void* p1 = cg_buddy_alloc(buddy, 256 * 1024);
    ASSERT(p1 == base, "First alloc at base");
    ASSERT(buddy->allocated == 256 * 1024, "Allocated 256KB");
    
    /* Alloc 256KB (should be buddy of p1) */
    void* p2 = cg_buddy_alloc(buddy, 256 * 1024);
    ASSERT(p2 == (char*)base + 256*1024, "Second alloc adjacent");
    
    /* Free p1, p2 still used */
    cg_buddy_free(buddy, p1, 256 * 1024);
    ASSERT(buddy->allocated == 256 * 1024, "Allocated 256KB after free");
    
    /* Free p2, should coalesce with p1 into 512KB block */
    cg_buddy_free(buddy, p2, 256 * 1024);
    ASSERT(buddy->allocated == 0, "Allocated 0 after free all");
    ASSERT(buddy->buddy_coalesces >= 1, "Coalescing occurred");
    
    /* Alloc 512KB (should take the coalesced block) */
    void* p3 = cg_buddy_alloc(buddy, 512 * 1024);
    ASSERT(p3 == base, "Alloc 512KB takes coalesced block");
    
    cg_buddy_destroy(buddy);
    free(base);
}

/*============================================================================
 * COMPACTION TESTS
 *============================================================================*/

void test_sliding_compaction(void) {
    printf("\n=== Sliding Compaction Tests ===\n");
    
    cg_memory_planner* planner = cg_memory_planner_new();
    
    /* Create fragmentation: Alloc A, B, C, then free B */
    void* a = cg_planned_alloc(planner, 1024, 0);
    void* b = cg_planned_alloc(planner, 1024, 1);
    void* c = cg_planned_alloc(planner, 1024, 2);
    
    cg_planned_free(planner, b, 1);
    
    ASSERT(cg_memory_fragmentation(planner->device_pool) > 0.0f, "Fragmentation exists");
    
    /* Run compaction */
    cg_memory_sliding_compact(planner);
    
    ASSERT(cg_memory_fragmentation(planner->device_pool) == 0.0f, "Fragmentation eliminated");
    
    /* Verify C was moved (pointer changed internally, but we can't easily check void* without deeper introspection) */
    /* In a real scenario we'd update tensor->data pointers. Here we assume memmove worked. */
    
    cg_memory_planner_free(planner);
}

int main(void) {
    test_subbin_allocation();
    test_buddy_allocator();
    test_sliding_compaction();
    printf("\nAll memory tests passed!\n");
    return 0;
}
