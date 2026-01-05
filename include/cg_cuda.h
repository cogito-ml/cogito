/**
 * CUDA Device Management for Cogito
 * 
 * Provides GPU device management, memory operations, and CUDA utilities.
 */

#ifndef CG_CUDA_H
#define CG_CUDA_H

#include <stdbool.h>
#include <stddef.h>

#include "cg_tensor.h"

/*============================================================================
 * DEVICE TYPES
 *============================================================================*/

typedef enum {
    CG_DEVICE_CPU = 0,
    CG_DEVICE_CUDA = 1
} cg_device_type;

/*============================================================================
 * CUDA CONTEXT
 *============================================================================*/

typedef struct {
    int device_id;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    size_t free_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
    size_t shared_memory_per_block;
    char device_name[256];
    bool is_available;
} cg_cuda_device_info;

typedef struct {
    void* stream;                   /* cudaStream_t */
    void* cublas_handle;            /* cublasHandle_t */
    int device_id;
    bool initialized;
} cg_cuda_context;

/*============================================================================
 * GLOBAL CUDA STATE
 *============================================================================*/

/**
 * Initialize CUDA subsystem.
 * Returns true if CUDA is available and initialized.
 */
bool cg_cuda_init(void);

/**
 * Shutdown CUDA subsystem and free resources.
 */
void cg_cuda_shutdown(void);

/**
 * Check if CUDA is available and initialized.
 */
bool cg_cuda_is_available(void);

/**
 * Get number of CUDA devices.
 */
int cg_cuda_device_count(void);

/**
 * Get information about a specific device.
 */
cg_cuda_device_info cg_cuda_get_device_info(int device_id);

/**
 * Set current CUDA device.
 */
void cg_cuda_set_device(int device_id);

/**
 * Get current CUDA device.
 */
int cg_cuda_get_device(void);

/**
 * Get global CUDA context.
 */
cg_cuda_context* cg_cuda_get_context(void);

/*============================================================================
 * MEMORY OPERATIONS
 *============================================================================*/

/**
 * Allocate GPU memory.
 */
void* cg_cuda_malloc(size_t size);

/**
 * Free GPU memory.
 */
void cg_cuda_free(void* ptr);

/**
 * Allocate pinned (page-locked) host memory.
 */
void* cg_cuda_malloc_host(size_t size);

/**
 * Free pinned host memory.
 */
void cg_cuda_free_host(void* ptr);

/**
 * Copy memory between host and device.
 */
typedef enum {
    CG_MEMCPY_HOST_TO_DEVICE,
    CG_MEMCPY_DEVICE_TO_HOST,
    CG_MEMCPY_DEVICE_TO_DEVICE
} cg_memcpy_kind;

void cg_cuda_memcpy(void* dst, const void* src, size_t size, cg_memcpy_kind kind);

/**
 * Async memory copy.
 */
void cg_cuda_memcpy_async(void* dst, const void* src, size_t size, 
                          cg_memcpy_kind kind, void* stream);

/**
 * Set GPU memory to value.
 */
void cg_cuda_memset(void* ptr, int value, size_t size);

/*============================================================================
 * SYNCHRONIZATION
 *============================================================================*/

/**
 * Synchronize current device.
 */
void cg_cuda_device_synchronize(void);

/**
 * Synchronize a stream.
 */
void cg_cuda_stream_synchronize(void* stream);

/*============================================================================
 * TENSOR GPU OPERATIONS
 *============================================================================*/

/**
 * Create a tensor on GPU.
 */
cg_tensor* cg_tensor_cuda_new(int* shape, int ndim, bool requires_grad);

/**
 * Create zeros tensor on GPU.
 */
cg_tensor* cg_tensor_cuda_zeros(int* shape, int ndim, bool requires_grad);

/**
 * Move tensor to GPU (creates new tensor, original unchanged).
 */
cg_tensor* cg_tensor_to_cuda(cg_tensor* t);

/**
 * Move tensor to CPU (creates new tensor, original unchanged).
 */
cg_tensor* cg_tensor_to_cpu(cg_tensor* t);

/**
 * Check if tensor is on GPU.
 */
bool cg_tensor_is_cuda(cg_tensor* t);

/*============================================================================
 * ERROR HANDLING
 *============================================================================*/

/**
 * Get last CUDA error message.
 */
const char* cg_cuda_get_last_error(void);

/**
 * Clear CUDA error state.
 */
void cg_cuda_clear_error(void);

/*============================================================================
 * KERNEL LAUNCH HELPERS
 *============================================================================*/

/**
 * Calculate optimal grid and block dimensions.
 */
void cg_cuda_calculate_launch_dims(int n_elements, int* grid_size, int* block_size);

/**
 * Calculate launch dims for 2D operations.
 */
void cg_cuda_calculate_launch_dims_2d(int rows, int cols, 
                                       int* grid_x, int* grid_y,
                                       int* block_x, int* block_y);

#endif /* CG_CUDA_H */
