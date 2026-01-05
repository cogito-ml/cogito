/**
 * CUDA Device Management Implementation
 * 
 * Provides GPU device management with simulation fallback for non-GPU environments.
 * When CG_USE_CUDA is not defined, all operations are simulated using CPU memory.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cg_cuda.h"
#include "cg_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef CG_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

/*============================================================================
 * GLOBAL STATE
 *============================================================================*/

static cg_cuda_context g_cuda_ctx = {0};
static bool g_cuda_initialized = false;
static char g_last_error[256] = {0};

#ifdef CG_USE_CUDA
/* Real CUDA error checking */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        snprintf(g_last_error, sizeof(g_last_error), \
                 "CUDA error: %s", cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(call, ret) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        snprintf(g_last_error, sizeof(g_last_error), \
                 "CUDA error: %s", cudaGetErrorString(err)); \
        return ret; \
    } \
} while(0)
#endif

/*============================================================================
 * SIMULATION MODE (When CUDA not available)
 *============================================================================*/

#ifndef CG_USE_CUDA

/* Simulated device info for testing */
static cg_cuda_device_info g_simulated_device = {
    .device_id = 0,
    .compute_capability_major = 8,
    .compute_capability_minor = 6,
    .total_memory = 8ULL * 1024 * 1024 * 1024,  /* 8 GB */
    .free_memory = 7ULL * 1024 * 1024 * 1024,   /* 7 GB */
    .multiprocessor_count = 68,
    .max_threads_per_block = 1024,
    .warp_size = 32,
    .shared_memory_per_block = 48 * 1024,       /* 48 KB */
    .device_name = "Simulated CUDA Device (RTX 3080 equivalent)",
    .is_available = true
};

bool cg_cuda_init(void) {
    if (g_cuda_initialized) return true;
    
    g_cuda_ctx.device_id = 0;
    g_cuda_ctx.stream = NULL;
    g_cuda_ctx.cublas_handle = NULL;
    g_cuda_ctx.initialized = true;
    g_cuda_initialized = true;
    
    printf("[CUDA SIM] CUDA simulation mode initialized\n");
    printf("[CUDA SIM] Simulated device: %s\n", g_simulated_device.device_name);
    printf("[CUDA SIM] Compute capability: %d.%d\n", 
           g_simulated_device.compute_capability_major,
           g_simulated_device.compute_capability_minor);
    
    return true;
}

void cg_cuda_shutdown(void) {
    if (!g_cuda_initialized) return;
    
    g_cuda_ctx.initialized = false;
    g_cuda_initialized = false;
    
    printf("[CUDA SIM] CUDA simulation shutdown\n");
}

bool cg_cuda_is_available(void) {
    return g_cuda_initialized;
}

int cg_cuda_device_count(void) {
    return 1;  /* Simulate single GPU */
}

cg_cuda_device_info cg_cuda_get_device_info(int device_id) {
    (void)device_id;
    return g_simulated_device;
}

void cg_cuda_set_device(int device_id) {
    g_cuda_ctx.device_id = device_id;
}

int cg_cuda_get_device(void) {
    return g_cuda_ctx.device_id;
}

cg_cuda_context* cg_cuda_get_context(void) {
    return &g_cuda_ctx;
}

/* Memory operations - simulate using regular malloc */
void* cg_cuda_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr) {
        g_simulated_device.free_memory -= size;
    }
    return ptr;
}

void cg_cuda_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

void* cg_cuda_malloc_host(size_t size) {
    /* In simulation, pinned memory is just regular memory */
    return malloc(size);
}

void cg_cuda_free_host(void* ptr) {
    free(ptr);
}

void cg_cuda_memcpy(void* dst, const void* src, size_t size, cg_memcpy_kind kind) {
    (void)kind;  /* All directions are same in simulation */
    memcpy(dst, src, size);
}

void cg_cuda_memcpy_async(void* dst, const void* src, size_t size, 
                          cg_memcpy_kind kind, void* stream) {
    (void)stream;
    cg_cuda_memcpy(dst, src, size, kind);
}

void cg_cuda_memset(void* ptr, int value, size_t size) {
    memset(ptr, value, size);
}

void cg_cuda_device_synchronize(void) {
    /* No-op in simulation */
}

void cg_cuda_stream_synchronize(void* stream) {
    (void)stream;
    /* No-op in simulation */
}

#else /* CG_USE_CUDA defined - Real CUDA implementation */

bool cg_cuda_init(void) {
    if (g_cuda_initialized) return true;
    
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        snprintf(g_last_error, sizeof(g_last_error), 
                 "No CUDA devices found");
        return false;
    }
    
    CUDA_CHECK_RET(cudaSetDevice(0), false);
    
    cudaStream_t stream;
    CUDA_CHECK_RET(cudaStreamCreate(&stream), false);
    g_cuda_ctx.stream = stream;
    
    cublasHandle_t handle;
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        snprintf(g_last_error, sizeof(g_last_error), 
                 "cuBLAS initialization failed");
        cudaStreamDestroy(stream);
        return false;
    }
    g_cuda_ctx.cublas_handle = handle;
    
    cublasSetStream(handle, stream);
    
    g_cuda_ctx.device_id = 0;
    g_cuda_ctx.initialized = true;
    g_cuda_initialized = true;
    
    return true;
}

void cg_cuda_shutdown(void) {
    if (!g_cuda_initialized) return;
    
    if (g_cuda_ctx.cublas_handle) {
        cublasDestroy((cublasHandle_t)g_cuda_ctx.cublas_handle);
    }
    if (g_cuda_ctx.stream) {
        cudaStreamDestroy((cudaStream_t)g_cuda_ctx.stream);
    }
    
    g_cuda_ctx.initialized = false;
    g_cuda_initialized = false;
}

bool cg_cuda_is_available(void) {
    return g_cuda_initialized;
}

int cg_cuda_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

cg_cuda_device_info cg_cuda_get_device_info(int device_id) {
    cg_cuda_device_info info = {0};
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        info.is_available = false;
        return info;
    }
    
    info.device_id = device_id;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_memory = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.shared_memory_per_block = prop.sharedMemPerBlock;
    strncpy(info.device_name, prop.name, sizeof(info.device_name) - 1);
    info.is_available = true;
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory = free_mem;
    
    return info;
}

void cg_cuda_set_device(int device_id) {
    cudaSetDevice(device_id);
    g_cuda_ctx.device_id = device_id;
}

int cg_cuda_get_device(void) {
    return g_cuda_ctx.device_id;
}

cg_cuda_context* cg_cuda_get_context(void) {
    return &g_cuda_ctx;
}

void* cg_cuda_malloc(size_t size) {
    void* ptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
        return NULL;
    }
    return ptr;
}

void cg_cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void* cg_cuda_malloc_host(size_t size) {
    void* ptr;
    if (cudaMallocHost(&ptr, size) != cudaSuccess) {
        return NULL;
    }
    return ptr;
}

void cg_cuda_free_host(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void cg_cuda_memcpy(void* dst, const void* src, size_t size, cg_memcpy_kind kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case CG_MEMCPY_HOST_TO_DEVICE: cuda_kind = cudaMemcpyHostToDevice; break;
        case CG_MEMCPY_DEVICE_TO_HOST: cuda_kind = cudaMemcpyDeviceToHost; break;
        case CG_MEMCPY_DEVICE_TO_DEVICE: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default: return;
    }
    cudaMemcpy(dst, src, size, cuda_kind);
}

void cg_cuda_memcpy_async(void* dst, const void* src, size_t size, 
                          cg_memcpy_kind kind, void* stream) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case CG_MEMCPY_HOST_TO_DEVICE: cuda_kind = cudaMemcpyHostToDevice; break;
        case CG_MEMCPY_DEVICE_TO_HOST: cuda_kind = cudaMemcpyDeviceToHost; break;
        case CG_MEMCPY_DEVICE_TO_DEVICE: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default: return;
    }
    cudaMemcpyAsync(dst, src, size, cuda_kind, (cudaStream_t)stream);
}

void cg_cuda_memset(void* ptr, int value, size_t size) {
    cudaMemset(ptr, value, size);
}

void cg_cuda_device_synchronize(void) {
    cudaDeviceSynchronize();
}

void cg_cuda_stream_synchronize(void* stream) {
    cudaStreamSynchronize((cudaStream_t)stream);
}

#endif /* CG_USE_CUDA */

/*============================================================================
 * COMMON IMPLEMENTATION (Works for both modes)
 *============================================================================*/

const char* cg_cuda_get_last_error(void) {
    return g_last_error;
}

void cg_cuda_clear_error(void) {
    g_last_error[0] = '\0';
#ifdef CG_USE_CUDA
    cudaGetLastError();  /* Clear CUDA error state */
#endif
}

void cg_cuda_calculate_launch_dims(int n_elements, int* grid_size, int* block_size) {
    *block_size = 256;  /* Standard block size */
    *grid_size = (n_elements + *block_size - 1) / *block_size;
    
    /* Cap grid size for very large arrays */
    if (*grid_size > 65535) {
        *grid_size = 65535;
    }
}

void cg_cuda_calculate_launch_dims_2d(int rows, int cols, 
                                       int* grid_x, int* grid_y,
                                       int* block_x, int* block_y) {
    /* Standard 2D block: 16x16 = 256 threads */
    *block_x = 16;
    *block_y = 16;
    
    *grid_x = (cols + *block_x - 1) / *block_x;
    *grid_y = (rows + *block_y - 1) / *block_y;
}

/*============================================================================
 * TENSOR GPU OPERATIONS
 *============================================================================*/

cg_tensor* cg_tensor_cuda_new(int* shape, int ndim, bool requires_grad) {
    cg_tensor* t = (cg_tensor*)calloc(1, sizeof(cg_tensor));
    if (!t) return NULL;
    
    /* Calculate size and strides */
    t->ndim = ndim;
    t->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->shape[i] = shape[i];
        t->strides[i] = t->size;
        t->size *= shape[i];
    }
    
    /* Allocate GPU memory */
    t->data = (float*)cg_cuda_malloc(t->size * sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }
    
    /* Allocate gradient if needed */
    if (requires_grad) {
        t->grad = (float*)cg_cuda_malloc(t->size * sizeof(float));
        if (!t->grad) {
            cg_cuda_free(t->data);
            free(t);
            return NULL;
        }
        cg_cuda_memset(t->grad, 0, t->size * sizeof(float));
    }
    
    t->requires_grad = requires_grad;
    t->ref_count = 1;
    
    return t;
}

cg_tensor* cg_tensor_cuda_zeros(int* shape, int ndim, bool requires_grad) {
    cg_tensor* t = cg_tensor_cuda_new(shape, ndim, requires_grad);
    if (t) {
        cg_cuda_memset(t->data, 0, t->size * sizeof(float));
    }
    return t;
}

cg_tensor* cg_tensor_to_cuda(cg_tensor* t) {
    if (!t) return NULL;
    
    cg_tensor* gpu_t = cg_tensor_cuda_new(t->shape, t->ndim, t->requires_grad);
    if (!gpu_t) return NULL;
    
    /* Copy data to GPU */
    cg_cuda_memcpy(gpu_t->data, t->data, t->size * sizeof(float), 
                   CG_MEMCPY_HOST_TO_DEVICE);
    
    if (t->requires_grad && t->grad) {
        cg_cuda_memcpy(gpu_t->grad, t->grad, t->size * sizeof(float), 
                       CG_MEMCPY_HOST_TO_DEVICE);
    }
    
    return gpu_t;
}

cg_tensor* cg_tensor_to_cpu(cg_tensor* t) {
    if (!t) return NULL;
    
    /* Allocate CPU tensor */
    cg_tensor* cpu_t = (cg_tensor*)calloc(1, sizeof(cg_tensor));
    if (!cpu_t) return NULL;
    
    /* Copy metadata */
    cpu_t->ndim = t->ndim;
    cpu_t->size = t->size;
    cpu_t->requires_grad = t->requires_grad;
    cpu_t->ref_count = 1;
    memcpy(cpu_t->shape, t->shape, sizeof(t->shape));
    memcpy(cpu_t->strides, t->strides, sizeof(t->strides));
    
    /* Allocate CPU memory */
    cpu_t->data = (float*)malloc(t->size * sizeof(float));
    if (!cpu_t->data) {
        free(cpu_t);
        return NULL;
    }
    
    /* Copy data from GPU */
    cg_cuda_memcpy(cpu_t->data, t->data, t->size * sizeof(float), 
                   CG_MEMCPY_DEVICE_TO_HOST);
    
    if (t->requires_grad && t->grad) {
        cpu_t->grad = (float*)malloc(t->size * sizeof(float));
        if (cpu_t->grad) {
            cg_cuda_memcpy(cpu_t->grad, t->grad, t->size * sizeof(float), 
                           CG_MEMCPY_DEVICE_TO_HOST);
        }
    }
    
    return cpu_t;
}

bool cg_tensor_is_cuda(cg_tensor* t) {
    (void)t;
    /* In full implementation, check tensor's device field */
    /* For now, return based on whether CUDA is initialized */
    return g_cuda_initialized;
}

#ifdef __cplusplus
}
#endif
