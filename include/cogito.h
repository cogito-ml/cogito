/**
 * Cogito - A Minimal C Machine Learning Library
 * 
 * "Cogito, ergo sum" - I think, therefore I am.
 * 
 * Philosophy: Minimal dependencies, explicit memory management, 
 *             pedagogical clarity over magic.
 */

#ifndef COGITO_H
#define COGITO_H

#include "cg_arena.h"
#include "cg_tensor.h"
#include "cg_layers.h"
#include "cg_optim.h"
#include "cg_loss.h"
#include "cg_datasets.h"
#include "cg_jit.h"
#include "cg_symbolic.h"
#include "cg_bf16.h"
#include "cg_pipeline.h"

/* CUDA/GPU support */
#include "cg_cuda.h"
#include "cg_tensor_kernels.h"
#include "cg_flash_attn_kernels.h"

/* Version information */
#define COGITO_VERSION_MAJOR 0
#define COGITO_VERSION_MINOR 1
#define COGITO_VERSION_PATCH 0

/* Error codes */
typedef enum {
    CG_SUCCESS = 0,
    CG_ERROR_NULL_POINTER,
    CG_ERROR_SHAPE_MISMATCH,
    CG_ERROR_OUT_OF_MEMORY,
    CG_ERROR_INVALID_ARGUMENT,
    CG_ERROR_FILE_NOT_FOUND,
    CG_ERROR_INVALID_FORMAT
} cg_error;

/* Get human-readable error message */
const char* cg_error_string(cg_error err);

#endif /* COGITO_H */
