cmake_minimum_required(VERSION 3.18)

# Check for CUDA before project declaration
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(cogito VERSION 0.1.0 LANGUAGES C CXX CUDA)
    set(CUDA_AVAILABLE TRUE)
else()
    project(cogito VERSION 0.1.0 LANGUAGES C CXX)
    set(CUDA_AVAILABLE FALSE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(COGITO_USE_BLAS "Use OpenBLAS for matrix operations" ON)
option(COGITO_USE_CUDA "Build CUDA kernels (requires CUDA Toolkit)" ${CUDA_AVAILABLE})
option(COGITO_CUDA_SIMULATION "Use CPU simulation for CUDA kernels" ON)
option(COGITO_BUILD_EXAMPLES "Build example programs" ON)
option(COGITO_BUILD_TESTS "Build test programs" ON)

message(STATUS "Cogito Build Configuration:")
message(STATUS "  CUDA Available: ${CUDA_AVAILABLE}")
message(STATUS "  CUDA Enabled: ${COGITO_USE_CUDA}")
message(STATUS "  CUDA Simulation: ${COGITO_CUDA_SIMULATION}")

#============================================================================
# CUDA Configuration
#============================================================================

if(COGITO_USE_CUDA AND CUDA_AVAILABLE)
    find_package(CUDAToolkit REQUIRED)
    
    # Set CUDA architectures (adjust based on target GPUs)
    # 70=V100, 75=Turing, 80=A100, 86=RTX30xx, 89=RTX40xx, 90=H100
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
    
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  cuBLAS: ${CUDAToolkit_LIBRARY_DIR}")
    
    add_compile_definitions(CG_USE_CUDA)
    
    # CUDA compile options
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

#============================================================================
# BLAS Configuration
#============================================================================

if(COGITO_USE_BLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        message(STATUS "  BLAS: ${BLAS_LIBRARIES}")
        add_compile_definitions(CG_USE_BLAS)
    else()
        message(WARNING "BLAS not found, using naive implementations")
        set(COGITO_USE_BLAS OFF)
    endif()
endif()

#============================================================================
# Platform Configuration
#============================================================================

if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

#============================================================================
# Source Files
#============================================================================

set(COGITO_SOURCES
    # Core
    src/core/tensor.c
    src/core/tensor_ops.c
    src/core/autograd.c
    src/core/graph.c
    src/core/broadcast.c
    src/core/symbolic.c
    src/core/bf16.c
    
    # Utils
    src/utils/arena.c
    src/utils/dataloader.c
    
    # JIT
    src/jit/graph_capture.c
    src/jit/fusion.c
    
    # Attention
    src/attention/flash_attn.c
    
    # Memory
    src/memory/planner.c
    
    # Quantization
    src/quantize/int4.c
    
    # Inference
    src/inference/speculative.c
    
    # Layers
    src/layers/linear.c
    src/layers/activations.c
    src/layers/conv2d.c
    src/layers/loss.c
    
    # Optimizers
    src/optim/sgd.c
    src/optim/adam.c
    
    # Datasets
    src/datasets/mnist.c
)

# CUDA sources (always include - uses simulation if CUDA unavailable)
set(COGITO_CUDA_SOURCES
    src/cuda/cg_cuda.c
    src/cuda/tensor_kernels.c
    src/cuda/flash_attn_kernels.c
)

# Combine all sources
list(APPEND COGITO_SOURCES ${COGITO_CUDA_SOURCES})

#============================================================================
# Library Targets
#============================================================================

# Static library
add_library(cogito_static STATIC ${COGITO_SOURCES})
set_target_properties(cogito_static PROPERTIES OUTPUT_NAME "cogito")

# Shared library
add_library(cogito SHARED ${COGITO_SOURCES})
set_target_properties(cogito PROPERTIES 
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Link dependencies
if(COGITO_USE_BLAS AND BLAS_FOUND)
    target_link_libraries(cogito ${BLAS_LIBRARIES})
    target_link_libraries(cogito_static ${BLAS_LIBRARIES})
endif()

if(COGITO_USE_CUDA AND CUDA_AVAILABLE)
    target_link_libraries(cogito CUDA::cudart CUDA::cublas)
    target_link_libraries(cogito_static CUDA::cudart CUDA::cublas)
endif()

# Math library (only needed on Unix)
if(UNIX)
    target_link_libraries(cogito m)
    target_link_libraries(cogito_static m)
endif()

#============================================================================
# Examples
#============================================================================

if(COGITO_BUILD_EXAMPLES)
    add_executable(xor_example examples/xor.c)
    target_link_libraries(xor_example cogito)

    add_executable(mnist_example examples/mnist.c)
    target_link_libraries(mnist_example cogito)
endif()

#============================================================================
# Tests
#============================================================================

if(COGITO_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_tensor tests/test_tensor.c)
    target_link_libraries(test_tensor cogito)
    add_test(NAME test_tensor COMMAND test_tensor)
    
    add_executable(test_autograd tests/test_autograd.c)
    target_link_libraries(test_autograd cogito)
    add_test(NAME test_autograd COMMAND test_autograd)
    
    add_executable(test_cuda tests/test_cuda.c)
    target_link_libraries(test_cuda cogito)
    add_test(NAME test_cuda COMMAND test_cuda)
endif()

#============================================================================
# Installation
#============================================================================

install(TARGETS cogito cogito_static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

#============================================================================
# Summary
#============================================================================

message(STATUS "")
message(STATUS "Cogito ${PROJECT_VERSION} configured successfully")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
if(COGITO_USE_CUDA AND CUDA_AVAILABLE)
    message(STATUS "  GPU acceleration: ENABLED")
else()
    message(STATUS "  GPU acceleration: CPU SIMULATION")
endif()
message(STATUS "")

