cmake_minimum_required(VERSION 3.16)
project(cogito VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(COGITO_USE_BLAS "Use OpenBLAS for matrix operations" ON)
option(COGITO_BUILD_EXAMPLES "Build example programs" ON)
option(COGITO_BUILD_TESTS "Build test programs" ON)

# Find OpenBLAS if requested
if(COGITO_USE_BLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
        add_compile_definitions(CG_USE_BLAS)
    else()
        message(WARNING "BLAS not found, using naive implementations")
        set(COGITO_USE_BLAS OFF)
    endif()
endif()

# Platform-specific settings
if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect source files
set(COGITO_SOURCES
    src/utils/arena.c
    src/utils/dataloader.c
    src/core/tensor.c
    src/core/tensor_ops.c
    src/core/autograd.c
    src/core/graph.c
    src/core/broadcast.c
    src/core/symbolic.c
    src/core/bf16.c
    src/jit/graph_capture.c
    src/jit/fusion.c
    src/attention/flash_attn.c
    src/memory/planner.c
    src/quantize/int4.c
    src/inference/speculative.c
    src/layers/linear.c
    src/layers/activations.c
    src/layers/conv2d.c
    src/layers/loss.c
    src/optim/sgd.c
    src/optim/adam.c
    src/datasets/mnist.c
)


# Create static library
add_library(cogito_static STATIC ${COGITO_SOURCES})
set_target_properties(cogito_static PROPERTIES OUTPUT_NAME "cogito")

# Create shared library for bindings
add_library(cogito SHARED ${COGITO_SOURCES})
set_target_properties(cogito PROPERTIES 
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)


if(COGITO_USE_BLAS AND BLAS_FOUND)
    target_link_libraries(cogito ${BLAS_LIBRARIES})
    target_link_libraries(cogito_static ${BLAS_LIBRARIES})
endif()

# Math library (only needed on Unix - MSVC includes it in standard library)
if(UNIX)
    target_link_libraries(cogito m)
    target_link_libraries(cogito_static m)
endif()

# Examples
if(COGITO_BUILD_EXAMPLES)
    add_executable(xor_example examples/xor.c)
    target_link_libraries(xor_example cogito)

    add_executable(mnist_example examples/mnist.c)
    target_link_libraries(mnist_example cogito)
endif()

# Tests
if(COGITO_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_tensor tests/test_tensor.c)
    target_link_libraries(test_tensor cogito)
    add_test(NAME test_tensor COMMAND test_tensor)
    
    add_executable(test_autograd tests/test_autograd.c)
    target_link_libraries(test_autograd cogito)
    add_test(NAME test_autograd COMMAND test_autograd)
endif()

# Installation
install(TARGETS cogito ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
