# Cogito

*"Cogito, ergo sum" — I think, therefore I am.*

A minimal, educational C machine learning library. No dependencies, explicit memory management, pedagogical clarity over magic.

## Features

- **Tensor System**: N-dimensional tensors with automatic differentiation
- **Autograd Engine**: Reverse-mode autodiff with computational graph
- **Neural Network Layers**: Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout, BatchNorm
- **Loss Functions**: MSE, Cross-Entropy, BCE, L1, Smooth L1
- **Optimizers**: SGD (with momentum/Nesterov), Adam
- **Memory Management**: Arena allocator for zero-fragmentation training
- **BLAS Support**: Optional OpenBLAS acceleration with naive fallback

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -G "MinGW Makefiles"  # or "Unix Makefiles" on Linux/macOS
cmake --build .

# Run tests
ctest --output-on-failure

# Run XOR example
./xor_example

# Run MNIST example (requires MNIST dataset files)
./mnist_example
```

## Example: XOR

```c
#include "cogito.h"

int main(void) {
    // Build model: 2 -> 4 -> 1
    cg_sequential* model = cg_sequential_new();
    cg_sequential_add(model, (cg_layer*)cg_linear_new(2, 4, true));
    cg_sequential_add(model, (cg_layer*)cg_relu_new());
    cg_sequential_add(model, (cg_layer*)cg_linear_new(4, 1, true));
    cg_sequential_add(model, (cg_layer*)cg_sigmoid_new());
    
    // Train with SGD
    cg_sgd* opt = cg_sgd_new_for_sequential(model, 0.5f, 0.9f, 0, false);
    
    // Training loop...
    for (int epoch = 0; epoch < 1000; epoch++) {
        cg_tensor* pred = cg_sequential_forward(model, X);
        cg_tensor* loss = cg_mse_loss(pred, y, CG_REDUCTION_MEAN);
        
        cg_optimizer_zero_grad((cg_optimizer*)opt);
        cg_backward(loss);
        cg_optimizer_step((cg_optimizer*)opt);
    }
    
    return 0;
}
```

## Project Structure

```
cogito/
├── include/
│   ├── cogito.h        # Main header
│   ├── cg_tensor.h     # Tensor operations
│   ├── cg_layers.h     # Neural network layers
│   ├── cg_optim.h      # Optimizers
│   ├── cg_loss.h       # Loss functions
│   └── cg_datasets.h   # Data loading
├── src/
│   ├── core/           # Tensor engine + autograd
│   ├── layers/         # Layer implementations
│   ├── optim/          # Optimizer implementations
│   ├── datasets/       # Data loaders
│   └── utils/          # Arena allocator
├── examples/
│   ├── xor.c           # Learn XOR function
│   └── mnist.c         # MNIST digit classification
└── tests/
    ├── test_tensor.c   # Tensor unit tests
    └── test_autograd.c # Gradient verification
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| BLAS | OpenBLAS with fallback | Performance when available |
| Memory | Arena allocator | Zero fragmentation |
| Graph | Static (rebuild each forward) | Simpler in C |
| Tensors | Copy semantics | Clear ownership |
| Errors | `assert()` | Fail-fast for learning |

## License

MIT
