# NeuralEngine

A neural network engine written in C++20, built from scratch to explore the mathematics and implementation of neural networks.

## Current Features

- N-dimensional tensors with contiguous storage
- Fully connected (dense) layers
- ReLU, sigmoid, and tanh activation functions
- Softmax cross-entropy loss
- Backpropagation and stochastic gradient descent
- MNIST dataset loader
- Numerical gradient checks

## MNIST Results

| Metric | Result |
|---|---:|
| Test accuracy | 94.61% |
| Correct predictions | 9,461 / 10,000 |
| Mean test loss | 0.178783 |
| Training epochs | 1 |

**Model architecture:** 784 → 512 → 256 → 128 → 64 → 10

The model was trained on MNIST using per-example stochastic gradient descent.

## Build

Requires a C++20 compiler and CMake.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run the example from the build directory:

```bash
cd build
./basic_example
```

The MNIST dataset files must be placed in `data/mnist/` in the project root, with the expected IDX filenames.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

## Status

This project is under active development. Mini-batch training, performance improvements, and model persistence are planned.
