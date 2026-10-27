# NeuralEngine

A neural network engine written in **C++20**, built from scratch to explore the mathematics, algorithms, and implementation details behind neural networks.

NeuralEngine implements the core training pipeline without relying on a high-level machine-learning framework. It includes tensor storage, fully connected layers, activation functions, loss calculation, backpropagation, stochastic gradient descent, and an MNIST training pipeline.

The project focuses on understanding how neural networks work internally, from matrix operations and derivatives to gradient propagation and parameter updates.

## Features

- **Tensor library:** Shape-aware tensor storage and indexing.
- **Dense layers:** Fully connected layers with configurable weight initialisation.
- **Activation functions:** ReLU for hidden layers.
- **Loss function:** Numerically stable softmax cross-entropy.
- **Backpropagation:** Analytical gradients calculated directly using matrix operations and the chain rule.
- **Optimisation:** Stochastic gradient descent (SGD).
- **MNIST support:** Dataset loading, training, and test evaluation.
- **Gradient verification:** Tests comparing analytical gradients with numerical approximations.
- **NeuralEngine Studio:** A graphical interface for experimenting with the engine, if built with the required Raylib dependency.

## MNIST Results

NeuralEngine was trained on the MNIST handwritten-digit dataset using a fully connected neural network.

| Metric | Result |
|---|---:|
| Training dataset | 60,000 images |
| Test dataset | 10,000 images |
| Network architecture | 784 → 512 → 256 → 128 → 64 → 10 |
| Optimiser | Stochastic Gradient Descent |
| Learning rate | 0.01 |
| Epochs | 5 |
| Test accuracy | **97.81%** |

These results are from the current implementation and training configuration. Results may vary with changes to initialisation, data ordering, hyperparameters, or implementation details.

## Mathematical Foundations

### 1. Forward Propagation

For layer \(l\), the pre-activation output is:

$$
\mathbf{z}^{(l)}
=
W^{(l)}\mathbf{a}^{(l-1)}
+
\mathbf{b}^{(l)}
$$

Here:

- \(W^{(l)}\) is the weight matrix.
- \(\mathbf{b}^{(l)}\) is the bias vector.
- \(\mathbf{a}^{(l-1)}\) is the input activation vector.
- \(\mathbf{z}^{(l)}\) is the pre-activation output.

For hidden layers, ReLU produces the activation vector:

$$
\mathbf{a}^{(l)}
=
\max\left(\mathbf{0},\mathbf{z}^{(l)}\right)
$$

The final layer produces logits without applying ReLU. These logits are passed to the loss function.

### 2. Softmax Cross-Entropy

For logits \(z_j\), the softmax probability of class \(j\) is:

$$
p_j
=
\frac{e^{z_j}}{\sum_k e^{z_k}}
$$

To improve numerical stability, the loss uses the Log-Sum-Exp formulation. Let

$$
m = \max_j z_j
$$

For target class \(y\), the cross-entropy loss is:

$$
\mathcal{L}
=
m
+
\ln\left(\sum_j e^{z_j-m}\right)
-
z_y
$$

For a one-hot target vector \(\mathbf{y}\), the gradient with respect to the logits simplifies to:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{z}}
=
\mathbf{p}-\mathbf{y}
$$

This simplification avoids explicitly calculating the full softmax Jacobian during backpropagation.

### 3. Dense-Layer Backpropagation

Define the incoming gradient for layer \(l\) as:

$$
\boldsymbol{\delta}^{(l)}
=
\frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(l)}}
$$

The weight gradient is the outer product of the error vector and the preceding layer's activation:

$$
\frac{\partial\mathcal{L}}{\partial W^{(l)}}
=
\boldsymbol{\delta}^{(l)}
\left(\mathbf{a}^{(l-1)}\right)^T
$$

The bias gradient is:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(l)}}
=
\boldsymbol{\delta}^{(l)}
$$

The gradient propagated to the previous activation vector is:

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{a}^{(l-1)}}
=
\left(W^{(l)}\right)^T
\boldsymbol{\delta}^{(l)}
$$

For a hidden ReLU layer, the gradient is multiplied elementwise by the derivative of ReLU:

$$
\boldsymbol{\delta}^{(l-1)}
=
\left[
\left(W^{(l)}\right)^T
\boldsymbol{\delta}^{(l)}
\right]
\odot
\mathbf{1}_{\mathbf{z}^{(l-1)}>0}
$$

Here, \(\odot\) denotes elementwise multiplication, and the indicator vector is 1 where the corresponding pre-activation is positive and 0 otherwise.

NeuralEngine implements these operations directly rather than relying on an external automatic differentiation framework.

### 4. Numerical Gradient Verification

Analytical gradients are checked against central finite differences:

$$
g_{\text{numeric}}
\approx
\frac{
\mathcal{L}(\theta+\varepsilon)
-
\mathcal{L}(\theta-\varepsilon)
}{
2\varepsilon
}
$$

Here, \(\theta\) represents a parameter being checked and \(\varepsilon\) is a small perturbation.

Comparing numerical and analytical gradients helps identify errors in the backward-pass implementation.

## Project Structure

```text
.
├── CMakeLists.txt
├── include/
│   └── neural/
│       ├── data/
│       │   └── dataset.hpp
│       ├── nn/
│       │   ├── layer.hpp
│       │   └── model.hpp
│       ├── ops/
│       │   ├── activations.hpp
│       │   ├── arithmetic.hpp
│       │   └── loss.hpp
│       ├── optim/
│       │   └── optimizer.hpp
│       └── tensor/
│           └── tensor.hpp
├── src/
│   ├── data/
│   │   └── dataset.cpp
│   ├── nn/
│   │   └── layer.cpp
│   ├── ops/
│   │   ├── activations.cpp
│   │   └── loss.cpp
│   ├── optim/
│   │   └── optimizer.cpp
│   └── tensor/
│       └── tensor.cpp
├── examples/
│   ├── basic.cpp
│   └── studio.cpp
├── tests/
│   ├── test_main.cpp
│   └── test_gradients.cpp
└── data/
    └── mnist/
```

## Building and Testing

### Requirements

- A C++20-compatible compiler, such as GCC 11+, Clang 13+, or a suitable MSVC version.
- CMake 3.20 or newer.
- Raylib, if building NeuralEngine Studio.

The core engine and headless example do not require Raylib.

### 1. Download the MNIST Dataset

From the project root:

```bash
mkdir -p data/mnist
cd data/mnist

curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
cd ../..
```

The dataset parser expects the extracted, uncompressed IDX files.

### 2. Configure and Compile

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 3. Run the Tests

```bash
ctest --test-dir build --output-on-failure
```

The test suite checks core tensor functionality and analytical gradients against numerical estimates, assuming the corresponding tests are enabled in the CMake configuration.

### 4. Train and Evaluate

Run the headless training example:

```bash
./build/basic_example
```

This command assumes the executable target is named `basic_example` and that the example's dataset paths are compatible with the working directory. Check your `CMakeLists.txt` and `examples/basic.cpp` if the executable or dataset cannot be found.

## Programmatic Usage

The following example demonstrates a single training step using the engine's C++ API. It creates a small network, performs a forward pass, calculates the loss, propagates gradients backwards, and updates the parameters.

```cpp
#include "neural/data/dataset.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

#include <cstddef>

int main() {
    neural::Dense fc1(784, 512, neural::WeightInit::He, 101);
    neural::Dense fc2(512, 10, neural::WeightInit::Xavier, 102);

    neural::ReLU relu;
    neural::SGD optimizer(0.01F);

    neural::MNISTDataset dataset(
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte"
    );

    const neural::Tensor input = dataset.image(0);
    const std::size_t target = dataset.label(0);

    // Forward pass
    const neural::Tensor hidden =
        relu.forward(fc1.forward(input));

    const neural::Tensor logits =
        fc2.forward(hidden);

    // Calculate loss and gradient with respect to logits
    const auto result =
        neural::softmax_cross_entropy(logits, target);

    // Backward pass: output layer, activation, input layer
    const neural::Tensor hidden_gradient =
        fc2.backward(result.gradient);

    const neural::Tensor input_gradient =
        fc1.backward(relu.backward(hidden_gradient));

    // Update parameters after gradients have been calculated
    optimizer.step(fc1);
    optimizer.step(fc2);

    (void)input_gradient;
    return 0;
}
```

**Note:** This example assumes `softmax_cross_entropy` returns a result with a `gradient` member, `Dense::backward` accepts the engine's tensor type, and `SGD::step` accepts a `Dense` layer. Adjust the example if your actual API uses different names or signatures.

## Design Goals

NeuralEngine is intended to make the internal mechanics of neural networks explicit.

- **Understandability:** Core algorithms are implemented directly rather than delegated to a high-level framework.
- **Mathematical transparency:** Forward propagation, loss calculation, and backpropagation correspond to explicit mathematical operations.
- **Modularity:** Tensor operations, neural-network layers, data loading, and optimisation are separated into distinct components.
- **Extensibility:** The architecture provides a foundation for adding more operations, layers, and optimisation methods.
- **Testing:** Numerical gradient checks help validate the correctness of analytical derivatives.

## Future Work

Potential directions for development include:

- Additional activation functions and loss functions.
- More optimisation algorithms, such as Adam.
- Batch-based training and matrix operations optimised for larger workloads.
- Convolutional layers and other neural-network architectures.
- Improved tensor operations and performance profiling.
- More extensive gradient checks and automated tests.
- Expanded visualisation and experimentation tools in NeuralEngine Studio.

## Acknowledgements

NeuralEngine was developed as a learning project to explore neural-network mathematics, C++ systems programming, and the implementation of machine-learning algorithms from scratch.

The MNIST dataset is used for handwritten-digit classification.

## License

This project is distributed under the terms of the MIT License, if the repository's `LICENSE` file specifies that licence.