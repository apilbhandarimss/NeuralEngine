# NeuralEngine

A neural network engine written in **C++20**, built from scratch to explore the mathematics, algorithms, and implementation details behind neural networks.

NeuralEngine implements the core training pipeline without relying on a high-level machine-learning framework. It includes tensor storage, fully connected layers, activation functions, loss calculation, backpropagation, stochastic gradient descent, batch normalisation, and an MNIST training pipeline.

The project focuses on understanding how neural networks work internally, from tensor operations and derivatives to gradient propagation, batch statistics, and parameter updates.

## Features

- **Tensor library:** Shape-aware tensor storage and indexing.
- **Dense layers:** Fully connected layers with configurable weight initialisation.
- **Activation functions:** ReLU for hidden layers.
- **Loss function:** Numerically stable softmax cross-entropy.
- **Backpropagation:** Analytical gradients calculated directly using the chain rule.
- **Mini-batch training:** Processes multiple training examples together.
- **Batch normalisation:** Normalises activations using batch statistics and learnable scale and shift parameters.
- **Optimisation:** Stochastic gradient descent (SGD).
- **MNIST support:** Dataset loading, training, and test evaluation.
- **Gradient verification:** Tests comparing analytical gradients with numerical approximations.
- **NeuralEngine Studio:** A graphical interface for experimenting with the engine when built with the required Raylib dependency.

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

These results are from a reported training run. Results may vary with changes to initialisation, data ordering, hyperparameters, and implementation details.

## Mathematical Foundations

### 1. Forward Propagation

For layer $l$, the pre-activation output is

$$
\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

Here:

- $W^{(l)}$ is the weight matrix.
- $\mathbf{b}^{(l)}$ is the bias vector.
- $\mathbf{a}^{(l-1)}$ is the input activation vector.
- $\mathbf{z}^{(l)}$ is the pre-activation output.

For hidden layers, ReLU produces the activation vector

$$
\mathbf{a}^{(l)} = \mathrm{ReLU}\left(\mathbf{z}^{(l)}\right) = \max\left(\mathbf{0},\mathbf{z}^{(l)}\right)
$$

The final layer produces logits without applying ReLU. These logits are passed to the loss function.

### 2. Softmax Cross-Entropy

For logits $z_1,z_2,\ldots,z_C$, the softmax probability of class $j$ is

$$
p_j = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}
$$

To improve numerical stability, the loss uses the Log-Sum-Exp formulation. Let

$$
m = \max_{1\leq j\leq C} z_j
$$

For target class $y$, the cross-entropy loss is

$$
\mathcal{L} = m + \ln\left(\sum_{j=1}^{C} e^{z_j-m}\right) - z_y
$$

For a one-hot target vector $\mathbf{y}$, the gradient with respect to the logits simplifies to

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{z}} = \mathbf{p} - \mathbf{y}
$$

This simplification avoids explicitly calculating the full softmax Jacobian during backpropagation.

### 3. Dense-Layer Backpropagation

Define the incoming gradient for layer $l$ as

$$
\boldsymbol{\delta}^{(l)} = \frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(l)}}
$$

The weight gradient is the outer product of the error vector and the preceding layer's activation:

$$
\frac{\partial\mathcal{L}}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)}\left(\mathbf{a}^{(l-1)}\right)^{T}
$$

The bias gradient is

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}
$$

The gradient propagated to the previous activation vector is

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{a}^{(l-1)}} = \left(W^{(l)}\right)^{T}\boldsymbol{\delta}^{(l)}
$$

For a hidden ReLU layer, the gradient is multiplied elementwise by the derivative of ReLU:

$$
\boldsymbol{\delta}^{(l-1)} = \left[\left(W^{(l)}\right)^{T}\boldsymbol{\delta}^{(l)}\right] \odot \mathbf{1}_{\mathbf{z}^{(l-1)}>0}
$$

Here, $\odot$ denotes elementwise multiplication, and the indicator vector is one where the corresponding pre-activation is positive and zero otherwise.

NeuralEngine implements these operations directly rather than relying on an external automatic differentiation framework.

### 4. Mini-Batch Training

Instead of processing only one example before updating the parameters, mini-batch training processes a group of $B$ examples together.

For individual losses $\mathcal{L}_1,\mathcal{L}_2,\ldots,\mathcal{L}_B$, the mean batch loss is

$$
\mathcal{L}_{\mathrm{batch}} = \frac{1}{B}\sum_{i=1}^{B}\mathcal{L}_i
$$

This formulation gives each example equal weight in the batch objective.

For example, a batch of 32 images produces a mean loss over those 32 examples. The gradients are propagated through the batch, and the optimiser updates the parameters according to the resulting gradients.

#### Gradient scaling

For one example, the gradient of softmax cross-entropy with respect to the logits is

$$
\frac{\partial\mathcal{L}_i}{\partial\mathbf{z}_i} = \mathbf{p}_i - \mathbf{y}_i
$$

For the mean batch loss, it becomes

$$
\frac{\partial\mathcal{L}_{\mathrm{batch}}}{\partial\mathbf{z}_i} = \frac{1}{B}\left(\mathbf{p}_i - \mathbf{y}_i\right)
$$

This distinction matters when implementing backpropagation.

Dense-layer weight and bias gradients are accumulated across examples. If the incoming gradients have already been scaled by $1/B$, the accumulated parameter gradients must not be divided by $B$ a second time.

The same principle applies to BatchNorm's learnable parameters. However, BatchNorm's input-gradient formula contains its own $1/B$ factor because of the derivative through the batch mean and variance. That factor is part of the mathematical derivative, not a second generic averaging operation.

### 5. Batch Normalisation

Batch normalisation (BatchNorm) normalises activations across a mini-batch for each feature. It then applies learnable scale and shift parameters.

For one feature, let $x_1,x_2,\ldots,x_B$ be the activations of that feature across a batch of $B$ examples.

#### Step 1: Calculate the batch mean

$$
\mu_B = \frac{1}{B}\sum_{i=1}^{B} x_i
$$

The mean represents the average activation for that feature in the current batch.

#### Step 2: Calculate the batch variance

$$
\sigma_B^2 = \frac{1}{B}\sum_{i=1}^{B}\left(x_i-\mu_B\right)^2
$$

The variance measures how far the activations are spread around their mean. This expression uses the population variance convention for the current batch.

#### Step 3: Normalise the activations

$$
\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

Here, $\epsilon$ is a small positive constant that helps prevent division by zero and improves numerical stability.

The normalised values have approximately zero mean and unit variance when the variance is nonzero and $\epsilon$ is small relative to it.

#### Step 4: Apply learnable scale and shift

$$
y_i = \gamma\hat{x}_i + \beta
$$

Here:

- $\gamma$ is the learnable scale parameter.
- $\beta$ is the learnable shift parameter.
- $y_i$ is the output activation.

The scale and shift allow the network to learn an appropriate transformation after normalisation rather than being restricted to the normalised distribution.

#### Backpropagation through BatchNorm

Let

$$
g_i = \frac{\partial\mathcal{L}}{\partial y_i}
$$

be the gradient arriving from the next operation.

The gradient with respect to the shift parameter is

$$
\frac{\partial\mathcal{L}}{\partial\beta} = \sum_{i=1}^{B} g_i
$$

The gradient with respect to the scale parameter is

$$
\frac{\partial\mathcal{L}}{\partial\gamma} = \sum_{i=1}^{B} g_i\hat{x}_i
$$

Both gradients accumulate contributions from every example in the batch.

The input gradient is more involved because each input affects its own normalised value and the shared batch mean and variance.

Define

$$
r = \frac{1}{\sqrt{\sigma_B^2+\epsilon}}
$$

Then the input gradient is

$$
\frac{\partial\mathcal{L}}{\partial x_i} = \frac{\gamma r}{B}\left[B g_i - \sum_{k=1}^{B} g_k - \hat{x}_i\sum_{k=1}^{B} g_k\hat{x}_k\right]
$$

This expression captures the direct path through the input as well as the indirect paths through the batch mean and variance.

Consequently, the gradient for one example depends on other examples in the same batch. BatchNorm's backward pass is not merely an elementwise derivative.

#### Batch size and BatchNorm

When $B=1$, the batch variance is zero. The normalised activation becomes zero under the standard formula, leaving the output determined by the shift parameter. The input gradient through this normalisation operation is also zero for that single example.

This is one reason BatchNorm is generally used with batches containing multiple examples.

### 6. Optimisation and Learning Rates

NeuralEngine currently implements stochastic gradient descent (SGD).

For weights and biases, the update rules are

$$
W \leftarrow W - \eta\frac{\partial\mathcal{L}}{\partial W}
$$

$$
b \leftarrow b - \eta\frac{\partial\mathcal{L}}{\partial b}
$$

where $\eta$ is the learning rate.

The learning rate controls the size of each parameter update. A learning rate that is too large can cause optimisation to overshoot useful parameter values or become unstable. A learning rate that is too small can make training unnecessarily slow.

BatchNorm can make optimisation less sensitive to activation scale and may allow larger learning rates in some settings. It does not guarantee stability or improved accuracy for every model.

The effect should be evaluated experimentally by comparing controlled runs with and without BatchNorm while keeping other training settings consistent.

### 7. Accuracy and Cross-Entropy Loss

Accuracy and cross-entropy loss measure different properties of a model's predictions.

- **Accuracy** measures the fraction of examples classified correctly.
- **Cross-entropy loss** penalises predictions that assign low probability to the correct class.

Accuracy can improve even when mean cross-entropy loss increases. A model may classify more examples correctly while assigning lower probabilities to some correct predictions.

Therefore, neither metric should be interpreted in isolation. Recording both helps distinguish classification performance from the quality of the predicted probability distribution.

### 8. Numerical Gradient Verification

Analytical gradients are checked against central finite differences:

$$
g_{\mathrm{numeric}} \approx \frac{\mathcal{L}(\theta+\varepsilon) - \mathcal{L}(\theta-\varepsilon)}{2\varepsilon}
$$

Here, $\theta$ represents a parameter being checked and $\varepsilon$ is a small perturbation.

Comparing numerical and analytical gradients helps identify errors in the backward-pass implementation. Gradient checks are valuable validation tools, although they do not prove that every possible execution path is correct.

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

The tree above reflects the documented project layout. If your current repository has additional files, such as a dedicated BatchNorm module, update this tree to match the actual files.

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

This command assumes the executable target is named `basic_example` and that the example's dataset paths are compatible with the working directory. Check `CMakeLists.txt` and `examples/basic.cpp` if the executable or dataset cannot be found.

## Programmatic Usage

The following example illustrates the general training pipeline using the documented single-example API. It creates a small network, performs a forward pass, calculates the loss, propagates gradients backwards, and updates the parameters.

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

    const neural::Tensor hidden =
        relu.forward(fc1.forward(input));

    const neural::Tensor logits =
        fc2.forward(hidden);

    const auto result =
        neural::softmax_cross_entropy(logits, target);

    const neural::Tensor hidden_gradient =
        fc2.backward(result.gradient);

    const neural::Tensor input_gradient =
        fc1.backward(relu.backward(hidden_gradient));

    optimizer.step(fc1);
    optimizer.step(fc2);

    (void)input_gradient;
    return 0;
}
```

This is a minimal illustration of the single-example pipeline, not the complete MNIST training configuration. Check the actual headers if your current API differs. The mini-batch and BatchNorm APIs should be documented with examples matching their actual function signatures before being presented as copy-and-paste usage.

## Design Goals

NeuralEngine is intended to make the internal mechanics of neural networks explicit.

- **Understandability:** Core algorithms are implemented directly rather than delegated to a high-level framework.
- **Mathematical transparency:** Forward propagation, loss calculation, and backpropagation correspond to explicit mathematical operations.
- **Modularity:** Tensor operations, neural-network layers, data loading, and optimisation are separated into distinct components.
- **Extensibility:** The architecture provides a foundation for adding more operations, layers, and optimisation methods.
- **Testing:** Numerical gradient checks help validate the correctness of analytical derivatives.

## Future Work

Potential directions for development include:

- [ ] Adam and other optimisation algorithms.
- [ ] Additional activation functions and loss functions.
- [ ] More extensive batch-training experiments.
- [ ] Convolutional layers and other neural-network architectures.
- [ ] Improved tensor operations and performance profiling.
- [ ] More extensive gradient checks and automated tests.
- [ ] Expanded visualisation and experimentation tools in NeuralEngine Studio.
- [ ] Further validation of model checkpoint saving and restoration.

## Acknowledgements

NeuralEngine was developed as a learning project to explore neural-network mathematics, C++ systems programming, and the implementation of machine-learning algorithms from scratch.

The MNIST dataset is used for handwritten-digit classification.

## License

This project is distributed under the terms of the MIT License if the repository's `LICENSE` file specifies that licence.