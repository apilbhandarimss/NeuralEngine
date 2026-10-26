# NeuralEngine

A minimalist, dependency-free neural network engine and interactive laboratory written from scratch in **C++20**. 

NeuralEngine explores the concrete mechanics of deep learning: contiguous tensor memory layouts, manual reverse-mode automatic differentiation (backpropagation), numerically stable log-sum-exp loss objectives, binary IDX archive parsing, and analytical gradient verification via central finite differences.

---

## Key Features

- **Contiguous Tensors**: Arbitrary $N$-dimensional tensors stored in contiguous heap buffers with row-major strides and zero-overhead non-owning `std::span` views.
- **Dense Affine Layers**: Forward and backward passes implemented directly via cache-friendly matrix transformations ($\mathbf{z} = W\mathbf{a} + \mathbf{b}$).
- **Nonlinear Activations**: Elementwise ReLU (with subgradient at zero), Sigmoid, and Hyperbolic Tangent ($\tanh$).
- **Stable Categorical Objective**: Combined Softmax Cross-Entropy loss computed via maximum-shifted Log-Sum-Exp arithmetic:
  $$\mathcal{L} = m + \ln\left(\sum_{j} e^{z_j - m}\right) - z_y, \qquad m = \max_j z_j$$
- **Exact Analytical Gradient**: Simplified output-layer error signal derived from the Softmax Jacobian:
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}$$
- **Weight Initializers**: He (Kaiming) normal initialization for ReLU layers and Xavier (Glorot) uniform initialization for output projections.
- **Pure C++ Dataset Pipeline**: Zero-dependency binary parser for standard big-endian MNIST IDX image and label files.
- **Numerical Gradient Checks**: Built-in verification comparing analytical backward gradients against central finite differences:
  $$g_{\text{numeric}} \approx \frac{\mathcal{L}(\theta + \varepsilon) - \mathcal{L}(\theta - \varepsilon)}{2\varepsilon}$$
- **NeuralEngine Studio**: A hardware-accelerated desktop GUI written with Raylib featuring real-time drawing inference, an IDX dataset visualizer, live loss plotting, and asynchronous training.

---

## Empirical Benchmark (MNIST)

Trained on the standard MNIST dataset (60,000 training samples, 10,000 test samples) using per-example Stochastic Gradient Descent ($\text{batch size} = 1$):

$$\mathbf{784} \xrightarrow{W_1} \mathbf{512} \xrightarrow{\text{ReLU}} \mathbf{256} \xrightarrow{\text{ReLU}} \mathbf{128} \xrightarrow{\text{ReLU}} \mathbf{64} \xrightarrow{\text{ReLU}} \mathbf{10} \ (\text{Logits})$$

| Metric | 1 Epoch Run | 5 Epoch Run |
|---|---|---|
| **Test Accuracy** | **94.61%** | **97.81%** |
| **Correct Predictions** | 9,461 / 10,000 | 9,781 / 10,000 |
| **Mean Test Loss** | 0.178783 | ~0.082100 |
| **Optimization** | SGD ($\eta = 0.01$) | SGD ($\eta = 0.01$) |
| **Trainable Parameters** | 575,050 | 575,050 |

### Parameter Breakdown

$$N_{\text{total}} = \sum_{l=1}^{L} \left(n_{\text{in}}^{(l)} \cdot n_{\text{out}}^{(l)} + n_{\text{out}}^{(l)}\right)$$

- **Layer 1** ($784 \to 512$): $784 \times 512 + 512 = 401,920$
- **Layer 2** ($512 \to 256$): $512 \times 256 + 256 = 131,328$
- **Layer 3** ($256 \to 128$): $256 \times 128 + 128 = 32,896$
- **Layer 4** ($128 \to 64$): $128 \times 64 + 64 = 8,256$
- **Layer 5** ($64 \to 10$): $64 \times 10 + 10 = 650$
- **Total Weights & Biases**: **575,050 parameters** (~2.3 MB in 32-bit float)

---

## Mathematical Foundations

### 1. Forward Pass
For layer $l$ taking activations $\mathbf{a}^{(l-1)}$:
$$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = \operatorname{ReLU}\left(\mathbf{z}^{(l)}\right) = \max\left(\mathbf{0}, \mathbf{z}^{(l)}\right)$$

### 2. Dense Layer Backpropagation
Given the upstream error vector arriving at layer $l$:
$$\mathbf{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

The gradients with respect to layer parameters and inputs are:
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \mathbf{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T \quad (\text{outer product})$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \mathbf{\delta}^{(l)}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(l-1)}} = (W^{(l)})^T \mathbf{\delta}^{(l)} \quad (\text{propagated upstream})$$

### 3. Hidden Layer Cascade
To cross the ReLU activation between dense operations:
$$\mathbf{\delta}^{(l-1)} = \left((W^{(l)})^T \mathbf{\delta}^{(l)}\right) \odot \mathbf{1}_{\mathbf{z}^{(l-1)} > 0}$$

---

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   └── neural/
│       ├── data/
│       │   └── dataset.hpp          # MNIST IDX binary reader
│       ├── nn/
│       │   ├── layer.hpp            # Dense layer interface
│       │   └── model.hpp            # Network container
│       ├── ops/
│       │   ├── activations.hpp      # ReLU, Sigmoid, Tanh
│       │   ├── arithmetic.hpp       # Contiguous matrix math
│       │   └── loss.hpp             # Fused Softmax Cross-Entropy
│       ├── optim/
│       │   └── optimizer.hpp        # SGD implementation
│       └── tensor/
│           └── tensor.hpp           # N-D contiguous tensor
├── src/
│   ├── data/dataset.cpp
│   ├── nn/layer.cpp
│   ├── ops/activations.cpp
│   ├── ops/loss.cpp
│   ├── optim/optimizer.cpp
│   └── tensor/tensor.cpp
├── examples/
│   ├── basic.cpp                    # Minimal headless training example
│   └── studio.cpp                   # Raylib graphical interface
├── tests/
│   ├── test_main.cpp                # Core unit test runner
│   └── test_gradients.cpp           # Finite-difference gradient checks
└── data/
    └── mnist/                       # Local MNIST binary IDX dataset
```

---

## Building and Testing

### Prerequisites
- A **C++20** compatible compiler (GCC 11+, Clang 13+, or MSVC 2019+)
- **CMake** 3.20 or newer
- *(Optional)* **Raylib 4.5+** (Required only for building `neural_studio`)

### 1. Clone & Download Dataset
Download the standard uncompressed MNIST files into `data/mnist/`:
```bash
mkdir -p data/mnist
cd data/mnist

# Download training & test image/label archives
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# Extract archives to obtain raw IDX binaries
gunzip *.gz
cd ../..
```

### 2. Configure and Compile
```bash
# Configure release build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Compile all targets in parallel
cmake --build build --parallel
```

### 3. Run Test Suite
Verify analytical gradients, tensor indexing, and serialization:
```bash
ctest --test-dir build --output-on-failure
```

---

## Quickstart Usage

### Headless CLI Example
Train and evaluate a 5-layer network from the command line:
```bash
./build/basic_example
```

### Programmatic C++20 API
```cpp
#include "neural/data/dataset.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

int main() {
    // 1. Initialize network layers
    neural::Dense fc1(784, 512, neural::WeightInit::He, /*seed=*/101);
    neural::ReLU relu1;
    neural::Dense fc2(512, 10, neural::WeightInit::Xavier, /*seed=*/102);

    neural::SGD optimizer(/*learning_rate=*/0.01f);

    // 2. Load dataset
    neural::MNISTDataset train_data("data/mnist/train-images-idx3-ubyte",
                                    "data/mnist/train-labels-idx1-ubyte");

    // 3. One forward + backward SGD step on sample 0
    neural::Tensor x = train_data.image(0);
    size_t target_label = train_data.label(0);

    // Forward
    neural::Tensor h = relu1.forward(fc1.forward(x));
    neural::Tensor logits = fc2.forward(h);

    // Loss + Logits Gradient
    neural::LossResult result = neural::softmax_cross_entropy(logits, target_label);

    // Backward pass
    neural::Tensor g = fc2.backward(result.logits_gradient);
    fc1.backward(relu1.backward(g));

    // Update weights
    optimizer.step(fc1);
    optimizer.step(fc2);

    return 0;
}
```

---

## NeuralEngine Studio

`neural_studio` is an immediate-mode desktop application built with Raylib to inspect, train, and test models interactively.

Launch the GUI:
```bash
./build/neural_studio
```

Launch headless verification checks (no window required):
```bash
./build/neural_studio --self-test
```

### Studio Capabilities
1. **Model Browser**: Scans workspace roots, validates model topologies, and loads saved parameters.
2. **Drawing Canvas**: A 280×280 input canvas that matches the MNIST preprocessing pipeline (bounding box isolation, aspect-ratio scaling to 20px, and center-of-mass centering) for real-time digit recognition.
3. **Data Explorer**: Visual inspection of arbitrary training and test IDX records alongside live confusion matrix heatmaps.
4. **Asynchronous Trainer**: Background thread training using decoupled model clones with live loss graphing, ETA calculation, and early-stop checkpoint preservation.

---

## Checkpoint Format

Models are serialized to human-readable plain text (`.txt`) with dimension and metadata headers:

```text
NEURAL_ENGINE_CHECKPOINT 1
ARCHITECTURE 784 512 256 128 64 10
LAYER fc1
WEIGHTS
2 512 784
0.0124 -0.0412 0.0089 ...
BIAS
1 512
0.0001 -0.0002 0.0000 ...
LAYER fc2
...
```

The deserializer performs validation against magic headers, layer rank, dimension bounds, unexpected EOFs, and non-finite (`NaN`/`Inf`) floating-point values.

---

## Roadmap

- [ ] Mini-batch matrix-matrix multiplication ($B \times N$).
- [ ] SIMD / AVX2 vectorization for matrix multiplication kernels.
- [ ] Momentum and Adam optimizers.
- [ ] 2D Convolution (`Conv2D`) and Max Pooling layers.

---
