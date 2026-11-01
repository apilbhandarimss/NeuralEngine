// compile with: -O3 -march=native -fopenmp -fno-math-errno
#include "neural/ops/activations.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace neural {

namespace {

constexpr int kThreads = 14;
// Cheap, memory-bound loops (ReLU, all backward passes).
constexpr std::size_t kParallelMinCheap = 32768;
// Loops dominated by exp/tanh are worth threading sooner.
constexpr std::size_t kParallelMinMath = 4096;

} // namespace

// ---------------------------------------------------------------- ReLU

Tensor ReLU::forward(const Tensor& input) const {
    Tensor result(input.shape(), 0.0F);

    const std::size_t n = input.size();
    const float* __restrict values = input.data().data();
    float* __restrict output = result.data().data();

    // Each iteration touches only index i -> no races.
    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMinCheap)
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = values[i] > 0.0F ? values[i] : 0.0F;
    }

    const_cast<ReLU*>(this)->cached_output_ = result;
    return result;
}

Tensor ReLU::backward(const Tensor& output_gradient) {
    if (!cached_output_) {
        throw std::logic_error("ReLU::backward called before forward");
    }

    const Tensor& output = *cached_output_;

    if (output_gradient.shape() != output.shape()) {
        throw std::invalid_argument("ReLU gradient shape mismatch");
    }

    Tensor input_gradient(output.shape(), 0.0F);

    const std::size_t n = output.size();
    const float* __restrict values = output.data().data();
    const float* __restrict gradient = output_gradient.data().data();
    float* __restrict result = input_gradient.data().data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMinCheap)
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = values[i] > 0.0F ? gradient[i] : 0.0F;
    }

    cached_output_.reset();
    return input_gradient;
}

// ------------------------------------------------------------- Sigmoid

Tensor Sigmoid::forward(const Tensor& input) const {
    Tensor result(input.shape(), 0.0F);

    const std::size_t n = input.size();
    const float* __restrict values = input.data().data();
    float* __restrict output = result.data().data();

    // Numerically stable sigmoid, written branch-free so it vectorizes.
    // With e = exp(-|x|):
    //   x >= 0 -> 1 / (1 + e)
    //   x <  0 -> e / (1 + e)
    // which is exactly the two original branches.
    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMinMath)
    for (std::size_t i = 0; i < n; ++i) {
        const float x = values[i];
        const float e = std::exp(-std::fabs(x));
        const float numerator = x >= 0.0F ? 1.0F : e;

        output[i] = numerator / (1.0F + e);
    }

    const_cast<Sigmoid*>(this)->cached_output_ = result;
    return result;
}

Tensor Sigmoid::backward(const Tensor& output_gradient) {
    if (!cached_output_) {
        throw std::logic_error("Sigmoid::backward called before forward");
    }

    const Tensor& output = *cached_output_;

    if (output_gradient.shape() != output.shape()) {
        throw std::invalid_argument("Sigmoid gradient shape mismatch");
    }

    Tensor input_gradient(output.shape(), 0.0F);

    const std::size_t n = output.size();
    const float* __restrict values = output.data().data();
    const float* __restrict gradient = output_gradient.data().data();
    float* __restrict result = input_gradient.data().data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMinCheap)
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = gradient[i] * values[i] * (1.0F - values[i]);
    }

    cached_output_.reset();
    return input_gradient;
}

// ---------------------------------------------------------------- Tanh

Tensor Tanh::forward(const Tensor& input) const {
    Tensor result(input.shape(), 0.0F);

    const std::size_t n = input.size();
    const float* __restrict values = input.data().data();
    float* __restrict output = result.data().data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMinMath)
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = std::tanh(values[i]);
    }

    const_cast<Tanh*>(this)->cached_output_ = result;
    return result;
}

Tensor Tanh::backward(const Tensor& output_gradient) {
    if (!cached_output_) {
        throw std::logic_error("Tanh::backward called before forward");
    }

    const Tensor& output = *cached_output_;

    if (output_gradient.shape() != output.shape()) {
        throw std::invalid_argument("Tanh gradient shape mismatch");
    }

    Tensor input_gradient(output.shape(), 0.0F);

    const std::size_t n = output.size();
    const float* __restrict values = output.data().data();
    const float* __restrict gradient = output_gradient.data().data();
    float* __restrict result = input_gradient.data().data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMinCheap)
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = gradient[i] * (1.0F - values[i] * values[i]);
    }

    cached_output_.reset();
    return input_gradient;
}

} // namespace neural