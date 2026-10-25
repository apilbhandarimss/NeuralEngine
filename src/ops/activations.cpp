
#include "neural/ops/activations.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace neural {

Tensor ReLU::forward(const Tensor& input) const {
    Tensor result(input.shape(), 0.0F);

    const auto values = input.data();
    auto output = result.data();

    for (std::size_t i = 0; i < input.size(); ++i) {
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

    const auto values = output.data();
    const auto gradient = output_gradient.data();
    auto result = input_gradient.data();

    for (std::size_t i = 0; i < output.size(); ++i) {
        result[i] = values[i] > 0.0F ? gradient[i] : 0.0F;
    }

    cached_output_.reset();
    return input_gradient;
}

Tensor Sigmoid::forward(const Tensor& input) const {
    Tensor result(input.shape(), 0.0F);

    const auto values = input.data();
    auto output = result.data();

    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = values[i];

        // Numerically stable sigmoid.
        if (x >= 0.0F) {
            const float e = std::exp(-x);
            output[i] = 1.0F / (1.0F + e);
        } else {
            const float e = std::exp(x);
            output[i] = e / (1.0F + e);
        }
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

    const auto values = output.data();
    const auto gradient = output_gradient.data();
    auto result = input_gradient.data();

    for (std::size_t i = 0; i < output.size(); ++i) {
        result[i] = gradient[i] * values[i] * (1.0F - values[i]);
    }

    cached_output_.reset();
    return input_gradient;
}

Tensor Tanh::forward(const Tensor& input) const {
    Tensor result(input.shape(), 0.0F);

    const auto values = input.data();
    auto output = result.data();

    for (std::size_t i = 0; i < input.size(); ++i) {
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

    const auto values = output.data();
    const auto gradient = output_gradient.data();
    auto result = input_gradient.data();

    for (std::size_t i = 0; i < output.size(); ++i) {
        result[i] = gradient[i] * (1.0F - values[i] * values[i]);
    }

    cached_output_.reset();
    return input_gradient;
}

}
