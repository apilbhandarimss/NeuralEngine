#include "neural/nn/layer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
namespace neural {

namespace {

std::size_t checked_feature_count(std::size_t count, const char* name) {
    if (count == 0) {
        throw std::invalid_argument(
            std::string(name) + " must be greater than zero"
        );
    }

    return count;
}

std::mt19937 make_random_engine() {
    std::random_device device;
    return std::mt19937(device());
}

} // namespace

Dense::Dense(
    std::size_t input_features,
    std::size_t output_features,
    WeightInit init
)
    : weights_({
          checked_feature_count(output_features, "output_features"),
          checked_feature_count(input_features, "input_features")
      }, 0.0F),
      bias_({output_features}, 0.0F),
      weight_gradients_(weights_.shape(), 0.0F),
      bias_gradients_(bias_.shape(), 0.0F) {
    auto rng = make_random_engine();
    initialise(init, rng);
}

Dense::Dense(
    std::size_t input_features,
    std::size_t output_features,
    WeightInit init,
    std::uint32_t seed
)
    : weights_({
          checked_feature_count(output_features, "output_features"),
          checked_feature_count(input_features, "input_features")
      }, 0.0F),
      bias_({output_features}, 0.0F),
      weight_gradients_(weights_.shape(), 0.0F),
      bias_gradients_(bias_.shape(), 0.0F) {
    std::mt19937 rng(seed);
    initialise(init, rng);
}

void Dense::initialise(WeightInit init, std::mt19937& rng) {
    const std::size_t inputs = input_features();
    const std::size_t outputs = output_features();

    float standard_deviation = 0.0F;

    switch (init) {
        case WeightInit::He:
            standard_deviation =
                std::sqrt(2.0F / static_cast<float>(inputs));
            break;

        case WeightInit::Xavier:
            standard_deviation =
                std::sqrt(
                    2.0F /
                    static_cast<float>(inputs + outputs)
                );
            break;

        default:
            throw std::invalid_argument(
                "Unsupported weight initialisation strategy"
            );
    }

    std::normal_distribution<float> distribution(
        0.0F,
        standard_deviation
    );

    auto values = weights_.data();

    for (float& value : values) {
        value = distribution(rng);
    }

    // Biases and gradients remain zero-initialised.
}

Tensor Dense::forward(const Tensor& input) const {
    if (input.rank() != 1) {
        throw std::invalid_argument(
            "Dense::forward expects a rank-1 input tensor"
        );
    }

    if (input.shape()[0] != input_features()) {
        throw std::invalid_argument(
            "Dense::forward input feature count mismatch"
        );
    }

    Tensor output({output_features()}, 0.0F);

    const auto input_values = input.data();
    const auto weight_values = weights_.data();
    const auto bias_values = bias_.data();
    auto output_values = output.data();

    const std::size_t inputs = input_features();
    const std::size_t outputs = output_features();

    for (std::size_t out = 0; out < outputs; ++out) {
        float sum = bias_values[out];

        for (std::size_t in = 0; in < inputs; ++in) {
            sum += weight_values[out * inputs + in]
                 * input_values[in];
        }

        output_values[out] = sum;
    }

    // forward() is const, so the cache is updated through const_cast.
    const_cast<Dense*>(this)->cached_input_ = input;

    return output;
}

Tensor Dense::backward(const Tensor& output_gradient) {
    if (!cached_input_) {
        throw std::logic_error(
            "Dense::backward called before forward"
        );
    }

    if (output_gradient.rank() != 1 ||
        output_gradient.shape()[0] != output_features()) {
        throw std::invalid_argument(
            "Dense::backward output gradient shape mismatch"
        );
    }

    const Tensor& input = *cached_input_;

    const auto input_values = input.data();
    const auto weight_values = weights_.data();
    const auto gradient_values = output_gradient.data();

    auto weight_gradient_values = weight_gradients_.data();
    auto bias_gradient_values = bias_gradients_.data();

    const std::size_t inputs = input_features();
    const std::size_t outputs = output_features();

    // This implementation stores gradients for one example.
    for (float& value : weight_gradient_values) {
        value = 0.0F;
    }

    for (float& value : bias_gradient_values) {
        value = 0.0F;
    }

    Tensor input_gradient({inputs}, 0.0F);
    auto input_gradient_values = input_gradient.data();

    for (std::size_t out = 0; out < outputs; ++out) {
        const float gradient = gradient_values[out];

        // dC/db[out] = dC/dy[out]
        bias_gradient_values[out] = gradient;

        for (std::size_t in = 0; in < inputs; ++in) {
            const std::size_t weight_index = out * inputs + in;

            // dC/dW[out,in] = dC/dy[out] * input[in]
            weight_gradient_values[weight_index] =
                gradient * input_values[in];

            // dC/dx[in] += W[out,in] * dC/dy[out]
            input_gradient_values[in] +=
                weight_values[weight_index] * gradient;
        }
    }

    cached_input_.reset();

    return input_gradient;
}

const Tensor& Dense::weights() const noexcept {
    return weights_;
}

const Tensor& Dense::bias() const noexcept {
    return bias_;
}

Tensor& Dense::weights() noexcept {
    return weights_;
}

Tensor& Dense::bias() noexcept {
    return bias_;
}

const Tensor& Dense::weight_gradients() const noexcept {
    return weight_gradients_;
}

const Tensor& Dense::bias_gradients() const noexcept {
    return bias_gradients_;
}

std::size_t Dense::input_features() const noexcept {
    return weights_.shape()[1];
}

std::size_t Dense::output_features() const noexcept {
    return weights_.shape()[0];
}

} 