
#include "neural/nn/layer.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
                    2.0F / static_cast<float>(inputs + outputs)
                );
            break;

        default:
            throw std::invalid_argument(
                "Unsupported weight initialisation strategy"
            );
    }

    std::normal_distribution<float> distribution(
        0.0F, standard_deviation
    );

    for (float& value : weights_.data()) {
        value = distribution(rng);
    }
}

Tensor Dense::forward(const Tensor& input, bool /*training*/) {
    const std::size_t inputs = input_features();
    const std::size_t outputs = output_features();

    const bool single = input.rank() == 1;
    const bool batched = input.rank() == 2;

    if (!single && !batched) {
        throw std::invalid_argument(
            "Dense::forward expects rank-1 or rank-2 input"
        );
    }

    if (input.shape().back() != inputs) {
        throw std::invalid_argument(
            "Dense::forward input feature count mismatch"
        );
    }

    const std::size_t batch_size =
        single ? 1 : input.shape()[0];

    if (batch_size == 0) {
        throw std::invalid_argument(
            "Dense::forward batch size must be greater than zero"
        );
    }

    Tensor output(
        single
            ? std::vector<std::size_t>{outputs}
            : std::vector<std::size_t>{batch_size, outputs},
        0.0F
    );

    const auto x = input.data();
    const auto w = weights_.data();
    const auto bias_values = bias_.data();
    auto y = output.data();

    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t out = 0; out < outputs; ++out) {
            float sum = bias_values[out];

            for (std::size_t in = 0; in < inputs; ++in) {
                const std::size_t xi =
                    single ? in : b * inputs + in;

                sum += w[out * inputs + in] * x[xi];
            }

            const std::size_t yi =
                single ? out : b * outputs + out;

            y[yi] = sum;
        }
    }

    cached_input_ = input;
    return output;
}

Tensor Dense::backward(const Tensor& output_gradient) {
    if (!cached_input_) {
        throw std::logic_error(
            "Dense::backward called before forward"
        );
    }

    const Tensor& input = *cached_input_;

    const bool single = input.rank() == 1;
    const std::size_t inputs = input_features();
    const std::size_t outputs = output_features();

    const std::size_t batch_size =
        single ? 1 : input.shape()[0];

    if (output_gradient.rank() != input.rank() ||
        output_gradient.shape().back() != outputs ||
        (!single &&
         output_gradient.shape()[0] != batch_size)) {
        throw std::invalid_argument(
            "Dense::backward output gradient shape mismatch"
        );
    }

    const auto x = input.data();
    const auto w = weights_.data();
    const auto dy = output_gradient.data();

    auto dw = weight_gradients_.data();
    auto db = bias_gradients_.data();

    for (float& value : dw) {
        value = 0.0F;
    }

    for (float& value : db) {
        value = 0.0F;
    }

    Tensor input_gradient(
        single
            ? std::vector<std::size_t>{inputs}
            : std::vector<std::size_t>{batch_size, inputs},
        0.0F
    );

    auto dx = input_gradient.data();

    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t out = 0; out < outputs; ++out) {
            const std::size_t dyi =
                single ? out : b * outputs + out;

            const float gradient = dy[dyi];

            db[out] += gradient;

            for (std::size_t in = 0; in < inputs; ++in) {
                const std::size_t xi =
                    single ? in : b * inputs + in;

                const std::size_t wi = out * inputs + in;

                dw[wi] += gradient * x[xi];
                dx[xi] += w[wi] * gradient;
            }
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

} // namespace neural
