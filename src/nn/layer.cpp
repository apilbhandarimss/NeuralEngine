// compile with: -O3 -march=native -fopenmp
#include "neural/nn/layer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural {

namespace {

constexpr int kThreads = 14;
constexpr std::size_t kParallelThreshold = 32768;
constexpr std::size_t kTile = 256;

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
    const std::size_t inputs  = input_features();
    const std::size_t outputs = output_features();

    const bool single = input.rank() == 1;
    if (!single && input.rank() != 2) {
        throw std::invalid_argument(
            "Dense::forward expects rank-1 or rank-2 input"
        );
    }
    if (input.shape().back() != inputs) {
        throw std::invalid_argument(
            "Dense::forward input feature count mismatch"
        );
    }

    const std::size_t batch = single ? 1 : input.shape()[0];
    if (batch == 0) {
        throw std::invalid_argument(
            "Dense::forward batch size must be greater than zero"
        );
    }

    Tensor output(
        single
            ? std::vector<std::size_t>{outputs}
            : std::vector<std::size_t>{batch, outputs},
        0.0F
    );

    const float* __restrict x    = input.data().data();
    const float* __restrict w    = weights_.data().data();
    const float* __restrict bias = bias_.data().data();
    float* __restrict y          = output.data().data();

    // Each (b, o) cell is written by exactly one thread -> no data races.
    // A rank-1 input is just batch == 1, so no branching in the hot loop.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(kThreads) \
        if (batch * outputs * inputs > kParallelThreshold)
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t o = 0; o < outputs; ++o) {
            const float* __restrict xr = x + b * inputs;
            const float* __restrict wr = w + o * inputs;

            float sum = 0.0F;
            #pragma omp simd reduction(+:sum)
            for (std::size_t i = 0; i < inputs; ++i) {
                sum += wr[i] * xr[i];
            }
            y[b * outputs + o] = sum + bias[o];
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
    const std::size_t inputs  = input_features();
    const std::size_t outputs = output_features();
    const std::size_t batch   = single ? 1 : input.shape()[0];

    if (output_gradient.rank() != input.rank() ||
        output_gradient.shape().back() != outputs ||
        (!single && output_gradient.shape()[0] != batch)) {
        throw std::invalid_argument(
            "Dense::backward output gradient shape mismatch"
        );
    }

    Tensor input_gradient(
        single
            ? std::vector<std::size_t>{inputs}
            : std::vector<std::size_t>{batch, inputs},
        0.0F
    );

    const float* __restrict x  = input.data().data();
    const float* __restrict w  = weights_.data().data();
    const float* __restrict dy = output_gradient.data().data();
    float* __restrict dw       = weight_gradients_.data().data();
    float* __restrict db       = bias_gradients_.data().data();
    float* __restrict dx       = input_gradient.data().data();

    const bool par = batch * outputs * inputs > kParallelThreshold;

    // Pass 1: dW and db.
    // Parallel over `o`; each thread owns row dw[o, :] and db[o].
    // Values are assigned (not accumulated across calls), so no separate
    // zeroing pass is needed.
    #pragma omp parallel for schedule(static) num_threads(kThreads) if (par)
    for (std::size_t o = 0; o < outputs; ++o) {
        float* __restrict dwr = dw + o * inputs;

        #pragma omp simd
        for (std::size_t i = 0; i < inputs; ++i) {
            dwr[i] = 0.0F;
        }

        float bsum = 0.0F;
        for (std::size_t b = 0; b < batch; ++b) {
            const float g = dy[b * outputs + o];
            const float* __restrict xr = x + b * inputs;
            bsum += g;

            #pragma omp simd
            for (std::size_t i = 0; i < inputs; ++i) {
                dwr[i] += g * xr[i];
            }
        }
        db[o] = bsum;
    }

    // Pass 2: dX.
    // Parallel over (batch, input tile); each thread owns its dx slice,
    // so there are no races. dx starts zeroed from the Tensor constructor.
    const std::size_t tiles = (inputs + kTile - 1) / kTile;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(kThreads) if (par)
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t t = 0; t < tiles; ++t) {
            const std::size_t i0 = t * kTile;
            const std::size_t i1 = (i0 + kTile < inputs) ? i0 + kTile : inputs;
            float* __restrict dxr = dx + b * inputs;

            for (std::size_t o = 0; o < outputs; ++o) {
                const float g = dy[b * outputs + o];
                const float* __restrict wr = w + o * inputs;

                #pragma omp simd
                for (std::size_t i = i0; i < i1; ++i) {
                    dxr[i] += wr[i] * g;
                }
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