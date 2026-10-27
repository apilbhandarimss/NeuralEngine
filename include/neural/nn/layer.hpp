
#pragma once

#include "neural/nn/layer_base.hpp"
#include "neural/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

namespace neural {

enum class WeightInit {
    He,
    Xavier
};

class Dense : public Layer {
public:
    Dense(
        std::size_t input_features,
        std::size_t output_features,
        WeightInit init = WeightInit::He
    );

    Dense(
        std::size_t input_features,
        std::size_t output_features,
        WeightInit init,
        std::uint32_t seed
    );

    [[nodiscard]] Tensor forward(
        const Tensor& input,
        bool training = true
    ) override;

    [[nodiscard]] Tensor backward(
        const Tensor& output_gradient
    ) override;

    [[nodiscard]] const Tensor& weights() const noexcept;
    [[nodiscard]] const Tensor& bias() const noexcept;

    [[nodiscard]] Tensor& weights() noexcept;
    [[nodiscard]] Tensor& bias() noexcept;

    [[nodiscard]] const Tensor& weight_gradients() const noexcept;
    [[nodiscard]] const Tensor& bias_gradients() const noexcept;

    [[nodiscard]] std::size_t input_features() const noexcept;
    [[nodiscard]] std::size_t output_features() const noexcept;

private:
    Tensor weights_;
    Tensor bias_;
    Tensor weight_gradients_;
    Tensor bias_gradients_;

    std::optional<Tensor> cached_input_;

    void initialise(WeightInit init, std::mt19937& rng);
};

} // namespace neural
