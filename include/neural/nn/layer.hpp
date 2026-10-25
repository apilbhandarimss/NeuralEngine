
#pragma once
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

class Dense {
public:
    Dense(
        std::size_t input_features,
        std::size_t output_features,
        WeightInit init = WeightInit::He
    );

    // Deterministic initialisation for reproducible tests and experiments.
    Dense(
        std::size_t input_features,
        std::size_t output_features,
        WeightInit init,
        std::uint32_t seed
    );

    [[nodiscard]] Tensor forward(const Tensor& input) const;

    // Receives dC/d(output), returns dC/d(input).
    [[nodiscard]] Tensor backward(const Tensor& output_gradient);

    // Read-only parameter access for inference and inspection.
    [[nodiscard]] const Tensor& weights() const noexcept;
    [[nodiscard]] const Tensor& bias() const noexcept;

    // Mutable parameter access for optimisers and parameter loading.
    [[nodiscard]] Tensor& weights() noexcept;
    [[nodiscard]] Tensor& bias() noexcept;

    // Read-only access to the latest parameter gradients.
    [[nodiscard]] const Tensor& weight_gradients() const noexcept;
    [[nodiscard]] const Tensor& bias_gradients() const noexcept;

    [[nodiscard]] std::size_t input_features() const noexcept;
    [[nodiscard]] std::size_t output_features() const noexcept;

private:
    Tensor weights_;
    Tensor bias_;

    Tensor weight_gradients_;
    Tensor bias_gradients_;

    // Input from the most recent forward pass.
    std::optional<Tensor> cached_input_;

    void initialise(WeightInit init, std::mt19937& rng);
};

}