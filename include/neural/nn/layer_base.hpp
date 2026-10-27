#pragma once

#include "neural/tensor/tensor.hpp"

namespace neural {

class Layer {
public:
    virtual ~Layer() = default;

    [[nodiscard]] virtual Tensor forward(
        const Tensor& input,
        bool training = true
    ) = 0;

    [[nodiscard]] virtual Tensor backward(
        const Tensor& output_gradient
    ) = 0;
};

} // namespace neural