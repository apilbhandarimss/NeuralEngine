#pragma once

#include "neural/tensor/tensor.hpp"

namespace neural {

class Operation {
public:
    virtual ~Operation() = default;

    [[nodiscard]]
    virtual Tensor forward(const Tensor& input) const = 0;

    // Receives dC/d(output), returns dC/d(input).
    [[nodiscard]]
    virtual Tensor backward(const Tensor& output_gradient) = 0;
};

}

