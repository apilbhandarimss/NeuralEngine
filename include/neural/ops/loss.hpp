
#pragma once

#include "neural/tensor/tensor.hpp"

#include <cstddef>

namespace neural {

struct LossResult {
    float loss;
    Tensor logits_gradient;
    Tensor probabilities;
};

[[nodiscard]]
LossResult softmax_cross_entropy(
    const Tensor& logits,
    std::size_t target
);

[[nodiscard]]
LossResult softmax_cross_entropy(const Tensor& logits, const Tensor& targets);

} // namespace neural
