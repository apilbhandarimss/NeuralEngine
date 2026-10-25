
#pragma once

#include "neural/tensor/tensor.hpp"

namespace neural {

// Element-wise arithmetic. Binary operations require matching shapes.
[[nodiscard]] Tensor operator+(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator-(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator*(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator/(const Tensor& lhs, const Tensor& rhs);

// Element-wise negation.
[[nodiscard]] Tensor operator-(const Tensor& input);

// Matrix multiplication and batched tensor multiplication.
// Supports input ranks 1 through 5, with broadcasting of batch dimensions.
// A vector-vector product returns a one-element tensor.
[[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs);

} // namespace neural
