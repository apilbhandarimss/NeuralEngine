#include "neural/ops/arithmetic.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace neural {
namespace {

std::size_t element_count(
    const std::vector<std::size_t>& shape
) {
    std::size_t count = 1;

    for (const std::size_t dimension : shape) {
        count *= dimension;
    }

    return count;
}

std::vector<std::size_t> strides(
    const std::vector<std::size_t>& shape
) {
    std::vector<std::size_t> result(shape.size(), 1);

    for (std::size_t i = shape.size(); i > 1; --i) {
        result[i - 2] = result[i - 1] * shape[i - 1];
    }

    return result;
}

std::vector<std::size_t> broadcast_shape(
    const std::vector<std::size_t>& lhs,
    const std::vector<std::size_t>& rhs
) {
    const std::size_t rank =
        lhs.size() > rhs.size() ? lhs.size() : rhs.size();

    std::vector<std::size_t> result(rank, 1);

    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t lhs_offset = rank - lhs.size();
        const std::size_t rhs_offset = rank - rhs.size();

        const std::size_t lhs_dim =
            i < lhs_offset ? 1 : lhs[i - lhs_offset];

        const std::size_t rhs_dim =
            i < rhs_offset ? 1 : rhs[i - rhs_offset];

        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw std::invalid_argument(
                "Incompatible tensor shapes for broadcasting"
            );
        }

        result[i] = lhs_dim > rhs_dim ? lhs_dim : rhs_dim;
    }

    return result;
}

std::size_t broadcast_offset(
    const std::vector<std::size_t>& input_shape,
    const std::vector<std::size_t>& input_strides,
    const std::vector<std::size_t>& output_shape,
    const std::vector<std::size_t>& output_strides,
    std::size_t flat_index
) {
    const std::size_t alignment =
        output_shape.size() - input_shape.size();

    std::size_t offset = 0;

    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        const std::size_t output_axis = alignment + i;

        const std::size_t coordinate =
            (flat_index / output_strides[output_axis])
            % output_shape[output_axis];

        if (input_shape[i] != 1) {
            offset += coordinate * input_strides[i];
        }
    }

    return offset;
}

template <typename Function>
Tensor elementwise_binary(
    const Tensor& lhs,
    const Tensor& rhs,
    Function function
) {
    const auto output_shape =
        broadcast_shape(lhs.shape(), rhs.shape());

    Tensor result(output_shape, 0.0F);

    const auto left = lhs.data();
    const auto right = rhs.data();
    auto output = result.data();

    const auto lhs_strides = strides(lhs.shape());
    const auto rhs_strides = strides(rhs.shape());
    const auto output_strides = strides(output_shape);

    for (std::size_t i = 0; i < result.size(); ++i) {
        const std::size_t lhs_index = broadcast_offset(
            lhs.shape(), lhs_strides,
            output_shape, output_strides, i
        );

        const std::size_t rhs_index = broadcast_offset(
            rhs.shape(), rhs_strides,
            output_shape, output_strides, i
        );

        output[i] = function(left[lhs_index], right[rhs_index]);
    }

    return result;
}

std::vector<std::size_t> batch_shape(
    const Tensor& tensor
) {
    const auto& shape = tensor.shape();

    if (tensor.rank() == 1) {
        return {};
    }

    return std::vector<std::size_t>(
        shape.begin(),
        shape.end() - 2
    );
}

std::size_t batch_offset(
    const std::vector<std::size_t>& input_batch_shape,
    const std::vector<std::size_t>& output_batch_shape,
    std::size_t output_batch_index
) {
    if (input_batch_shape.empty()) {
        return 0;
    }

    const std::size_t alignment =
        output_batch_shape.size() - input_batch_shape.size();

    std::size_t input_batch_index = 0;

    for (std::size_t i = 0; i < input_batch_shape.size(); ++i) {
        const std::size_t output_axis = alignment + i;

        std::size_t stride = 1;

        for (
            std::size_t j = output_axis + 1;
            j < output_batch_shape.size();
            ++j
        ) {
            stride *= output_batch_shape[j];
        }

        const std::size_t coordinate =
            (output_batch_index / stride)
            % output_batch_shape[output_axis];

        const std::size_t input_coordinate =
            input_batch_shape[i] == 1 ? 0 : coordinate;

        input_batch_index =
            input_batch_index * input_batch_shape[i]
            + input_coordinate;
    }

    return input_batch_index;
}

} // namespace

Tensor operator+(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a + b; }
    );
}

Tensor operator-(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a - b; }
    );
}

Tensor operator*(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a * b; }
    );
}

Tensor operator/(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a / b; }
    );
}

Tensor operator-(const Tensor& input) {
    Tensor result(input.shape(), 0.0F);

    const auto values = input.data();
    auto output = result.data();

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = -values[i];
    }

    return result;
}

Tensor transpose(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "transpose: input must be a rank-2 tensor"
        );
    }

    const std::size_t rows = input.shape()[0];
    const std::size_t cols = input.shape()[1];

    Tensor result({cols, rows}, 0.0F);

    const auto values = input.data();
    auto output = result.data();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            output[j * rows + i] = values[i * cols + j];
        }
    }

    return result;
}

Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs
) {
    const std::size_t lhs_rank = lhs.rank();
    const std::size_t rhs_rank = rhs.rank();

    if (
        lhs_rank < 1 || lhs_rank > 5
        || rhs_rank < 1 || rhs_rank > 5
    ) {
        throw std::invalid_argument(
            "matmul: input ranks must be between 1 and 5"
        );
    }

    const auto& lhs_shape = lhs.shape();
    const auto& rhs_shape = rhs.shape();

    const std::size_t m =
        lhs_rank == 1 ? 1 : lhs_shape[lhs_rank - 2];

    const std::size_t lhs_k = lhs_shape[lhs_rank - 1];

    const std::size_t rhs_k =
        rhs_rank == 1 ? rhs_shape[0] : rhs_shape[rhs_rank - 2];

    const std::size_t n =
        rhs_rank == 1 ? 1 : rhs_shape[rhs_rank - 1];

    if (lhs_k != rhs_k) {
        throw std::invalid_argument(
            "matmul: inner dimensions do not match"
        );
    }

    const auto lhs_batch_shape = batch_shape(lhs);
    const auto rhs_batch_shape = batch_shape(rhs);

    const auto output_batch_shape =
        broadcast_shape(lhs_batch_shape, rhs_batch_shape);

    std::vector<std::size_t> output_shape = output_batch_shape;

    if (lhs_rank > 1) {
        output_shape.push_back(m);
    }

    if (rhs_rank > 1) {
        output_shape.push_back(n);
    }

    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    Tensor result(output_shape, 0.0F);

    const auto left = lhs.data();
    const auto right = rhs.data();
    auto output = result.data();

    const std::size_t batch_count =
        element_count(output_batch_shape);

    const std::size_t lhs_matrix_size = m * lhs_k;
    const std::size_t rhs_matrix_size = rhs_k * n;
    const std::size_t output_matrix_size = m * n;

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        const std::size_t lhs_base =
            batch_offset(lhs_batch_shape, output_batch_shape, batch)
            * lhs_matrix_size;

        const std::size_t rhs_base =
            batch_offset(rhs_batch_shape, output_batch_shape, batch)
            * rhs_matrix_size;

        const std::size_t output_base =
            batch * output_matrix_size;

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t k = 0; k < lhs_k; ++k) {
                const float lhs_value =
                    left[lhs_base + i * lhs_k + k];

                for (std::size_t j = 0; j < n; ++j) {
                    output[output_base + i * n + j] +=
                        lhs_value * right[rhs_base + k * n + j];
                }
            }
        }
    }

    return result;
}

} // namespace neural