
#include <neural/tensor/tensor.hpp>
#include <cstddef>
#include <stdexcept>

namespace neural {

namespace {

std::size_t element_count(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape must not be empty");
    }

    std::size_t count = 1;

    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument("Tensor dimensions must be non-zero");
        }

        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("Tensor element count overflow");
        }

        count *= dimension;
    }

    return count;
}

} // namespace

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)),
      data_(element_count(shape_), 0.0F) {}

Tensor::Tensor(std::vector<std::size_t> shape, float fill_value)
    : shape_(std::move(shape)),
      data_(element_count(shape_), fill_value) {}

const std::vector<std::size_t>& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

std::size_t Tensor::size() const noexcept {
    return data_.size();
}

std::size_t Tensor::offset(
    std::initializer_list<std::size_t> indices
) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument(
            "Tensor index rank does not match shape"
        );
    }

    std::size_t result = 0;
    std::size_t axis = 0;

    for (const std::size_t index : indices) {
        if (index >= shape_[axis]) {
            throw std::out_of_range("Tensor index out of range");
        }

        result = result * shape_[axis] + index;
        ++axis;
    }

    return result;
}

float& Tensor::at(
    std::initializer_list<std::size_t> indices
) {
    return data_[offset(indices)];
}

const float& Tensor::at(
    std::initializer_list<std::size_t> indices
) const {
    return data_[offset(indices)];
}

std::span<float> Tensor::data() noexcept {
    return data_;
}

std::span<const float> Tensor::data() const noexcept {
    return data_;
}

} // namespace neural
