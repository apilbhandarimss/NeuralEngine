
#pragma once
#include <cstddef>
#include <span>
#include <vector>
#include <initializer_list>

namespace neural {

    class Tensor {
    public:
        explicit Tensor(std::vector<std::size_t> shape);
        Tensor(std::vector<std::size_t> shape, float fill_value);

        [[nodiscard]]
        const std::vector<std::size_t>& shape() const noexcept;

        [[nodiscard]]
        std::size_t rank() const noexcept;

        [[nodiscard]]
        std::size_t size() const noexcept;

        float& at(std::initializer_list<std::size_t> indices);

        const float& at(
            std::initializer_list<std::size_t> indices
        ) const;

        [[nodiscard]]
        std::span<const float> data() const noexcept;

        [[nodiscard]]
        std::span<float> data() noexcept;

    private:
        [[nodiscard]]
        std::size_t offset(
            std::initializer_list<std::size_t> indices
        ) const;

        std::vector<std::size_t> shape_;
        std::vector<float> data_;
    };

}
