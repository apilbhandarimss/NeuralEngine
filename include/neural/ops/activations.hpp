
#pragma once

#include "neural/ops/operation.hpp"

#include <optional>

namespace neural {

class ReLU final : public Operation {
public:
    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] Tensor backward(const Tensor& output_gradient) override;

private:
    std::optional<Tensor> cached_output_;
};

class Sigmoid final : public Operation {
public:
    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] Tensor backward(const Tensor& output_gradient) override;

private:
    std::optional<Tensor> cached_output_;
};

class Tanh final : public Operation {
public:
    [[nodiscard]] Tensor forward(const Tensor& input) const override;
    [[nodiscard]] Tensor backward(const Tensor& output_gradient) override;

private:
    std::optional<Tensor> cached_output_;
};

}
