
#pragma once

#include "neural/nn/layer_base.hpp"
#include "neural/tensor/tensor.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace neural {

class BatchNorm : public Layer{
public:
    explicit BatchNorm(
        std::size_t features,
        float epsilon = 1e-5F,
        float momentum = 0.1F
    );

    [[nodiscard]]
    Tensor forward(const Tensor& input, bool training = true);

    [[nodiscard]]
    Tensor backward(const Tensor& output_gradient);

    // Learnable parameters
    [[nodiscard]]
     std::vector<float>& gamma() noexcept;
    [[nodiscard]] 
    std::vector<float>& beta() noexcept;
    [[nodiscard]]
    const std::vector<float>& gamma_grad() const noexcept;

    [[nodiscard]]
    const std::vector<float>& beta_grad() const noexcept;

    // Running statistics for inference
    [[nodiscard]]
    const std::vector<float>& running_mean() const noexcept;

    [[nodiscard]]
    const std::vector<float>& running_variance() const noexcept;

    // Configuration
    [[nodiscard]]
    std::size_t features() const noexcept;

    [[nodiscard]]
    float epsilon() const noexcept;

    [[nodiscard]]
    float momentum() const noexcept;

    void update(float learning_rate);

private:
    std::size_t features_;
    float epsilon_;
    float momentum_;

    // Learnable parameters
    std::vector<float> gamma_;
    std::vector<float> beta_;

    // Parameter gradients
    std::vector<float> gamma_grad_;
    std::vector<float> beta_grad_;

    // Running statistics for inference
    std::vector<float> running_mean_;
    std::vector<float> running_variance_;

    // Cached values required by backward()
std::optional<Tensor> input_;
std::optional<Tensor> normalized_;
    std::vector<float> batch_mean_;
    std::vector<float> inverse_std_;

    std::size_t batch_size_ = 0;
    bool training_cache_valid_ = false;
    
};


} 
