#pragma once
#include "neural/nn/batch_norm.hpp"

#include "neural/nn/layer.hpp"

namespace neural {

class SGD {
public:
    void step(BatchNorm& layer) const;
    explicit SGD(float learning_rate);

    void step(Dense& layer) const;

    [[nodiscard]] float learning_rate() const noexcept;

private:
    float learning_rate_;
};
class Adam {
public:
    explicit Adam(float learning_rate);

    void step(Dense& layer) const;
    void step(BatchNorm& layer) const;
    [[nodiscard]] float learning_rate() const noexcept;

private:
    float learning_rate_;
    float beta1_ = 0.9F;
float beta2_ = 0.999F;
float epsilon_ = 1e-8F;

std::vector<float> weight_m_;
std::vector<float> weight_v_;
std::vector<float> bias_m_;
std::vector<float> bias_v_;

std::size_t timestep_ = 0;
};

}