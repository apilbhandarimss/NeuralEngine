#pragma once
#include <unordered_map>
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

class MomentumSGD {
public:
    explicit MomentumSGD(float learning_rate,float momentum_=0.9F,bool use_norm_=true);

    void step(Dense& layer);
    void step(BatchNorm& layer);

    [[nodiscard]] float learning_rate() const noexcept;

private:


    
    float learning_rate_;
    float momentum_ ;
    bool use_norm_;
    struct VelocityState {
    std::vector<float> weight_velocity;
    std::vector<float> bias_velocity;
};

std::unordered_map<Dense*, VelocityState> dense_states_;
std::unordered_map<BatchNorm*, VelocityState> norm_states_;

    
};
class RMSprop {
public:
    explicit RMSprop(float learning_rate,float momentum_=0.999F,bool use_norm_=true);

    void step(Dense& layer);
    void step(BatchNorm& layer);

    [[nodiscard]] float learning_rate() const noexcept;

private:


    float epsilon_=1e-5F;
    float learning_rate_;
    float momentum_ ;
    bool use_norm_;
    struct VelocityState {
    std::vector<float> weight_velocity;
    std::vector<float> bias_velocity;
};

std::unordered_map<Dense*, VelocityState> dense_states_;
std::unordered_map<BatchNorm*, VelocityState> norm_states_;

    
};
class Adam {
public:
    explicit Adam(float learning_rate, float beta1 = 0.9F, float beta2 = 0.999F);

    void step(Dense& layer);
    void step(BatchNorm& layer);

    [[nodiscard]] float learning_rate() const noexcept;

private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_ = 1e-8F;

    struct AdamState {
        std::vector<float> weight_m;
        std::vector<float> bias_m;
        std::vector<float> weight_v;
        std::vector<float> bias_v;
        std::size_t timestep = 0;
    };

    std::unordered_map<Dense*, AdamState> dense_states_;
    std::unordered_map<BatchNorm*, AdamState> norm_states_;
};

}