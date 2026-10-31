#include "neural/optim/optimizer.hpp"
#include "neural/nn/layer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace neural {

SGD::SGD(float learning_rate)
    : learning_rate_(learning_rate) {
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument(
            "Learning rate must be finite and positive"
        );
    }
}

void SGD::step(Dense& layer) const {
    auto weights = layer.weights().data();
    const auto weight_gradients = layer.weight_gradients().data();

    auto biases = layer.bias().data();
    const auto bias_gradients = layer.bias_gradients().data();

    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate_ * weight_gradients[i];
    }

    for (std::size_t i = 0; i < biases.size(); ++i) {
        biases[i] -= learning_rate_ * bias_gradients[i];
    }
}

float SGD::learning_rate() const noexcept {
    return learning_rate_;
}
void SGD::step(BatchNorm& layer) const
{
    auto& gamma = layer.gamma();
    auto& beta = layer.beta();

    const auto& gamma_grad = layer.gamma_grad();
    const auto& beta_grad = layer.beta_grad();

    for (std::size_t i = 0; i < gamma.size(); ++i) {
        gamma[i] -= learning_rate_ * gamma_grad[i];
        beta[i] -= learning_rate_ * beta_grad[i];
    }
}
MomentumSGD::MomentumSGD(float learning_rate, float momentum, bool use_conv)
    : learning_rate_(learning_rate), momentum_(momentum), use_norm_(use_conv)
{
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument("Learning rate must be finite and positive");
    }

    if (!std::isfinite(momentum_) || momentum_ < 0.0F || momentum_ >= 1.0F) {
        throw std::invalid_argument("Momentum must be finite and in [0, 1)");
    }
}

void MomentumSGD::step(Dense& layer)
{
    auto weights = layer.weights().data();
    const auto weight_gradients = layer.weight_gradients().data();

    auto biases = layer.bias().data();
    const auto bias_gradients = layer.bias_gradients().data();

    auto& state = dense_states_[&layer];

if (state.weight_velocity.size() != weight_gradients.size()) {
    state.weight_velocity.assign(weight_gradients.size(), 0.0F);
}

if (state.bias_velocity.size() != bias_gradients.size()) {
    state.bias_velocity.assign(bias_gradients.size(), 0.0F);
}

    for (std::size_t i = 0; i < weight_gradients.size(); ++i) {
        if (use_norm_) {
            state.weight_velocity[i] = momentum_ * state.weight_velocity[i] + (1.0F - momentum_) * weight_gradients[i];
        } else {
            state.weight_velocity[i] = momentum_ * state.weight_velocity[i] + weight_gradients[i];
        }
    }

    for (std::size_t i = 0; i < bias_gradients.size(); ++i) {
        if (use_norm_) {
            state.bias_velocity[i] = momentum_ * state.bias_velocity[i] + (1.0F - momentum_) * bias_gradients[i];
        } else {
            state.bias_velocity[i] = momentum_ * state.bias_velocity[i] + bias_gradients[i];
        }
    }

    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate_ * state.weight_velocity[i];
    }

    for (std::size_t i = 0; i < biases.size(); ++i) {
        biases[i] -= learning_rate_ * state.bias_velocity[i];
    }
}
void MomentumSGD::step(BatchNorm& layer)
{
    auto& gamma = layer.gamma();
    auto& beta = layer.beta();

    const auto& gamma_grad = layer.gamma_grad();
    const auto& beta_grad = layer.beta_grad();

    auto& state = norm_states_[&layer];

    if (state.weight_velocity.size() != gamma_grad.size()) {
        state.weight_velocity.assign(gamma_grad.size(), 0.0F);
    }

    if (state.bias_velocity.size() != beta_grad.size()) {
        state.bias_velocity.assign(beta_grad.size(), 0.0F);
    }

    for (std::size_t i = 0; i < gamma.size(); ++i) {
        if (use_norm_) {
            state.weight_velocity[i] = momentum_ * state.weight_velocity[i] + (1.0F - momentum_) * gamma_grad[i];
            state.bias_velocity[i] = momentum_ * state.bias_velocity[i] + (1.0F - momentum_) * beta_grad[i];
        } else {
            state.weight_velocity[i] = momentum_ * state.weight_velocity[i] + gamma_grad[i];
            state.bias_velocity[i] = momentum_ * state.bias_velocity[i] + beta_grad[i];
        }

        gamma[i] -= learning_rate_ * state.weight_velocity[i];
        beta[i] -= learning_rate_ * state.bias_velocity[i];
    }
}
RMSprop::RMSprop(float learning_rate, float momentum, bool use_conv)
    : learning_rate_(learning_rate), momentum_(momentum), use_norm_(use_conv)
{
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument("Learning rate must be finite and positive");
    }

    if (!std::isfinite(momentum_) || momentum_ < 0.0F || momentum_ >= 1.0F) {
        throw std::invalid_argument("Momentum must be finite and in [0, 1)");
    }
}

void RMSprop::step(Dense& layer)
{
    auto weights = layer.weights().data();
    const auto weight_gradients = layer.weight_gradients().data();

    auto biases = layer.bias().data();
    const auto bias_gradients = layer.bias_gradients().data();

    auto& state = dense_states_[&layer];

if (state.weight_velocity.size() != weight_gradients.size()) {
    state.weight_velocity.assign(weight_gradients.size(), 0.0F);
}

if (state.bias_velocity.size() != bias_gradients.size()) {
    state.bias_velocity.assign(bias_gradients.size(), 0.0F);
}
//side note about the naming convention here 
// actually weight_velocity is very misleading as it is squared avg of gradient
//i was too lazy so be careful while reading this the code, 
//and the definations
//the use of momentum can be consfusing it is just beta, in the following eqn
//Vt=Vt-1*Beta+(1-Beta)*gt^2

    for (std::size_t i = 0; i < weight_gradients.size(); ++i) {
    
            state.weight_velocity[i] = momentum_ * state.weight_velocity[i] + (1.0F - momentum_) * weight_gradients[i]*weight_gradients[i];
        
    }

    for (std::size_t i = 0; i < bias_gradients.size(); ++i) {
        
            state.bias_velocity[i] = momentum_ * state.bias_velocity[i] + (1.0F - momentum_) * bias_gradients[i]*bias_gradients[i];
    }

for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] -= learning_rate_ * weight_gradients[i] / (std::sqrt(state.weight_velocity[i]) + epsilon_);
}

for (std::size_t i = 0; i < biases.size(); ++i) {
    biases[i] -= learning_rate_ * bias_gradients[i] / (std::sqrt(state.bias_velocity[i]) + epsilon_);
}
}
void RMSprop::step(BatchNorm& layer)
{
    auto& gamma = layer.gamma();
    auto& beta = layer.beta();

    const auto& gamma_grad = layer.gamma_grad();
    const auto& beta_grad = layer.beta_grad();

    auto& state = norm_states_[&layer];

    if (state.weight_velocity.size() != gamma_grad.size()) {
        state.weight_velocity.assign(gamma_grad.size(), 0.0F);
    }

    if (state.bias_velocity.size() != beta_grad.size()) {
        state.bias_velocity.assign(beta_grad.size(), 0.0F);
    }

for (std::size_t i = 0; i < gamma.size(); ++i) {
    state.weight_velocity[i] = momentum_ * state.weight_velocity[i] + (1.0F - momentum_) * gamma_grad[i] * gamma_grad[i];
    state.bias_velocity[i] = momentum_ * state.bias_velocity[i] + (1.0F - momentum_) * beta_grad[i] * beta_grad[i];

    gamma[i] -= learning_rate_ * gamma_grad[i] / (std::sqrt(state.weight_velocity[i]) +epsilon_);
    beta[i] -= learning_rate_ * beta_grad[i] / (std::sqrt(state.bias_velocity[i]) + epsilon_);
}
}
Adam::Adam(float learning_rate, float beta1, float beta2)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2)
{
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument("Learning rate must be finite and positive");
    }

    if (!std::isfinite(beta1_) || beta1_ < 0.0F || beta1_ >= 1.0F) {
        throw std::invalid_argument("Beta1 must be in [0, 1)");
    }

    if (!std::isfinite(beta2_) || beta2_ < 0.0F || beta2_ >= 1.0F) {
        throw std::invalid_argument("Beta2 must be in [0, 1)");
    }
}

void Adam::step(Dense& layer)
{
    auto weights = layer.weights().data();
    const auto weight_gradients = layer.weight_gradients().data();

    auto biases = layer.bias().data();
    const auto bias_gradients = layer.bias_gradients().data();

    auto& state = dense_states_[&layer];

    if (state.weight_m.size() != weight_gradients.size()) {
        state.weight_m.assign(weight_gradients.size(), 0.0F);
        state.weight_v.assign(weight_gradients.size(), 0.0F);
        state.timestep = 0;
    }

    if (state.bias_m.size() != bias_gradients.size()) {
        state.bias_m.assign(bias_gradients.size(), 0.0F);
        state.bias_v.assign(bias_gradients.size(), 0.0F);
        state.timestep = 0;
    }

    ++state.timestep;

    const float correction1 =
        1.0F - std::pow(beta1_, static_cast<float>(state.timestep));

    const float correction2 =
        1.0F - std::pow(beta2_, static_cast<float>(state.timestep));

    for (std::size_t i = 0; i < weight_gradients.size(); ++i) {
        state.weight_m[i] =
            beta1_ * state.weight_m[i] +
            (1.0F - beta1_) * weight_gradients[i];

        state.weight_v[i] =
            beta2_ * state.weight_v[i] +
            (1.0F - beta2_) * weight_gradients[i] * weight_gradients[i];

        const float m_hat = state.weight_m[i] / correction1;
        const float v_hat = state.weight_v[i] / correction2;

        weights[i] -= learning_rate_ * m_hat /
            (std::sqrt(v_hat) + epsilon_);
    }

    for (std::size_t i = 0; i < bias_gradients.size(); ++i) {
        state.bias_m[i] =
            beta1_ * state.bias_m[i] +
            (1.0F - beta1_) * bias_gradients[i];

        state.bias_v[i] =
            beta2_ * state.bias_v[i] +
            (1.0F - beta2_) * bias_gradients[i] * bias_gradients[i];

        const float m_hat = state.bias_m[i] / correction1;
        const float v_hat = state.bias_v[i] / correction2;

        biases[i] -= learning_rate_ * m_hat /
            (std::sqrt(v_hat) + epsilon_);
    }
}

void Adam::step(BatchNorm& layer)
{
    auto& gamma = layer.gamma();
    auto& beta = layer.beta();

    const auto& gamma_grad = layer.gamma_grad();
    const auto& beta_grad = layer.beta_grad();

    auto& state = norm_states_[&layer];

    if (state.weight_m.size() != gamma_grad.size()) {
        state.weight_m.assign(gamma_grad.size(), 0.0F);
        state.weight_v.assign(gamma_grad.size(), 0.0F);
        state.timestep = 0;
    }

    if (state.bias_m.size() != beta_grad.size()) {
        state.bias_m.assign(beta_grad.size(), 0.0F);
        state.bias_v.assign(beta_grad.size(), 0.0F);
        state.timestep = 0;
    }

    ++state.timestep;

    const float correction1 =
        1.0F - std::pow(beta1_, static_cast<float>(state.timestep));

    const float correction2 =
        1.0F - std::pow(beta2_, static_cast<float>(state.timestep));

    for (std::size_t i = 0; i < gamma.size(); ++i) {
        state.weight_m[i] =
            beta1_ * state.weight_m[i] +
            (1.0F - beta1_) * gamma_grad[i];

        state.weight_v[i] =
            beta2_ * state.weight_v[i] +
            (1.0F - beta2_) * gamma_grad[i] * gamma_grad[i];

        state.bias_m[i] =
            beta1_ * state.bias_m[i] +
            (1.0F - beta1_) * beta_grad[i];

        state.bias_v[i] =
            beta2_ * state.bias_v[i] +
            (1.0F - beta2_) * beta_grad[i] * beta_grad[i];

        const float gamma_m_hat = state.weight_m[i] / correction1;
        const float gamma_v_hat = state.weight_v[i] / correction2;

        const float beta_m_hat = state.bias_m[i] / correction1;
        const float beta_v_hat = state.bias_v[i] / correction2;

        gamma[i] -= learning_rate_ * gamma_m_hat /
            (std::sqrt(gamma_v_hat) + epsilon_);

        beta[i] -= learning_rate_ * beta_m_hat /
            (std::sqrt(beta_v_hat) + epsilon_);
    }
}

float Adam::learning_rate() const noexcept
{
    return learning_rate_;
}
}