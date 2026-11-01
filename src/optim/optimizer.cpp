// compile with: -O3 -march=native -fopenmp -fno-math-errno
#include "neural/optim/optimizer.hpp"
#include "neural/nn/layer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace neural {

namespace {

constexpr int kThreads = 14;
// Element-wise updates are cheap, so tiny layers stay single-threaded.
constexpr std::size_t kParallelMin = 16384;

} // namespace

// ---------------------------------------------------------------- SGD

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

    const float lr = learning_rate_;

    const std::size_t wn = weights.size();
    float* __restrict w = weights.data();
    const float* __restrict dw = weight_gradients.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (wn > kParallelMin)
    for (std::size_t i = 0; i < wn; ++i) {
        w[i] -= lr * dw[i];
    }

    const std::size_t bn = biases.size();
    float* __restrict b = biases.data();
    const float* __restrict db = bias_gradients.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (bn > kParallelMin)
    for (std::size_t i = 0; i < bn; ++i) {
        b[i] -= lr * db[i];
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

    const std::size_t n = gamma.size();
    if (n == 0) return;

    const float lr = learning_rate_;
    float* __restrict g = &gamma[0];
    float* __restrict bt = &beta[0];
    const float* __restrict dg = &gamma_grad[0];
    const float* __restrict dbt = &beta_grad[0];

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMin)
    for (std::size_t i = 0; i < n; ++i) {
        g[i] -= lr * dg[i];
        bt[i] -= lr * dbt[i];
    }
}

// ------------------------------------------------------- MomentumSGD

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

    // use_norm_ was branched on inside the loop; it is loop-invariant, so
    // fold it into one scale factor (1 for plain, 1 - momentum for EMA).
    const float lr = learning_rate_;
    const float mom = momentum_;
    const float scale = use_norm_ ? (1.0F - momentum_) : 1.0F;

    const std::size_t wn = weight_gradients.size();
    float* __restrict w = weights.data();
    const float* __restrict dw = weight_gradients.data();
    float* __restrict vw = state.weight_velocity.data();

    // Velocity update and weight update fused: one pass over memory.
    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (wn > kParallelMin)
    for (std::size_t i = 0; i < wn; ++i) {
        vw[i] = mom * vw[i] + scale * dw[i];
        w[i] -= lr * vw[i];
    }

    const std::size_t bn = bias_gradients.size();
    float* __restrict b = biases.data();
    const float* __restrict db = bias_gradients.data();
    float* __restrict vb = state.bias_velocity.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (bn > kParallelMin)
    for (std::size_t i = 0; i < bn; ++i) {
        vb[i] = mom * vb[i] + scale * db[i];
        b[i] -= lr * vb[i];
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

    const std::size_t n = gamma.size();
    if (n == 0) return;

    const float lr = learning_rate_;
    const float mom = momentum_;
    const float scale = use_norm_ ? (1.0F - momentum_) : 1.0F;

    float* __restrict g = &gamma[0];
    float* __restrict bt = &beta[0];
    const float* __restrict dg = &gamma_grad[0];
    const float* __restrict dbt = &beta_grad[0];
    float* __restrict vg = state.weight_velocity.data();
    float* __restrict vb = state.bias_velocity.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMin)
    for (std::size_t i = 0; i < n; ++i) {
        vg[i] = mom * vg[i] + scale * dg[i];
        vb[i] = mom * vb[i] + scale * dbt[i];

        g[i] -= lr * vg[i];
        bt[i] -= lr * vb[i];
    }
}

// ----------------------------------------------------------- RMSprop

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

    // Side note on naming: weight_velocity is really the running average of
    // squared gradients, and "momentum" is just beta in:
    //   V_t = beta * V_{t-1} + (1 - beta) * g_t^2
    const float lr = learning_rate_;
    const float beta = momentum_;
    const float one_minus_beta = 1.0F - momentum_;
    const float eps = epsilon_;

    const std::size_t wn = weight_gradients.size();
    float* __restrict w = weights.data();
    const float* __restrict dw = weight_gradients.data();
    float* __restrict vw = state.weight_velocity.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (wn > kParallelMin)
    for (std::size_t i = 0; i < wn; ++i) {
        const float g = dw[i];
        vw[i] = beta * vw[i] + one_minus_beta * g * g;
        w[i] -= lr * g / (std::sqrt(vw[i]) + eps);
    }

    const std::size_t bn = bias_gradients.size();
    float* __restrict b = biases.data();
    const float* __restrict db = bias_gradients.data();
    float* __restrict vb = state.bias_velocity.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (bn > kParallelMin)
    for (std::size_t i = 0; i < bn; ++i) {
        const float g = db[i];
        vb[i] = beta * vb[i] + one_minus_beta * g * g;
        b[i] -= lr * g / (std::sqrt(vb[i]) + eps);
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

    const std::size_t n = gamma.size();
    if (n == 0) return;

    const float lr = learning_rate_;
    const float decay = momentum_;
    const float one_minus_decay = 1.0F - momentum_;
    const float eps = epsilon_;

    float* __restrict g = &gamma[0];
    float* __restrict bt = &beta[0];
    const float* __restrict dg = &gamma_grad[0];
    const float* __restrict dbt = &beta_grad[0];
    float* __restrict vg = state.weight_velocity.data();
    float* __restrict vb = state.bias_velocity.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMin)
    for (std::size_t i = 0; i < n; ++i) {
        const float gg = dg[i];
        const float bg = dbt[i];

        vg[i] = decay * vg[i] + one_minus_decay * gg * gg;
        vb[i] = decay * vb[i] + one_minus_decay * bg * bg;

        g[i] -= lr * gg / (std::sqrt(vg[i]) + eps);
        bt[i] -= lr * bg / (std::sqrt(vb[i]) + eps);
    }
}

// -------------------------------------------------------------- Adam

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

    // Loop invariants hoisted: divisions become multiplications.
    const float lr = learning_rate_;
    const float b1 = beta1_;
    const float b2 = beta2_;
    const float omb1 = 1.0F - beta1_;
    const float omb2 = 1.0F - beta2_;
    const float inv1 = 1.0F / correction1;
    const float inv2 = 1.0F / correction2;
    const float eps = epsilon_;

    const std::size_t wn = weight_gradients.size();
    float* __restrict w = weights.data();
    const float* __restrict dw = weight_gradients.data();
    float* __restrict mw = state.weight_m.data();
    float* __restrict vw = state.weight_v.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (wn > kParallelMin)
    for (std::size_t i = 0; i < wn; ++i) {
        const float g = dw[i];

        mw[i] = b1 * mw[i] + omb1 * g;
        vw[i] = b2 * vw[i] + omb2 * g * g;

        const float m_hat = mw[i] * inv1;
        const float v_hat = vw[i] * inv2;

        w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }

    const std::size_t bn = bias_gradients.size();
    float* __restrict b = biases.data();
    const float* __restrict db = bias_gradients.data();
    float* __restrict mb = state.bias_m.data();
    float* __restrict vb = state.bias_v.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (bn > kParallelMin)
    for (std::size_t i = 0; i < bn; ++i) {
        const float g = db[i];

        mb[i] = b1 * mb[i] + omb1 * g;
        vb[i] = b2 * vb[i] + omb2 * g * g;

        const float m_hat = mb[i] * inv1;
        const float v_hat = vb[i] * inv2;

        b[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
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

    const std::size_t n = gamma.size();
    if (n == 0) return;

    const float lr = learning_rate_;
    const float b1 = beta1_;
    const float b2 = beta2_;
    const float omb1 = 1.0F - beta1_;
    const float omb2 = 1.0F - beta2_;
    const float inv1 = 1.0F / correction1;
    const float inv2 = 1.0F / correction2;
    const float eps = epsilon_;

    float* __restrict g = &gamma[0];
    float* __restrict bt = &beta[0];
    const float* __restrict dg = &gamma_grad[0];
    const float* __restrict dbt = &beta_grad[0];
    float* __restrict mg = state.weight_m.data();
    float* __restrict vg = state.weight_v.data();
    float* __restrict mb = state.bias_m.data();
    float* __restrict vb = state.bias_v.data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMin)
    for (std::size_t i = 0; i < n; ++i) {
        const float gg = dg[i];
        const float bg = dbt[i];

        mg[i] = b1 * mg[i] + omb1 * gg;
        vg[i] = b2 * vg[i] + omb2 * gg * gg;

        mb[i] = b1 * mb[i] + omb1 * bg;
        vb[i] = b2 * vb[i] + omb2 * bg * bg;

        const float gamma_m_hat = mg[i] * inv1;
        const float gamma_v_hat = vg[i] * inv2;

        const float beta_m_hat = mb[i] * inv1;
        const float beta_v_hat = vb[i] * inv2;

        g[i] -= lr * gamma_m_hat / (std::sqrt(gamma_v_hat) + eps);
        bt[i] -= lr * beta_m_hat / (std::sqrt(beta_v_hat) + eps);
    }
}

float Adam::learning_rate() const noexcept
{
    return learning_rate_;
}

} // namespace neural