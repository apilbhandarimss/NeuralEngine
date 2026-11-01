// compile with: -O3 -march=native -fopenmp -fno-math-errno
#include "neural/nn/batch_norm.hpp"
#include "neural/tensor/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural {

namespace {

constexpr int kThreads = 14;
// Below this many elements, thread startup costs more than it saves.
constexpr std::size_t kParallelMin = 16384;

} // namespace

BatchNorm::BatchNorm(
    std::size_t features,
    float epsilon,
    float momentum
)
    : features_(features),
      epsilon_(epsilon),
      momentum_(momentum),
      gamma_(features, 1.0F),
      beta_(features, 0.0F),
      gamma_grad_(features, 0.0F),
      beta_grad_(features, 0.0F),
      running_mean_(features, 0.0F),
      running_variance_(features, 1.0F),
      batch_mean_(features, 0.0F),
      inverse_std_(features, 0.0F)
{
    if (features_ == 0) {
        throw std::invalid_argument(
            "BatchNorm requires at least one feature"
        );
    }

    if (!std::isfinite(epsilon_) || epsilon_ <= 0.0F) {
        throw std::invalid_argument(
            "BatchNorm epsilon must be finite and positive"
        );
    }

    if (!std::isfinite(momentum_) ||
        momentum_ < 0.0F ||
        momentum_ > 1.0F) {
        throw std::invalid_argument(
            "BatchNorm momentum must be finite and between 0 and 1"
        );
    }
}

Tensor BatchNorm::forward(const Tensor& input, bool training)
{
    const auto& input_shape = input.shape();

    if (input_shape.size() != 2) {
        throw std::invalid_argument(
            "BatchNorm expects a rank-2 tensor"
        );
    }

    const std::size_t batch_size = input_shape[0];

    if (batch_size == 0 || input_shape[1] != features_) {
        throw std::invalid_argument(
            "Incompatible BatchNorm input shape"
        );
    }

    Tensor output(input_shape, 0.0F);

    const std::size_t F = features_;
    const std::size_t B = batch_size;
    const bool par = B * F > kParallelMin;

    const float* x = input.data().data();
    float* y = output.data().data();
    const float* gamma = gamma_.data();
    const float* beta = beta_.data();

    if (training) {
        batch_mean_.assign(F, 0.0F);
        std::vector<float> variance(F, 0.0F);

        float* mean = batch_mean_.data();
        float* var = variance.data();

        // Calculate the mean for each feature.
        // Each thread sums its own rows into a private copy of mean[]
        // (OpenMP array reduction), so there are no races.
        #pragma omp parallel for schedule(static) num_threads(kThreads) \
            if (par) reduction(+ : mean[0 : F])
        for (std::size_t i = 0; i < B; ++i) {
            const float* xr = x + i * F;

            #pragma omp simd
            for (std::size_t j = 0; j < F; ++j) {
                mean[j] += xr[j];
            }
        }

        const float inv_b = 1.0F / static_cast<float>(B);

        #pragma omp simd
        for (std::size_t j = 0; j < F; ++j) {
            mean[j] *= inv_b;
        }

        // Calculate population variance for each feature.
        #pragma omp parallel for schedule(static) num_threads(kThreads) \
            if (par) reduction(+ : var[0 : F])
        for (std::size_t i = 0; i < B; ++i) {
            const float* xr = x + i * F;

            #pragma omp simd
            for (std::size_t j = 0; j < F; ++j) {
                const float difference = xr[j] - mean[j];
                var[j] += difference * difference;
            }
        }

        // Calculate inverse standard deviation and update running statistics.
        inverse_std_.resize(F);

        float* inv_std = inverse_std_.data();
        float* run_mean = running_mean_.data();
        float* run_var = running_variance_.data();

        const float momentum = momentum_;
        const float one_minus_momentum = 1.0F - momentum_;
        const float epsilon = epsilon_;

        #pragma omp simd
        for (std::size_t j = 0; j < F; ++j) {
            var[j] *= inv_b;

            inv_std[j] = 1.0F / std::sqrt(var[j] + epsilon);

            run_mean[j] =
                one_minus_momentum * run_mean[j]
                + momentum * mean[j];

            run_var[j] =
                one_minus_momentum * run_var[j]
                + momentum * var[j];
        }

        // Cache normalised input for backward propagation.
        normalized_.emplace(input_shape, 0.0F);
        float* xh = normalized_->data().data();

        // Rows are independent: each thread owns whole rows of xh and y.
        #pragma omp parallel for schedule(static) num_threads(kThreads) if (par)
        for (std::size_t i = 0; i < B; ++i) {
            const float* xr = x + i * F;
            float* xhr = xh + i * F;
            float* yr = y + i * F;

            #pragma omp simd
            for (std::size_t j = 0; j < F; ++j) {
                const float normalized_value =
                    (xr[j] - mean[j]) * inv_std[j];

                xhr[j] = normalized_value;
                yr[j] = gamma[j] * normalized_value + beta[j];
            }
        }

        input_ = input;
        batch_size_ = batch_size;
        training_cache_valid_ = true;
    } else {
        // Inference uses running statistics, not current batch statistics.
        // The original recomputed sqrt for every element; per-feature scale
        // and shift are computed once instead:
        //   y = gamma * (x - mean) * inv_std + beta = x * scale + shift
        std::vector<float> scale_v(F);
        std::vector<float> shift_v(F);

        float* scale = scale_v.data();
        float* shift = shift_v.data();
        const float* run_mean = running_mean_.data();
        const float* run_var = running_variance_.data();
        const float epsilon = epsilon_;

        #pragma omp simd
        for (std::size_t j = 0; j < F; ++j) {
            const float inverse_std =
                1.0F / std::sqrt(run_var[j] + epsilon);

            scale[j] = gamma[j] * inverse_std;
            shift[j] = beta[j] - run_mean[j] * scale[j];
        }

        #pragma omp parallel for schedule(static) num_threads(kThreads) if (par)
        for (std::size_t i = 0; i < B; ++i) {
            const float* xr = x + i * F;
            float* yr = y + i * F;

            #pragma omp simd
            for (std::size_t j = 0; j < F; ++j) {
                yr[j] = xr[j] * scale[j] + shift[j];
            }
        }

        training_cache_valid_ = false;
        input_.reset();
        normalized_.reset();
    }

    return output;
}

Tensor BatchNorm::backward(const Tensor& output_gradient)
{
    if (!training_cache_valid_ || !input_ || !normalized_) {
        throw std::logic_error(
            "BatchNorm backward called without a valid training forward pass"
        );
    }

    if (output_gradient.shape() != input_->shape()) {
        throw std::invalid_argument(
            "BatchNorm gradient shape does not match input shape"
        );
    }

    const std::size_t F = features_;
    const std::size_t B = batch_size_;
    const bool par = B * F > kParallelMin;

    gamma_grad_.assign(F, 0.0F);
    beta_grad_.assign(F, 0.0F);

    const float* dy = output_gradient.data().data();
    const float* xh = normalized_->data().data();

    float* gg = gamma_grad_.data();
    float* bg = beta_grad_.data();

    // Calculate parameter gradients across the batch.
    // Array reduction: each thread accumulates into private copies, then
    // they are combined, so there are no races.
    #pragma omp parallel for schedule(static) num_threads(kThreads) \
        if (par) reduction(+ : gg[0 : F], bg[0 : F])
    for (std::size_t i = 0; i < B; ++i) {
        const float* dyr = dy + i * F;
        const float* xhr = xh + i * F;

        #pragma omp simd
        for (std::size_t j = 0; j < F; ++j) {
            const float grad = dyr[j];

            bg[j] += grad;
            gg[j] += grad * xhr[j];
        }
    }

    // Gamma and beta gradients are not averaged here because the loss
    // function already averages over the batch.

    // Calculate input gradients. Do not average these separately.
    // sum_gradient == beta_grad_ and sum_gradient_normalized == gamma_grad_,
    // so the second full pass over the batch the original did is not needed.
    Tensor input_gradient(input_->shape(), 0.0F);
    float* dx = input_gradient.data().data();

    const float n = static_cast<float>(B);
    const float inv_n = 1.0F / n;

    std::vector<float> coef_v(F);
    float* coef = coef_v.data();
    const float* gamma = gamma_.data();
    const float* inv_std = inverse_std_.data();

    #pragma omp simd
    for (std::size_t j = 0; j < F; ++j) {
        coef[j] = gamma[j] * inv_std[j] * inv_n;
    }

    // Rows are independent: each thread owns whole rows of dx.
    #pragma omp parallel for schedule(static) num_threads(kThreads) if (par)
    for (std::size_t i = 0; i < B; ++i) {
        const float* dyr = dy + i * F;
        const float* xhr = xh + i * F;
        float* dxr = dx + i * F;

        #pragma omp simd
        for (std::size_t j = 0; j < F; ++j) {
            dxr[j] = coef[j]
                * (n * dyr[j] - bg[j] - xhr[j] * gg[j]);
        }
    }

    training_cache_valid_ = false;
    input_.reset();
    normalized_.reset();

    return input_gradient;
}

std::vector<float>& BatchNorm::gamma() noexcept
{
    return gamma_;
}

std::vector<float>& BatchNorm::beta() noexcept
{
    return beta_;
}

const std::vector<float>& BatchNorm::gamma_grad() const noexcept
{
    return gamma_grad_;
}

const std::vector<float>& BatchNorm::beta_grad() const noexcept
{
    return beta_grad_;
}

} // namespace neural