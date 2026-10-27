
#include "neural/nn/batch_norm.hpp"
#include "neural/tensor/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural {

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

    if (training) {
        batch_mean_.assign(features_, 0.0F);
        std::vector<float> variance(features_, 0.0F);

        // Calculate the mean for each feature.
        for (std::size_t i = 0; i < batch_size; ++i) {
            for (std::size_t j = 0; j < features_; ++j) {
                batch_mean_[j] += input.at({i, j});
            }
        }

        for (std::size_t j = 0; j < features_; ++j) {
            batch_mean_[j] /= static_cast<float>(batch_size);
        }

        // Calculate population variance for each feature.
        for (std::size_t i = 0; i < batch_size; ++i) {
            for (std::size_t j = 0; j < features_; ++j) {
                const float difference =
                    input.at({i, j}) - batch_mean_[j];

                variance[j] += difference * difference;
            }
        }

        // Calculate inverse standard deviation and update running statistics.
        inverse_std_.resize(features_);

        for (std::size_t j = 0; j < features_; ++j) {
            variance[j] /= static_cast<float>(batch_size);

            inverse_std_[j] =
                1.0F / std::sqrt(variance[j] + epsilon_);

            running_mean_[j] =
                (1.0F - momentum_) * running_mean_[j]
                + momentum_ * batch_mean_[j];

            running_variance_[j] =
                (1.0F - momentum_) * running_variance_[j]
                + momentum_ * variance[j];
        }

        // Cache normalised input for backward propagation.
        normalized_.emplace(input_shape, 0.0F);

        for (std::size_t i = 0; i < batch_size; ++i) {
            for (std::size_t j = 0; j < features_; ++j) {
                const float normalized_value =
                    (input.at({i, j}) - batch_mean_[j])
                    * inverse_std_[j];

                normalized_->at({i, j}) = normalized_value;

                output.at({i, j}) =
                    gamma_[j] * normalized_value + beta_[j];
            }
        }

        input_ = input;
        batch_size_ = batch_size;
        training_cache_valid_ = true;
    } else {
        // Inference uses running statistics, not current batch statistics.
        for (std::size_t i = 0; i < batch_size; ++i) {
            for (std::size_t j = 0; j < features_; ++j) {
                const float inverse_std =
                    1.0F / std::sqrt(
                        running_variance_[j] + epsilon_
                    );

                const float normalized_value =
                    (input.at({i, j}) - running_mean_[j])
                    * inverse_std;

                output.at({i, j}) =
                    gamma_[j] * normalized_value + beta_[j];
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

    gamma_grad_.assign(features_, 0.0F);
    beta_grad_.assign(features_, 0.0F);

    // Calculate parameter gradients across the batch.
    for (std::size_t i = 0; i < batch_size_; ++i) {
        for (std::size_t j = 0; j < features_; ++j) {
            const float grad = output_gradient.at({i, j});
            const float x_hat = normalized_->at({i, j});

            beta_grad_[j] += grad;
            gamma_grad_[j] += grad * x_hat;
        }
    }

    // Average gamma and beta gradients over the batch.
    // const float scale =
    //     1.0F / static_cast<float>(batch_size_);

    // for (std::size_t j = 0; j < features_; ++j) {
    //     gamma_grad_[j] *= scale;
    //     beta_grad_[j] *= scale;
    // } this is not need as i have ammended loss function 

    // Calculate input gradients. Do not average these separately.
    Tensor input_gradient(input_->shape(), 0.0F);

    const float n = static_cast<float>(batch_size_);

    for (std::size_t j = 0; j < features_; ++j) {
        float sum_gradient = 0.0F;
        float sum_gradient_normalized = 0.0F;

        for (std::size_t i = 0; i < batch_size_; ++i) {
            const float grad = output_gradient.at({i, j});
            const float x_hat = normalized_->at({i, j});

            sum_gradient += grad;
            sum_gradient_normalized += grad * x_hat;
        }

        for (std::size_t i = 0; i < batch_size_; ++i) {
            const float grad = output_gradient.at({i, j});
            const float x_hat = normalized_->at({i, j});

            input_gradient.at({i, j}) =
                (gamma_[j] * inverse_std_[j] / n)
                * (n * grad
                   - sum_gradient
                   - x_hat * sum_gradient_normalized);
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
