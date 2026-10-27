
#include "neural/ops/loss.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace neural {

LossResult softmax_cross_entropy(
    const Tensor& logits,
    std::size_t target
) {
    if (logits.rank() != 1 || logits.size() == 0) {
        throw std::invalid_argument(
            "Softmax cross-entropy expects non-empty rank-1 logits"
        );
    }

    if (target >= logits.size()) {
        throw std::out_of_range(
            "Target class is outside the logits range"
        );
    }

    const auto input = logits.data();

    float max_logit = input[0];

    for (const float value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Logits must be finite");
        }

        if (value > max_logit) {
            max_logit = value;
        }
    }

    Tensor probabilities({logits.size()}, 0.0F);
    auto probs = probabilities.data();

    float exp_sum = 0.0F;

    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(input[i] - max_logit);
        exp_sum += probs[i];
    }

    for (float& probability : probs) {
        probability /= exp_sum;
    }

    const float loss =
        std::log(exp_sum) + max_logit - input[target];

    Tensor gradient({logits.size()}, 0.0F);
    auto grad = gradient.data();

    for (std::size_t i = 0; i < logits.size(); ++i) {
        grad[i] = probs[i];
    }

    grad[target] -= 1.0F;

    return {
        loss,
        std::move(gradient),
        std::move(probabilities)
    };
}

LossResult softmax_cross_entropy(
    const Tensor& logits,
    const Tensor& targets
) {
    if (logits.rank() != 2) {
        throw std::invalid_argument(
            "Batch softmax cross-entropy expects rank-2 logits"
        );
    }

    const std::size_t batch_size = logits.shape()[0];
    const std::size_t num_classes = logits.shape()[1];

    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument(
            "Logits batch and class dimensions must be non-empty"
        );
    }

    if (targets.rank() != 1 ||
        targets.shape()[0] != batch_size) {
        throw std::invalid_argument(
            "Targets must have shape {batch_size}"
        );
    }

    Tensor probabilities(logits.shape(), 0.0F);
    Tensor gradient(logits.shape(), 0.0F);

    const auto input = logits.data();
    const auto target_values = targets.data();

    auto probs = probabilities.data();
    auto grad = gradient.data();

    float total_loss = 0.0F;

    for (std::size_t b = 0; b < batch_size; ++b) {
        const float target_value = target_values[b];

        if (!std::isfinite(target_value) ||
            target_value < 0.0F ||
            target_value >= static_cast<float>(num_classes) ||
            std::floor(target_value) != target_value) {
            throw std::invalid_argument(
                "Each target must be a valid integer class index"
            );
        }

        const std::size_t target =
            static_cast<std::size_t>(target_value);

        const std::size_t offset = b * num_classes;

        float max_logit = input[offset];

        for (std::size_t j = 0; j < num_classes; ++j) {
            const float value = input[offset + j];

            if (!std::isfinite(value)) {
                throw std::invalid_argument("Logits must be finite");
            }

            if (value > max_logit) {
                max_logit = value;
            }
        }

        float exp_sum = 0.0F;

        for (std::size_t j = 0; j < num_classes; ++j) {
            const float value =
                std::exp(input[offset + j] - max_logit);

            probs[offset + j] = value;
            exp_sum += value;
        }

        for (std::size_t j = 0; j < num_classes; ++j) {
            probs[offset + j] /= exp_sum;
            grad[offset + j] = probs[offset + j];
        }

        total_loss +=
            std::log(exp_sum) + max_logit - input[offset + target];

        grad[offset + target] -= 1.0F;
    }

    // Scale gradients to match the mean batch loss.
    const float scale = 1.0F / static_cast<float>(batch_size);

    for (float& value : grad) {
        value *= scale;
    }

    return {
        total_loss / static_cast<float>(batch_size),
        std::move(gradient),
        std::move(probabilities)
    };
}

} // namespace neural
