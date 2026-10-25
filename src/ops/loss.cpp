
#include "neural/ops/loss.hpp"

#include <cmath>
#include <stdexcept>

namespace neural {

LossResult softmax_cross_entropy(
    const Tensor& logits,
    std::size_t target
) {
    if (logits.rank() != 1) {
        throw std::invalid_argument(
            "Softmax cross-entropy expects rank-1 logits"
        );
    }

    if (target >= logits.size()) {
        throw std::out_of_range(
            "Target class is outside the logits range"
        );
    }

    const auto input = logits.data();

    // Subtracting the maximum improves numerical stability.
    float max_logit = input[0];

    for (const float value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Logits must be finite"
            );
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

    // Cross-entropy for the correct class.
    const float loss =
        std::log(exp_sum) + max_logit - input[target];

    // d(loss)/d(logit) = probability - one_hot(target)
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

} // namespace neural
