// compile with: -O3 -march=native -fopenmp -fno-math-errno
// (do NOT add -ffast-math: it breaks the std::isfinite checks below)
#include "neural/ops/loss.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace neural {

namespace {

constexpr int kThreads = 14;
// exp() makes each element fairly expensive, so the cutoff is lower than
// for pure element-wise ops.
constexpr std::size_t kParallelMin = 4096;

} // namespace

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

    const std::size_t n = logits.size();
    const float* input = logits.data().data();

    float max_logit = input[0];
    int nonfinite = 0;

    #pragma omp simd reduction(max : max_logit) reduction(| : nonfinite)
    for (std::size_t i = 0; i < n; ++i) {
        const float value = input[i];

        nonfinite |= !std::isfinite(value);
        max_logit = value > max_logit ? value : max_logit;
    }

    if (nonfinite) {
        throw std::invalid_argument("Logits must be finite");
    }

    Tensor probabilities({n}, 0.0F);
    Tensor gradient({n}, 0.0F);

    float* probs = probabilities.data().data();
    float* grad = gradient.data().data();

    float exp_sum = 0.0F;

    #pragma omp simd reduction(+ : exp_sum)
    for (std::size_t i = 0; i < n; ++i) {
        const float value = std::exp(input[i] - max_logit);

        probs[i] = value;
        exp_sum += value;
    }

    const float inv_sum = 1.0F / exp_sum;

    // Normalise and copy into the gradient in one pass.
    #pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        const float probability = probs[i] * inv_sum;

        probs[i] = probability;
        grad[i] = probability;
    }

    const float loss =
        std::log(exp_sum) + max_logit - input[target];

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

    const float* input = logits.data().data();
    const float* target_values = targets.data().data();

    float* probs = probabilities.data().data();
    float* grad = gradient.data().data();

    // Mean-batch-loss scale, applied while the gradient is written
    // (no separate pass over the whole gradient afterwards).
    const float scale = 1.0F / static_cast<float>(batch_size);
    const float class_count = static_cast<float>(num_classes);

    float total_loss = 0.0F;
    int bad_target = 0;
    int bad_logit = 0;

    // Exceptions must not escape an OpenMP region, so errors are recorded
    // in flags (OR-reduced across threads) and thrown after the loop.
    // Each row b owns probs/grad[b * C .. (b + 1) * C), so there are no races;
    // total_loss is combined with a reduction.
    #pragma omp parallel for schedule(static) num_threads(kThreads) \
        if (batch_size * num_classes > kParallelMin) \
        reduction(+ : total_loss) reduction(| : bad_target, bad_logit)
    for (std::size_t b = 0; b < batch_size; ++b) {
        const float target_value = target_values[b];

        if (!std::isfinite(target_value) ||
            target_value < 0.0F ||
            target_value >= class_count ||
            std::floor(target_value) != target_value) {
            bad_target = 1;
            continue;
        }

        const std::size_t target =
            static_cast<std::size_t>(target_value);

        const std::size_t offset = b * num_classes;

        const float* in = input + offset;
        float* p = probs + offset;
        float* g = grad + offset;

        float max_logit = in[0];
        int nonfinite = 0;

        #pragma omp simd reduction(max : max_logit) reduction(| : nonfinite)
        for (std::size_t j = 0; j < num_classes; ++j) {
            const float value = in[j];

            nonfinite |= !std::isfinite(value);
            max_logit = value > max_logit ? value : max_logit;
        }

        if (nonfinite) {
            bad_logit = 1;
            continue;
        }

        float exp_sum = 0.0F;

        #pragma omp simd reduction(+ : exp_sum)
        for (std::size_t j = 0; j < num_classes; ++j) {
            const float value = std::exp(in[j] - max_logit);

            p[j] = value;
            exp_sum += value;
        }

        const float inv_sum = 1.0F / exp_sum;

        // Normalise, store probabilities, and write the scaled gradient.
        #pragma omp simd
        for (std::size_t j = 0; j < num_classes; ++j) {
            const float probability = p[j] * inv_sum;

            p[j] = probability;
            g[j] = probability * scale;
        }

        total_loss +=
            std::log(exp_sum) + max_logit - in[target];

        g[target] -= scale;
    }

    if (bad_target) {
        throw std::invalid_argument(
            "Each target must be a valid integer class index"
        );
    }

    if (bad_logit) {
        throw std::invalid_argument("Logits must be finite");
    }

    return {
        total_loss * scale,
        std::move(gradient),
        std::move(probabilities)
    };
}

} // namespace neural