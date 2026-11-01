// Change kBatchSize, kEpochs and kLearningRate below, then rebuild.

#include "neural/data/dataset.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kInputs = 784;
constexpr std::size_t kClasses = 10;
constexpr std::size_t kMaxBatch = 4096;

// ---- Settings: change these ----
constexpr std::size_t kBatchSize = 2;   // 1 = update after every sample
constexpr std::size_t kEpochs = 1;
constexpr float kLearningRate = 0.001F;  // use ~0.0001 for batch size 1

static_assert(kBatchSize >= 1 && kBatchSize <= kMaxBatch, "kBatchSize must be in [1, 4096]");
static_assert(kEpochs >= 1, "kEpochs must be at least 1");
static_assert(kLearningRate > 0.0F && kLearningRate <= 1.0F, "kLearningRate must be in (0, 1]");

}  // namespace

int main() {
    try {
        const std::size_t batch_size = kBatchSize;
        const std::size_t epochs = kEpochs;
        const float lr = kLearningRate;

        std::cout << "batch_size=" << batch_size << " epochs=" << epochs << " lr=" << lr << '\n';

        neural::MNISTDataset training(
            "../data/mnist/train-images-idx3-ubyte",
            "../data/mnist/train-labels-idx1-ubyte"
        );

        neural::MNISTDataset testing(
            "../data/mnist/t10k-images-idx3-ubyte",
            "../data/mnist/t10k-labels-idx1-ubyte"
        );

        neural::Dense fc1(784, 512, neural::WeightInit::He, 101);
        neural::Dense fc2(512, 256, neural::WeightInit::He, 102);
        neural::Dense fc3(256, 128, neural::WeightInit::He, 103);
        neural::Dense fc4(128, 64, neural::WeightInit::He, 104);
        neural::Dense fc5(64, 10, neural::WeightInit::Xavier, 105);

        neural::ReLU relu1, relu2, relu3, relu4;
        neural::RMSprop optimizer(lr);

        std::vector<std::size_t> indices(training.size());
        std::iota(indices.begin(), indices.end(), std::size_t{0});

        std::mt19937 rng(42);

        const std::size_t batches_per_epoch = (indices.size() + batch_size - 1) / batch_size;
        const std::size_t log_every = std::max<std::size_t>(1, batches_per_epoch / 10);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);

            double window_loss = 0.0;
            std::size_t window_samples = 0;
            std::size_t batch_index = 0;

            for (std::size_t start = 0; start < indices.size(); start += batch_size) {
                // The last batch of an epoch may be smaller than batch_size.
                const std::size_t count = std::min(batch_size, indices.size() - start);

                neural::Tensor input({count, kInputs}, 0.0F);
                std::vector<std::size_t> targets(count);

                for (std::size_t b = 0; b < count; ++b) {
                    const auto image = training.image(indices[start + b]);
                    targets[b] = training.label(indices[start + b]);
                    std::copy(image.data().begin(), image.data().end(),
                              input.data().begin() + b * kInputs);
                }

                // Forward
                const auto a1 = relu1.forward(fc1.forward(input));
                const auto a2 = relu2.forward(fc2.forward(a1));
                const auto a3 = relu3.forward(fc3.forward(a2));
                const auto a4 = relu4.forward(fc4.forward(a3));
                const auto logits = fc5.forward(a4);

                // Per-sample loss, gradients averaged over the actual batch size
                neural::Tensor output_gradient({count, kClasses}, 0.0F);
                const float inv = 1.0F / static_cast<float>(count);

                for (std::size_t b = 0; b < count; ++b) {
                    neural::Tensor sample_logits({kClasses}, 0.0F);
                    std::copy(logits.data().begin() + b * kClasses,
                              logits.data().begin() + (b + 1) * kClasses,
                              sample_logits.data().begin());

                    const auto result = neural::softmax_cross_entropy(sample_logits, targets[b]);
                    window_loss += static_cast<double>(result.loss);
                    ++window_samples;

                    for (std::size_t j = 0; j < kClasses; ++j) {
                        output_gradient.data()[b * kClasses + j] = result.logits_gradient.data()[j] * inv;
                    }
                }

                if (!std::isfinite(window_loss)) {
                    throw std::runtime_error("Loss became non-finite; lower the learning rate");
                }

                // Backward
                auto gradient = fc5.backward(output_gradient);
                gradient = relu4.backward(gradient);
                gradient = fc4.backward(gradient);
                gradient = relu3.backward(gradient);
                gradient = fc3.backward(gradient);
                gradient = relu2.backward(gradient);
                gradient = fc2.backward(gradient);
                gradient = relu1.backward(gradient);
                gradient = fc1.backward(gradient);

                // One update per batch
                optimizer.step(fc1);
                optimizer.step(fc2);
                optimizer.step(fc3);
                optimizer.step(fc4);
                optimizer.step(fc5);

                ++batch_index;
                if (batch_index % log_every == 0) {
                    std::cout << "epoch " << epoch + 1 << "/" << epochs
                              << "  batch " << batch_index << "/" << batches_per_epoch
                              << "  avg loss " << window_loss / static_cast<double>(window_samples) << '\n';
                    window_loss = 0.0;
                    window_samples = 0;
                }
            }
        }

        // Evaluation, one sample at a time
        std::size_t correct = 0;

        for (std::size_t i = 0; i < testing.size(); ++i) {
            const auto input = testing.image(i);
            const auto target = testing.label(i);

            const auto a1 = relu1.forward(fc1.forward(input));
            const auto a2 = relu2.forward(fc2.forward(a1));
            const auto a3 = relu3.forward(fc3.forward(a2));
            const auto a4 = relu4.forward(fc4.forward(a3));
            const auto logits = fc5.forward(a4);

            const auto predicted = static_cast<std::size_t>(
                std::distance(logits.data().begin(),
                              std::max_element(logits.data().begin(), logits.data().end())));

            if (predicted == target) ++correct;
        }

        const double accuracy =
            100.0 * static_cast<double>(correct) / static_cast<double>(testing.size());

        std::cout << "\nAccuracy: " << accuracy << "%\n";
        std::cout << "Correct predictions: " << correct << "/" << testing.size() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}