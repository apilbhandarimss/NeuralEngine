
#include "neural/data/dataset.hpp"
#include "neural/nn/batch_norm.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"
#include "neural/tensor/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main() {
    constexpr std::size_t input_features = 784;
    constexpr std::size_t output_features = 10;
    constexpr std::size_t batch_size = 32;
    constexpr std::size_t epochs = 1;

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

    neural::BatchNorm bn1(512);
    neural::BatchNorm bn2(256);
    neural::BatchNorm bn3(128);
    neural::BatchNorm bn4(64);

    neural::ReLU relu1, relu2, relu3, relu4;

    neural::SGD optimizer(0.1F);

    std::vector<std::size_t> indices(training.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double total_loss = 0.0;
        std::size_t processed = 0;

        for (std::size_t start = 0;
             start < indices.size();
             start += batch_size) {

            const std::size_t current_batch =
                std::min(batch_size, indices.size() - start);

            neural::Tensor inputs({current_batch, input_features});
            neural::Tensor targets({current_batch});

            // Collect examples into one batch.
            for (std::size_t i = 0; i < current_batch; ++i) {
                const auto image = training.image(indices[start + i]);

                for (std::size_t j = 0; j < input_features; ++j) {
                    inputs.at({i, j}) = image.at({j});
                }

                targets.at({i}) =
                    static_cast<float>(training.label(indices[start + i]));
            }

            // Forward pass: Dense -> BatchNorm -> activation.
            const auto z1 = fc1.forward(inputs, true);
            const auto n1 = bn1.forward(z1, true);
            const auto a1 = relu1.forward(n1);

            const auto z2 = fc2.forward(a1, true);
            const auto n2 = bn2.forward(z2, true);
            const auto a2 = relu2.forward(n2);

            const auto z3 = fc3.forward(a2, true);
            const auto n3 = bn3.forward(z3, true);
            const auto a3 = relu3.forward(n3);

            const auto z4 = fc4.forward(a3, true);
            const auto n4 = bn4.forward(z4, true);
            const auto a4 = relu4.forward(n4);

            const auto logits = fc5.forward(a4, true);

            const auto result =
                neural::softmax_cross_entropy(logits, targets);

            total_loss +=
                static_cast<double>(result.loss) *
                static_cast<double>(current_batch);

            // Backward pass in reverse order.
            auto gradient = fc5.backward(result.logits_gradient);

            gradient = relu4.backward(gradient);
            gradient = bn4.backward(gradient);
            gradient = fc4.backward(gradient);

            gradient = relu3.backward(gradient);
            gradient = bn3.backward(gradient);
            gradient = fc3.backward(gradient);

            gradient = relu2.backward(gradient);
            gradient = bn2.backward(gradient);
            gradient = fc2.backward(gradient);

            gradient = relu1.backward(gradient);
            gradient = bn1.backward(gradient);
            gradient = fc1.backward(gradient);

            // One parameter update per batch.
            optimizer.step(fc1);
            optimizer.step(fc2);
            optimizer.step(fc3);
            optimizer.step(fc4);
            optimizer.step(fc5);

            optimizer.step(bn1);
            optimizer.step(bn2);
            optimizer.step(bn3);
            optimizer.step(bn4);

            processed += current_batch;

            if (processed % 1000 < current_batch ||
                processed == training.size()) {
                std::cout
                    << "Epoch " << (epoch + 1)
                    << " | Samples: " << processed
                    << "/" << training.size()
                    << " | Mean loss: "
                    << total_loss / static_cast<double>(processed)
                    << '\n';
            }
        }
    }

    // Evaluation uses running BatchNorm statistics.
    std::size_t correct = 0;

    for (std::size_t start = 0;
         start < testing.size();
         start += batch_size) {

        const std::size_t current_batch =
            std::min(batch_size, testing.size() - start);

        neural::Tensor inputs({current_batch, input_features});

        for (std::size_t i = 0; i < current_batch; ++i) {
            const auto image = testing.image(start + i);

            for (std::size_t j = 0; j < input_features; ++j) {
                inputs.at({i, j}) = image.at({j});
            }
        }

        const auto z1 = fc1.forward(inputs, false);
        const auto n1 = bn1.forward(z1, false);
        const auto a1 = relu1.forward(n1);

        const auto z2 = fc2.forward(a1, false);
        const auto n2 = bn2.forward(z2, false);
        const auto a2 = relu2.forward(n2);

        const auto z3 = fc3.forward(a2, false);
        const auto n3 = bn3.forward(z3, false);
        const auto a3 = relu3.forward(n3);

        const auto z4 = fc4.forward(a3, false);
        const auto n4 = bn4.forward(z4, false);
        const auto a4 = relu4.forward(n4);

        const auto logits = fc5.forward(a4, false);

        for (std::size_t i = 0; i < current_batch; ++i) {
            std::size_t predicted = 0;
            float highest_logit = logits.at({i, 0});

            for (std::size_t j = 1; j < output_features; ++j) {
                const float value = logits.at({i, j});

                if (value > highest_logit) {
                    highest_logit = value;
                    predicted = j;
                }
            }

            if (predicted == testing.label(start + i)) {
                ++correct;
            }
        }

        std::cout << "\rEvaluated "
                  << (start + current_batch)
                  << "/" << testing.size()
                  << std::flush;
    }

    const double accuracy =
        100.0 * static_cast<double>(correct) /
        static_cast<double>(testing.size());

    std::cout << "\nTest accuracy: " << accuracy << "%\n";
    std::cout << "Correct predictions: "
              << correct << "/" << testing.size() << '\n';

    return 0;
}
