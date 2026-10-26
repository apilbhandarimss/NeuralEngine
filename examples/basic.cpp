#include "neural/data/dataset.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void save_tensor(std::ofstream& file,
                 const char* name,
                 const neural::Tensor& tensor) {
    file << name << '\n';
    file << tensor.rank();

    for (const std::size_t dimension : tensor.shape()) {
        file << ' ' << dimension;
    }
    file << '\n';

    std::size_t count = 0;

    for (const float value : tensor.data()) {
        file << value << ' ';

        if (++count % 8 == 0) {
            file << '\n';
        }
    }

    if (count % 8 != 0) {
        file << '\n';
    }
}

void save_layer(std::ofstream& file,
                const char* name,
                const neural::Dense& layer) {
    file << "LAYER " << name << '\n';
    save_tensor(file, "WEIGHTS", layer.weights());
    save_tensor(file, "BIAS", layer.bias());
}

void save_model(const char* path,
                const neural::Dense& fc1,
                const neural::Dense& fc2,
                const neural::Dense& fc3,
                const neural::Dense& fc4,
                const neural::Dense& fc5) {
    std::ofstream file(path);

    if (!file) {
        throw std::runtime_error("Could not open model checkpoint");
    }

    file << std::setprecision(
        std::numeric_limits<float>::max_digits10
    );

    file << "NEURAL_ENGINE_CHECKPOINT 1\n";
    file << "ARCHITECTURE 784 512 256 128 64 10\n";

    save_layer(file, "fc1", fc1);
    save_layer(file, "fc2", fc2);
    save_layer(file, "fc3", fc3);
    save_layer(file, "fc4", fc4);
    save_layer(file, "fc5", fc5);

    if (!file) {
        throw std::runtime_error("Failed while writing model checkpoint");
    }
}

} // namespace

int main() {
    try {
        const neural::MNISTDataset training(
            "../data/mnist/train-images-idx3-ubyte",
            "../data/mnist/train-labels-idx1-ubyte"
        );

        const neural::MNISTDataset testing(
            "../data/mnist/t10k-images-idx3-ubyte",
            "../data/mnist/t10k-labels-idx1-ubyte"
        );

        std::cout << "Training images: " << training.size() << '\n';
        std::cout << "Test images:     " << testing.size() << '\n';

        // Network: 784 -> 512 -> 256 -> 128 -> 64 -> 10
        neural::Dense fc1(784, 512, neural::WeightInit::He, 101);
        neural::Dense fc2(512, 256, neural::WeightInit::He, 102);
        neural::Dense fc3(256, 128, neural::WeightInit::He, 103);
        neural::Dense fc4(128, 64, neural::WeightInit::He, 104);
        neural::Dense fc5(64, 10, neural::WeightInit::Xavier, 105);

        neural::ReLU relu1;
        neural::ReLU relu2;
        neural::ReLU relu3;
        neural::ReLU relu4;

        neural::SGD optimizer(0.01F);

        constexpr std::size_t epochs = 5;
        constexpr std::size_t report_interval = 5000;

        std::vector<std::size_t> indices(training.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(42);

        // Training
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);

            double interval_loss = 0.0;
            std::size_t interval_count = 0;

            std::cout << "\nEpoch " << epoch + 1 << '/' << epochs << '\n';

            for (std::size_t step = 0; step < indices.size(); ++step) {
                const std::size_t index = indices[step];

                const neural::Tensor input = training.image(index);
                const std::size_t target = training.label(index);

                // Forward propagation
                const neural::Tensor z1 = fc1.forward(input);
                const neural::Tensor a1 = relu1.forward(z1);

                const neural::Tensor z2 = fc2.forward(a1);
                const neural::Tensor a2 = relu2.forward(z2);

                const neural::Tensor z3 = fc3.forward(a2);
                const neural::Tensor a3 = relu3.forward(z3);

                const neural::Tensor z4 = fc4.forward(a3);
                const neural::Tensor a4 = relu4.forward(z4);

                const neural::Tensor logits = fc5.forward(a4);

                const neural::LossResult result =
                    neural::softmax_cross_entropy(logits, target);

                interval_loss += static_cast<double>(result.loss);
                ++interval_count;

                // Backpropagation
                neural::Tensor gradient =
                    fc5.backward(result.logits_gradient);

                gradient = relu4.backward(gradient);
                gradient = fc4.backward(gradient);

                gradient = relu3.backward(gradient);
                gradient = fc3.backward(gradient);

                gradient = relu2.backward(gradient);
                gradient = fc2.backward(gradient);

                gradient = relu1.backward(gradient);
                (void)fc1.backward(gradient);

                // Update only after all gradients have been calculated.
                optimizer.step(fc1);
                optimizer.step(fc2);
                optimizer.step(fc3);
                optimizer.step(fc4);
                optimizer.step(fc5);

                if ((step + 1) % report_interval == 0 ||
                    step + 1 == indices.size()) {
                    std::cout
                        << "  Samples: " << step + 1 << '/'
                        << indices.size()
                        << " | Mean loss: "
                        << interval_loss /
                               static_cast<double>(interval_count)
                        << '\n';

                    interval_loss = 0.0;
                    interval_count = 0;
                }
            }
        }

        // Save trained parameters as plain text.
        save_model(
            "../mnist_model.txt",
            fc1, fc2, fc3, fc4, fc5
        );

        std::cout << "\nModel saved to ../mnist_model.txt\n";

        // Evaluation: no parameter updates on test data.
        std::size_t correct = 0;
        double total_loss = 0.0;

        for (std::size_t i = 0; i < testing.size(); ++i) {
            const neural::Tensor input = testing.image(i);
            const std::size_t target = testing.label(i);

            const neural::Tensor z1 = fc1.forward(input);
            const neural::Tensor a1 = relu1.forward(z1);

            const neural::Tensor z2 = fc2.forward(a1);
            const neural::Tensor a2 = relu2.forward(z2);

            const neural::Tensor z3 = fc3.forward(a2);
            const neural::Tensor a3 = relu3.forward(z3);

            const neural::Tensor z4 = fc4.forward(a3);
            const neural::Tensor a4 = relu4.forward(z4);

            const neural::Tensor logits = fc5.forward(a4);

            const auto result =
                neural::softmax_cross_entropy(logits, target);

            total_loss += static_cast<double>(result.loss);

            const auto predicted_it = std::max_element(
                logits.data().begin(),
                logits.data().end()
            );

            const auto predicted = static_cast<std::size_t>(
                std::distance(logits.data().begin(), predicted_it)
            );

            if (predicted == target) {
                ++correct;
            }

            if ((i + 1) % 1000 == 0) {
                std::cout << "Evaluated " << i + 1 << '/'
                          << testing.size() << " test images\n";
            }
        }

        const double accuracy =
            100.0 * static_cast<double>(correct) /
            static_cast<double>(testing.size());

        std::cout << "\nTraining complete.\n"
                  << "Test accuracy: " << accuracy << "% ("
                  << correct << '/' << testing.size() << ")\n"
                  << "Mean test loss: "
                  << total_loss / static_cast<double>(testing.size())
                  << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}