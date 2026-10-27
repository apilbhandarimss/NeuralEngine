#include "neural/data/dataset.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main() {
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

    neural::Sigmoid relu1, relu2, relu3, relu4;
    neural::SGD optimizer(0.01F);

    std::vector<std::size_t> indices(training.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
int n=0;
    for (std::size_t epoch = 0; epoch < 1; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (const auto index : indices) {
            n++;
            const auto input = training.image(index);
            const auto target = training.label(index);

            const auto a1 = relu1.forward(fc1.forward(input));
            const auto a2 = relu2.forward(fc2.forward(a1));
            const auto a3 = relu3.forward(fc3.forward(a2));
            const auto a4 = relu4.forward(fc4.forward(a3));
            const auto logits = fc5.forward(a4);

            const auto result =
                neural::softmax_cross_entropy(logits, target);

            auto gradient = fc5.backward(result.logits_gradient);
            gradient = relu4.backward(gradient);
            gradient = fc4.backward(gradient);
            gradient = relu3.backward(gradient);
            gradient = fc3.backward(gradient);
            gradient = relu2.backward(gradient);
            gradient = fc2.backward(gradient);
            gradient = relu1.backward(gradient);
            gradient = fc1.backward(gradient);

            optimizer.step(fc1);
            optimizer.step(fc2);
            optimizer.step(fc3);
            optimizer.step(fc4);
            optimizer.step(fc5);
            if(n%1000==0)
            std::cout<<"Number of Cheaks done :"<<n<<std::endl;

        }

    }

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
            std::distance(
                logits.data().begin(),
                std::max_element(
                    logits.data().begin(),
                    logits.data().end()
                )
            )
        );
        std::cout<<"Number of Cheaks done :"<<i<<std::endl;

        if (predicted == target)
            ++correct;
    }

    const double accuracy =
        100.0 * static_cast<double>(correct) /
        static_cast<double>(testing.size());
        std::cout<<accuracy;
}