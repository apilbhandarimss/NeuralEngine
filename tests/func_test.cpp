#include "neural/tensor/tensor.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("FAILED: " + message);
    }
}

void check_close(
    float actual,
    float expected,
    const std::string& message,
    float tolerance = 1.0e-5F
) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance) {
        std::cerr << "FAILED: " << message
                  << " | expected " << expected
                  << ", got " << actual << '\n';
        throw std::runtime_error(message);
    }
}

void test_tensor() {
    neural::Tensor tensor({2, 3}, 0.0F);

    check(tensor.rank() == 2, "Tensor rank");
    check(tensor.size() == 6, "Tensor element count");
    check(tensor.shape()[0] == 2, "Tensor first dimension");
    check(tensor.shape()[1] == 3, "Tensor second dimension");

    tensor.at({1, 2}) = 7.0F;
    check_close(tensor.data()[5], 7.0F, "Tensor row-major indexing");

    std::cout << "[PASS] Tensor construction and indexing\n";
}

void test_dense_forward_backward_and_sgd() {
    neural::Dense dense(2, 2, neural::WeightInit::Xavier, 123U);

    // Set known parameters so the expected results are deterministic.
    auto weights = dense.weights().data();
    weights[0] = 1.0F;
    weights[1] = 2.0F;
    weights[2] = -1.0F;
    weights[3] = 0.5F;

    auto biases = dense.bias().data();
    biases[0] = 0.5F;
    biases[1] = -1.0F;

    neural::Tensor input({2}, 0.0F);
    input.data()[0] = 2.0F;
    input.data()[1] = 3.0F;

    const neural::Tensor output = dense.forward(input);

    check_close(output.data()[0], 8.5F, "Dense forward output 0");
    check_close(output.data()[1], -1.5F, "Dense forward output 1");

    neural::Tensor output_gradient({2}, 0.0F);
    output_gradient.data()[0] = 0.1F;
    output_gradient.data()[1] = -0.2F;

    const neural::Tensor input_gradient = dense.backward(output_gradient);

    check_close(input_gradient.data()[0], 0.3F, "Dense input gradient 0");
    check_close(input_gradient.data()[1], 0.1F, "Dense input gradient 1");

    const auto weight_gradients = dense.weight_gradients().data();
    check_close(weight_gradients[0], 0.2F, "Weight gradient 0");
    check_close(weight_gradients[1], 0.3F, "Weight gradient 1");
    check_close(weight_gradients[2], -0.4F, "Weight gradient 2");
    check_close(weight_gradients[3], -0.6F, "Weight gradient 3");

    const auto bias_gradients = dense.bias_gradients().data();
    check_close(bias_gradients[0], 0.1F, "Bias gradient 0");
    check_close(bias_gradients[1], -0.2F, "Bias gradient 1");

    neural::SGD optimizer(0.1F);
    optimizer.step(dense);

    const auto updated_weights = dense.weights().data();
    check_close(updated_weights[0], 0.98F, "SGD weight update 0");
    check_close(updated_weights[1], 1.97F, "SGD weight update 1");
    check_close(updated_weights[2], -0.96F, "SGD weight update 2");
    check_close(updated_weights[3], 0.56F, "SGD weight update 3");

    const auto updated_biases = dense.bias().data();
    check_close(updated_biases[0], 0.49F, "SGD bias update 0");
    check_close(updated_biases[1], -0.98F, "SGD bias update 1");

    std::cout << "[PASS] Dense forward, backward, and SGD updates\n";
}

void test_relu() {
    neural::ReLU relu;

    neural::Tensor input({3}, 0.0F);
    input.data()[0] = -1.0F;
    input.data()[1] = 2.0F;
    input.data()[2] = 0.0F;

    const neural::Tensor output = relu.forward(input);

    check_close(output.data()[0], 0.0F, "ReLU negative input");
    check_close(output.data()[1], 2.0F, "ReLU positive input");
    check_close(output.data()[2], 0.0F, "ReLU zero input");

    neural::Tensor gradient({3}, 1.0F);
    const neural::Tensor input_gradient = relu.backward(gradient);

    check_close(input_gradient.data()[0], 0.0F, "ReLU negative gradient");
    check_close(input_gradient.data()[1], 1.0F, "ReLU positive gradient");
    check_close(input_gradient.data()[2], 0.0F, "ReLU zero gradient");

    std::cout << "[PASS] ReLU forward and backward\n";
}

void test_sigmoid_and_tanh() {
    neural::Tensor input({3}, 0.0F);
    input.data()[0] = -1.0F;
    input.data()[1] = 0.0F;
    input.data()[2] = 1.0F;

    neural::Sigmoid sigmoid;
    const neural::Tensor sigmoid_output = sigmoid.forward(input);

    check_close(sigmoid_output.data()[0], 1.0F / (1.0F + std::exp(1.0F)),
                "Sigmoid(-1)");
    check_close(sigmoid_output.data()[1], 0.5F, "Sigmoid(0)");
    check_close(sigmoid_output.data()[2], 1.0F / (1.0F + std::exp(-1.0F)),
                "Sigmoid(1)");

    neural::Tensor gradient({3}, 1.0F);
    const neural::Tensor sigmoid_gradient = sigmoid.backward(gradient);

    for (const float value : sigmoid_gradient.data()) {
        check(std::isfinite(value), "Sigmoid gradient must be finite");
    }

    neural::Tanh tanh;
    const neural::Tensor tanh_output = tanh.forward(input);

    check_close(tanh_output.data()[0], std::tanh(-1.0F), "Tanh(-1)");
    check_close(tanh_output.data()[1], 0.0F, "Tanh(0)");
    check_close(tanh_output.data()[2], std::tanh(1.0F), "Tanh(1)");

    const neural::Tensor tanh_gradient = tanh.backward(gradient);

    for (const float value : tanh_gradient.data()) {
        check(std::isfinite(value), "Tanh gradient must be finite");
    }

    std::cout << "[PASS] Sigmoid and Tanh forward/backward\n";
}

void test_softmax_cross_entropy() {
    neural::Tensor logits({3}, 0.0F);
    logits.data()[0] = 2.0F;
    logits.data()[1] = 1.0F;
    logits.data()[2] = 0.0F;

    const neural::LossResult result =
        neural::softmax_cross_entropy(logits, 0);

    check(std::isfinite(result.loss), "Cross-entropy loss must be finite");
    check(result.loss > 0.0F, "Cross-entropy loss must be positive");

    const auto probabilities = result.probabilities.data();
    check_close(
        probabilities[0] + probabilities[1] + probabilities[2],
        1.0F,
        "Softmax probabilities sum to one"
    );

    check(probabilities[0] > probabilities[1],
          "Higher logit should have higher probability");
    check(probabilities[1] > probabilities[2],
          "Logit ordering should be preserved");

    const auto gradient = result.logits_gradient.data();
    check_close(gradient[0], probabilities[0] - 1.0F,
                "Cross-entropy gradient for target");
    check_close(gradient[1], probabilities[1],
                "Cross-entropy gradient for non-target 1");
    check_close(gradient[2], probabilities[2],
                "Cross-entropy gradient for non-target 2");
    check_close(gradient[0] + gradient[1] + gradient[2], 0.0F,
                "Cross-entropy gradient sums to zero");

    std::cout << "[PASS] Softmax cross-entropy and gradients\n";
}

} // namespace

int main() {
    try {
        test_tensor();
        test_dense_forward_backward_and_sgd();
        test_relu();
        test_sigmoid_and_tanh();
        test_softmax_cross_entropy();

        std::cout << "\nAll basic integration checks passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "\nTest run stopped: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}