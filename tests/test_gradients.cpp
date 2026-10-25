#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

constexpr float epsilon = 1.0e-3F;
constexpr float tolerance = 2.0e-3F;

void check_gradient(
    const char* name,
    float analytical,
    float numerical
) {
    const float error = std::abs(analytical - numerical);

    if (error > tolerance) {
        std::cerr << "FAIL: " << name
                  << " analytical=" << analytical
                  << " numerical=" << numerical
                  << " error=" << error << '\n';
        throw std::runtime_error("Gradient check failed");
    }

    std::cout << "PASS: " << name
              << " analytical=" << analytical
              << " numerical=" << numerical << '\n';
}

float loss_for_dense_weight(
    neural::Dense& dense,
    const neural::Tensor& input,
    const neural::Tensor& output_gradient
) {
    const neural::Tensor output = dense.forward(input);
    float loss = 0.0F;

    for (std::size_t i = 0; i < output.size(); ++i) {
        loss += output.data()[i] * output_gradient.data()[i];
    }

    return loss;
}

void check_dense_weight_gradient() {
    neural::Dense dense(2, 2, neural::WeightInit::Xavier, 123U);

    auto weights = dense.weights().data();
    weights[0] = 0.3F;
    weights[1] = -0.2F;
    weights[2] = 0.4F;
    weights[3] = 0.1F;

    neural::Tensor input({2}, 0.0F);
    input.data()[0] = 0.7F;
    input.data()[1] = -0.5F;

    neural::Tensor output_gradient({2}, 0.0F);
    output_gradient.data()[0] = 0.6F;
    output_gradient.data()[1] = -0.8F;

    // Analytical gradients from backpropagation.
    (void)dense.forward(input);
    (void)dense.backward(output_gradient);

    const float analytical =
        dense.weight_gradients().data()[0];

    // Numerical gradient for weights[0].
    const float original = dense.weights().data()[0];

    dense.weights().data()[0] = original + epsilon;
    const float loss_plus =
        loss_for_dense_weight(dense, input, output_gradient);

    dense.weights().data()[0] = original - epsilon;
    const float loss_minus =
        loss_for_dense_weight(dense, input, output_gradient);

    dense.weights().data()[0] = original;

    const float numerical =
        (loss_plus - loss_minus) / (2.0F * epsilon);

    check_gradient("Dense weight", analytical, numerical);
}

void check_relu_gradient() {
    neural::ReLU relu;

    neural::Tensor input({3}, 0.0F);
    input.data()[0] = -1.0F;
    input.data()[1] = 0.5F;
    input.data()[2] = 2.0F;

    neural::Tensor upstream({3}, 0.0F);
    upstream.data()[0] = 0.3F;
    upstream.data()[1] = -0.7F;
    upstream.data()[2] = 0.2F;

    (void)relu.forward(input);
    const neural::Tensor analytical =
        relu.backward(upstream);

    for (std::size_t i = 0; i < input.size(); ++i) {
        const float original = input.data()[i];

        input.data()[i] = original + epsilon;
        const neural::Tensor plus = relu.forward(input);
        float loss_plus = 0.0F;
        for (std::size_t j = 0; j < input.size(); ++j) {
            loss_plus += plus.data()[j] * upstream.data()[j];
        }

        input.data()[i] = original - epsilon;
        const neural::Tensor minus = relu.forward(input);
        float loss_minus = 0.0F;
        for (std::size_t j = 0; j < input.size(); ++j) {
            loss_minus += minus.data()[j] * upstream.data()[j];
        }

        input.data()[i] = original;

        const float numerical =
            (loss_plus - loss_minus) / (2.0F * epsilon);

        check_gradient("ReLU input", analytical.data()[i], numerical);
    }
}

void check_softmax_cross_entropy_gradient() {
    neural::Tensor logits({3}, 0.0F);
    logits.data()[0] = 0.2F;
    logits.data()[1] = -0.4F;
    logits.data()[2] = 0.8F;

    constexpr std::size_t target = 2;

    const neural::LossResult result =
        neural::softmax_cross_entropy(logits, target);

    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float original = logits.data()[i];

        logits.data()[i] = original + epsilon;
        const float loss_plus =
            neural::softmax_cross_entropy(logits, target).loss;

        logits.data()[i] = original - epsilon;
        const float loss_minus =
            neural::softmax_cross_entropy(logits, target).loss;

        logits.data()[i] = original;

        const float numerical =
            (loss_plus - loss_minus) / (2.0F * epsilon);

        check_gradient(
            "Cross-entropy logits",
            result.logits_gradient.data()[i],
            numerical
        );
    }
}

} // namespace

int main() {
    try {
        check_dense_weight_gradient();
        check_relu_gradient();
        check_softmax_cross_entropy_gradient();

        std::cout << "\nAll gradient checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}