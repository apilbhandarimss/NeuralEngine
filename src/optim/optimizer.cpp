#include "neural/optim/optimizer.hpp"

#include <cmath>
#include <stdexcept>

namespace neural {

SGD::SGD(float learning_rate)
    : learning_rate_(learning_rate) {
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) {
        throw std::invalid_argument(
            "Learning rate must be finite and positive"
        );
    }
}

void SGD::step(Dense& layer) const {
    auto weights = layer.weights().data();
    const auto weight_gradients = layer.weight_gradients().data();

    auto biases = layer.bias().data();
    const auto bias_gradients = layer.bias_gradients().data();

    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate_ * weight_gradients[i];
    }

    for (std::size_t i = 0; i < biases.size(); ++i) {
        biases[i] -= learning_rate_ * bias_gradients[i];
    }
}

float SGD::learning_rate() const noexcept {
    return learning_rate_;
}

}