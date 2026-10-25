
#include "neural/ops/activations.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

using namespace neural;

void check(
    const char* name,
    const Tensor& result,
    const float* expected,
    std::size_t count
) {
    if (result.size() != count) {
        throw std::runtime_error("Incorrect output size");
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (std::abs(result.data()[i] - expected[i]) > 1e-5F) {
            throw std::runtime_error(
                std::string(name) + ": incorrect result"
            );
        }
    }

    std::cout << "[PASS] " << name << '\n';
}

int main() {
    const Tensor input(
        std::vector<std::size_t>{5},
        0.0F
    );

    // Fill the tensor through its mutable span.
    auto values = const_cast<Tensor&>(input).data();
    values[0] = -2.0F;
    values[1] = -1.0F;
    values[2] = 0.0F;
    values[3] = 1.0F;
    values[4] = 2.0F;

    const float relu_expected[] = {0, 0, 0, 1, 2};
    const float sigmoid_expected[] = {
        0.1192029F, 0.2689414F, 0.5F,
        0.7310586F, 0.8807971F
    };
    const float tanh_expected[] = {
        -0.9640276F, -0.7615942F, 0,
        0.7615942F, 0.9640276F
    };

    check("ReLU", ReLU{}.forward(input), relu_expected, 5);
    check("Sigmoid", Sigmoid{}.forward(input), sigmoid_expected, 5);
    check("Tanh", Tanh{}.forward(input), tanh_expected, 5);

    std::cout << "All activation tests passed.\n";
}
