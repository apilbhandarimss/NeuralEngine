#include "neural/tensor/tensor.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    neural::Tensor tensor({2, 3});
    assert(tensor.size() == 6);
    assert(tensor.shape().size() == 2);

    tensor.at({1, 2}) = 7.0F;
    assert(tensor.at({1, 2}) == 7.0F);

    bool threw = false;
    try {
        (void)tensor.at({2, 0});
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    return 0;
}
