
#include "neural/ops/arithmetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

using neural::Tensor;
using neural::matmul;

Tensor make_tensor(
    std::vector<std::size_t> shape,
    std::initializer_list<float> values
) {
    Tensor tensor(std::move(shape), 0.0F);

    if (tensor.size() != values.size()) {
        throw std::invalid_argument("Incorrect number of tensor values");
    }

    std::copy(values.begin(), values.end(), tensor.data().begin());
    return tensor;
}

void check_tensor(
    const char* name,
    const Tensor& actual,
    const std::vector<std::size_t>& expected_shape,
    std::initializer_list<float> expected_values
) {
    if (actual.shape() != expected_shape) {
        throw std::runtime_error(
            std::string(name) + ": incorrect output shape"
        );
    }

    if (actual.size() != expected_values.size()) {
        throw std::runtime_error(
            std::string(name) + ": incorrect number of output values"
        );
    }

    std::size_t i = 0;

    for (const float expected : expected_values) {
        if (std::abs(actual.data()[i] - expected) > 1e-5F) {
            throw std::runtime_error(
                std::string(name) + ": incorrect value at index "
                + std::to_string(i)
            );
        }
        ++i;
    }

    std::cout << "[PASS] " << name << '\n';
}

int main() {
    // 1. Ordinary matrix multiplication: (2, 3) x (3, 2).
    const Tensor a = make_tensor(
        {2, 3},
        {1, 2, 3,
         4, 5, 6}
    );

    const Tensor b = make_tensor(
        {3, 2},
        {7, 8,
         9, 10,
         11, 12}
    );

    check_tensor(
        "Matrix multiplication",
        matmul(a, b),
        {2, 2},
        {58, 64, 139, 154}
    );

    // 2. Vector x matrix: (3) x (3, 2).
    const Tensor v = make_tensor({3}, {1, 2, 3});

    check_tensor(
        "Vector-matrix multiplication",
        matmul(v, b),
        {2},
        {58, 64}
    );

    // 3. Matrix x vector: (2, 3) x (3).
    check_tensor(
        "Matrix-vector multiplication",
        matmul(a, v),
        {2},
        {14, 32}
    );

    // 4. Vector dot product: (3) x (3) -> (1).
    const Tensor w = make_tensor({3}, {4, 5, 6});

    check_tensor(
        "Vector dot product",
        matmul(v, w),
        {1},
        {32}
    );

    // 5. Batched multiplication: (2, 2, 3) x (2, 3, 2).
    const Tensor batched_a = make_tensor(
        {2, 2, 3},
        {1, 2, 3,
         4, 5, 6,

         2, 0, 1,
         1, 3, 2}
    );

    const Tensor batched_b = make_tensor(
        {2, 3, 2},
        {7, 8,
         9, 10,
         11, 12,

         7, 8,
         9, 10,
         11, 12}
    );

    check_tensor(
        "Batched multiplication",
        matmul(batched_a, batched_b),
        {2, 2, 2},
        {58, 64, 139, 154,
         25, 28, 56, 62}
    );

    // 6. Batch broadcasting: (2, 2, 3) x (1, 3, 2).
    const Tensor single_batch = make_tensor(
        {1, 3, 2},
        {7, 8,
         9, 10,
         11, 12}
    );

    check_tensor(
        "Batch broadcasting",
        matmul(batched_a, single_batch),
        {2, 2, 2},
        {58, 64, 139, 154,
         25, 28, 56, 62}
    );

    // 7. Incompatible matrix dimensions must throw.
    bool threw = false;

    try {
        const Tensor invalid = make_tensor({2, 2}, {1, 2, 3, 4});
        (void)matmul(a, invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    if (!threw) {
        throw std::runtime_error(
            "Incompatible dimensions were not rejected"
        );
    }

    std::cout << "[PASS] Invalid dimensions rejected\n";
    std::cout << "\nAll tests passed.\n";

    return 0;
}
