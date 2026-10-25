#pragma once

#include "neural/core/device.hpp"

namespace neural {

// Backend owns device-specific execution and memory policies.
// Do not put CPU/CUDA conditionals throughout layers and operations.
class Backend {
public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual Device device() const noexcept = 0;
};

} // namespace neural
