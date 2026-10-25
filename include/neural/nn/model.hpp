#pragma once

#include "neural/nn/layer.hpp"

#include <memory>
#include <vector>

namespace neural {

// Model composes layers. Ownership is explicit: Model owns its layers.
class Model {
public:
    void add(std::unique_ptr<Layer> layer);
    [[nodiscard]] Tensor forward(const Tensor& input) const;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace neural
