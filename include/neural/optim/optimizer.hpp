#pragma once

#include "neural/nn/layer.hpp"

namespace neural {

class SGD {
public:
    explicit SGD(float learning_rate);

    void step(Dense& layer) const;

    [[nodiscard]] float learning_rate() const noexcept;

private:
    float learning_rate_;
};

}