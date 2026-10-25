// #include "neural/nn/model.hpp"

// #include <stdexcept>
// #include <utility>

// namespace neural {

// void Model::add(std::unique_ptr<Layer> layer) {
//     if (!layer) {
//         throw std::invalid_argument("Cannot add a null layer");
//     }
//     layers_.push_back(std::move(layer));
// }

// Tensor Model::forward(const Tensor& input) const {
//     Tensor result = input;
//     for (const auto& layer : layers_) {
//         result = layer->forward(result);
//     }
//     return result;
// }

// } // namespace neural
