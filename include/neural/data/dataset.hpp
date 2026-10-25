#pragma once

#include "neural/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace neural {

class Dataset {
public:
    virtual ~Dataset() = default;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
};

class MNISTDataset final : public Dataset {
public:
    MNISTDataset(
        const std::filesystem::path& images_path,
        const std::filesystem::path& labels_path
    );

    [[nodiscard]] std::size_t size() const noexcept override;

    // Returns 784 normalised pixel values in [0, 1].
    [[nodiscard]] Tensor image(std::size_t index) const;

    // Returns the digit class in [0, 9].
    [[nodiscard]] std::uint8_t label(std::size_t index) const;

private:
    std::vector<std::uint8_t> images_;
    std::vector<std::uint8_t> labels_;
    std::size_t image_count_{};
};

} // namespace neural