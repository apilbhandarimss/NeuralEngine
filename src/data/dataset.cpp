#include "neural/data/dataset.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace neural {

namespace {

std::uint32_t read_big_endian_u32(std::istream& stream) {
    unsigned char bytes[4]{};

    if (!stream.read(
            reinterpret_cast<char*>(bytes),
            static_cast<std::streamsize>(sizeof(bytes)))) {
        throw std::runtime_error("Unexpected end of MNIST file header");
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24U)
         | (static_cast<std::uint32_t>(bytes[1]) << 16U)
         | (static_cast<std::uint32_t>(bytes[2]) << 8U)
         | static_cast<std::uint32_t>(bytes[3]);
}

std::size_t checked_product(
    std::size_t a,
    std::size_t b,
    const char* description
) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::runtime_error(
            std::string("MNIST size overflow: ") + description
        );
    }

    return a * b;
}

} // namespace

MNISTDataset::MNISTDataset(
    const std::filesystem::path& images_path,
    const std::filesystem::path& labels_path
) {
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file) {
        throw std::runtime_error(
            "Cannot open MNIST images file: " + images_path.string()
        );
    }

    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file) {
        throw std::runtime_error(
            "Cannot open MNIST labels file: " + labels_path.string()
        );
    }

    const std::uint32_t image_magic = read_big_endian_u32(images_file);
    const std::uint32_t image_count = read_big_endian_u32(images_file);
    const std::uint32_t rows = read_big_endian_u32(images_file);
    const std::uint32_t columns = read_big_endian_u32(images_file);

    const std::uint32_t label_magic = read_big_endian_u32(labels_file);
    const std::uint32_t label_count = read_big_endian_u32(labels_file);

    if (image_magic != 2051U) {
        throw std::runtime_error("Invalid MNIST image magic number");
    }

    if (label_magic != 2049U) {
        throw std::runtime_error("Invalid MNIST label magic number");
    }

    if (rows != 28U || columns != 28U) {
        throw std::runtime_error("MNIST images must be 28x28 pixels");
    }

    if (image_count != label_count) {
        throw std::runtime_error("MNIST image and label counts differ");
    }

    if (image_count == 0U) {
        throw std::runtime_error("MNIST dataset is empty");
    }

    constexpr std::size_t pixels_per_image = 28U * 28U;

    const std::size_t total_pixels = checked_product(
        static_cast<std::size_t>(image_count),
        pixels_per_image,
        "image pixels"
    );

    if (total_pixels >
        static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("MNIST image file is too large to read");
    }

    images_.resize(total_pixels);
    labels_.resize(static_cast<std::size_t>(label_count));
    image_count_ = static_cast<std::size_t>(image_count);

    if (!images_file.read(
            reinterpret_cast<char*>(images_.data()),
            static_cast<std::streamsize>(images_.size()))) {
        throw std::runtime_error("MNIST image data is truncated");
    }

    if (!labels_file.read(
            reinterpret_cast<char*>(labels_.data()),
            static_cast<std::streamsize>(labels_.size()))) {
        throw std::runtime_error("MNIST label data is truncated");
    }

    for (const std::uint8_t digit : labels_) {
        if (digit > 9U) {
            throw std::runtime_error("MNIST contains an invalid digit label");
        }
    }
}

std::size_t MNISTDataset::size() const noexcept {
    return image_count_;
}

Tensor MNISTDataset::image(std::size_t index) const {
    if (index >= image_count_) {
        throw std::out_of_range("MNIST image index out of range");
    }

    constexpr std::size_t pixels_per_image = 28U * 28U;
    const std::size_t offset = index * pixels_per_image;

    Tensor result({pixels_per_image}, 0.0F);
    auto output = result.data();

    for (std::size_t i = 0; i < pixels_per_image; ++i) {
        output[i] = static_cast<float>(images_[offset + i]) / 255.0F;
    }

    return result;
}

std::uint8_t MNISTDataset::label(std::size_t index) const {
    if (index >= image_count_) {
        throw std::out_of_range("MNIST label index out of range");
    }

    return labels_[index];
}

} 
