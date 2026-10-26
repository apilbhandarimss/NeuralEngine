
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "neural/data/dataset.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

#include "raylib.h"

// Undo common platform macro pollutions from Raylib / Windows
#ifdef CloseWindow
#undef CloseWindow
#endif
#ifdef DrawText
#undef DrawText
#endif
#ifdef ShowCursor
#undef ShowCursor
#endif
#ifdef LoadImage
#undef LoadImage
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* CHECKPOINT_MAGIC = "NEURAL_ENGINE_CHECKPOINT";
constexpr long CHECKPOINT_VERSION = 1;
constexpr size_t MAX_LAYER_WIDTH = 65536;
constexpr size_t MAX_LAYER_COUNT = 32;
constexpr size_t MAX_PARAMETERS = 60'000'000;
constexpr size_t MNIST_INPUTS = 28 * 28;
constexpr size_t MNIST_CLASSES = 10;
constexpr uint32_t BASE_SEED = 100;
constexpr uint32_t SHUFFLE_SEED = 42;
constexpr size_t HISTORY_CAP = 4000;
constexpr size_t MAX_MISSES = 20000;
constexpr size_t GRID_COLS = 8;

constexpr int CANVAS_SIZE = 280;
constexpr float BRUSH_RADIUS = 11.0f;

constexpr const char* FILE_TRAIN_IMAGES = "train-images-idx3-ubyte";
constexpr const char* FILE_TRAIN_LABELS = "train-labels-idx1-ubyte";
constexpr const char* FILE_TEST_IMAGES  = "t10k-images-idx3-ubyte";
constexpr const char* FILE_TEST_LABELS  = "t10k-labels-idx1-ubyte";

// --- General string & math helpers ---

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim_whitespace(const std::string& s) {
    size_t start = 0;
    size_t end = s.size();
    while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string format_fixed(double val, int precision) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", precision, val);
    return buf;
}

std::string format_bytes(uintmax_t bytes) {
    const double b = static_cast<double>(bytes);
    if (b >= 1024.0 * 1024.0) return format_fixed(b / (1024.0 * 1024.0), 1) + " MB";
    if (b >= 1024.0) return format_fixed(b / 1024.0, 1) + " KB";
    return std::to_string(bytes) + " B";
}

std::string format_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) return "--";
    const auto s = static_cast<long long>(seconds);
    char buf[32];
    if (s >= 3600) {
        std::snprintf(buf, sizeof(buf), "%lldh %02lldm", s / 3600, (s / 60) % 60);
    } else {
        std::snprintf(buf, sizeof(buf), "%lldm %02llds", s / 60, s % 60);
    }
    return buf;
}

bool safe_is_file(const fs::path& p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

bool parse_size_t(const std::string& text, size_t& out) {
    const std::string str = trim_whitespace(text);
    if (str.empty()) return false;
    size_t val = 0;
    const auto res = std::from_chars(str.data(), str.data() + str.size(), val);
    if (res.ec != std::errc{} || res.ptr != str.data() + str.size()) return false;
    out = val;
    return true;
}

bool parse_float_val(const std::string& text, float& out) {
    const std::string str = trim_whitespace(text);
    if (str.empty()) return false;
    char* end = nullptr;
    const float val = std::strtof(str.c_str(), &end);
    if (end != str.c_str() + str.size()) return false;
    out = val;
    return true;
}

double compute_nice_step(double range, int target_ticks) {
    if (!(range > 0.0) || !std::isfinite(range)) return 1.0;
    const double raw = range / static_cast<double>(std::max(1, target_ticks));
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double norm = raw / mag;
    const double step = norm < 1.5 ? 1.0 : (norm < 3.0 ? 2.0 : (norm < 7.0 ? 5.0 : 10.0));
    return step * mag;
}

std::string format_tick(double val) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3g", val);
    return buf;
}

// --- Loss History Tracking ---

struct LossPoint {
    double samples = 0.0;
    float loss = 0.0f;
};

void compress_loss_history(std::vector<LossPoint>& history, size_t max_points) {
    if (history.size() <= max_points) return;
    std::vector<LossPoint> next;
    next.reserve(history.size() / 2 + 1);
    for (size_t i = 0; i + 1 < history.size(); i += 2) {
        next.push_back({
            (history[i].samples + history[i + 1].samples) * 0.5,
            (history[i].loss + history[i + 1].loss) * 0.5f
        });
    }
    if (history.size() % 2 == 1) {
        next.push_back(history.back());
    }
    history = std::move(next);
}

// --- Checkpoint I/O ---

class InvalidCheckpointError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string read_bounded_line(std::istream& in, size_t max_chars = 4096) {
    std::string line;
    char c = 0;
    while (line.size() < max_chars && in.get(c)) {
        if (c == '\n') break;
        line.push_back(c);
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

size_t calculate_params(const std::vector<size_t>& arch) {
    size_t count = 0;
    for (size_t i = 0; i + 1 < arch.size(); ++i) {
        count += arch[i] * arch[i + 1] + arch[i + 1];
    }
    return count;
}

std::string format_architecture(const std::vector<size_t>& arch) {
    std::string s;
    for (size_t i = 0; i < arch.size(); ++i) {
        if (i > 0) s += '-';
        s += std::to_string(arch[i]);
    }
    return s;
}

std::vector<size_t> parse_architecture_spec(const std::string& input) {
    std::string sanitized = input;
    for (char& c : sanitized) {
        if (c == ',' || c == '-' || c == '>' || c == '\t') c = ' ';
    }
    std::istringstream stream(sanitized);
    std::vector<size_t> arch;
    std::string tok;
    while (stream >> tok) {
        const bool all_digits = std::all_of(tok.begin(), tok.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        });
        if (!all_digits) {
            throw std::runtime_error("Architecture contains non-digit: '" + tok + "'");
        }
        if (tok.size() > 6) {
            throw std::runtime_error("Layer width value too large: " + tok);
        }
        arch.push_back(static_cast<size_t>(std::stoull(tok)));
    }
    if (arch.size() < 2) {
        throw std::runtime_error("Architecture must specify at least input and output sizes");
    }
    if (arch.size() > MAX_LAYER_COUNT + 1) {
        throw std::runtime_error("Too many layers in architecture spec");
    }
    for (size_t width : arch) {
        if (width == 0 || width > MAX_LAYER_WIDTH) {
            throw std::runtime_error("Layer dimensions must be in range [1, " + std::to_string(MAX_LAYER_WIDTH) + "]");
        }
    }
    if (calculate_params(arch) > MAX_PARAMETERS) {
        throw std::runtime_error("Total parameters exceed safety limit");
    }
    return arch;
}

std::vector<size_t> read_checkpoint_header(std::istream& in) {
    std::string line = read_bounded_line(in);
    std::istringstream head(line);
    std::string magic;
    if (!(head >> magic) || magic != CHECKPOINT_MAGIC) {
        throw InvalidCheckpointError("File does not start with NEURAL_ENGINE_CHECKPOINT");
    }
    long version = 0;
    if (!(head >> version)) {
        throw std::runtime_error("Missing checkpoint version number");
    }
    if (version != CHECKPOINT_VERSION) {
        throw std::runtime_error("Incompatible checkpoint version " + std::to_string(version) +
                                 " (expected " + std::to_string(CHECKPOINT_VERSION) + ")");
    }

    line = read_bounded_line(in);
    const std::string prefix = "ARCHITECTURE";
    if (line.rfind(prefix, 0) != 0) {
        throw std::runtime_error("Expected ARCHITECTURE block in checkpoint header");
    }
    return parse_architecture_spec(line.substr(prefix.size()));
}

std::vector<float> load_tensor_block(std::istream& in, const char* expected_tag, const std::vector<size_t>& shape) {
    std::string tag;
    if (!(in >> tag) || tag != expected_tag) {
        throw std::runtime_error(std::string("Missing expected section '") + expected_tag + "'");
    }
    size_t rank = 0;
    if (!(in >> rank) || rank != shape.size()) {
        throw std::runtime_error(std::string("Rank mismatch in section ") + expected_tag);
    }
    size_t total = 1;
    for (size_t expected_dim : shape) {
        size_t actual_dim = 0;
        if (!(in >> actual_dim) || actual_dim != expected_dim) {
            throw std::runtime_error(std::string("Dimension mismatch in ") + expected_tag);
        }
        total *= actual_dim;
    }
    std::vector<float> out(total);
    for (size_t i = 0; i < total; ++i) {
        float val = 0.0f;
        if (!(in >> val)) {
            throw std::runtime_error(std::string("Premature EOF or invalid numeric data in ") + expected_tag);
        }
        if (!std::isfinite(val)) {
            throw std::runtime_error(std::string("Non-finite weight detected in ") + expected_tag);
        }
        out[i] = val;
    }
    return out;
}

void write_tensor_data(std::ostream& out, const char* label, const neural::Tensor& t) {
    out << label << '\n' << t.rank();
    for (size_t dim : t.shape()) {
        out << ' ' << dim;
    }
    out << '\n';

    size_t count = 0;
    for (float val : t.data()) {
        out << val << ' ';
        if (++count % 8 == 0) out << '\n';
    }
    if (count % 8 != 0) out << '\n';
}

// --- Feedforward Network Wrapper ---

class Network {
public:
    Network(std::vector<size_t> architecture, neural::WeightInit hidden_init, uint32_t seed)
        : arch_(std::move(architecture)) {
        if (arch_.size() < 2) {
            throw std::invalid_argument("Network requires at least 2 layer sizes");
        }
        for (size_t i = 0; i + 1 < arch_.size(); ++i) {
            const bool is_output = (i + 2 == arch_.size());
            const auto init = is_output ? neural::WeightInit::Xavier : hidden_init;
            layers_.push_back(std::make_unique<neural::Dense>(
                arch_[i], arch_[i + 1], init, seed + 1U + static_cast<uint32_t>(i)));
        }
        relus_ = std::vector<neural::ReLU>(layers_.size() - 1);
    }

    Network(Network&&) noexcept = default;
    Network& operator=(Network&&) noexcept = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    const std::vector<size_t>& architecture() const noexcept { return arch_; }
    size_t layer_count() const noexcept { return layers_.size(); }
    neural::Dense& layer(size_t i) { return *layers_.at(i); }
    const neural::Dense& layer(size_t i) const { return *layers_.at(i); }

    bool is_mnist_compatible() const noexcept {
        return arch_.front() == MNIST_INPUTS && arch_.back() == MNIST_CLASSES;
    }

    Network clone() const {
        Network copy(arch_, neural::WeightInit::He, BASE_SEED);
        for (size_t i = 0; i < layers_.size(); ++i) {
            const auto& src = *layers_[i];
            auto& dst = *copy.layers_[i];
            auto src_w = src.weights().data();
            auto src_b = src.bias().data();
            auto dst_w = dst.weights().data();
            auto dst_b = dst.bias().data();
            std::copy(src_w.begin(), src_w.end(), dst_w.begin());
            std::copy(src_b.begin(), src_b.end(), dst_b.begin());
        }
        return copy;
    }

    neural::Tensor forward(const neural::Tensor& input) {
        neural::Tensor x = layers_[0]->forward(input);
        for (size_t i = 1; i < layers_.size(); ++i) {
            x = relus_[i - 1].forward(x);
            x = layers_[i]->forward(x);
        }
        return x;
    }

    float train_step(const neural::Tensor& input, size_t target, const neural::SGD& optimizer) {
        const neural::Tensor logits = forward(input);
        const neural::LossResult result = neural::softmax_cross_entropy(logits, target);

        neural::Tensor grad = layers_.back()->backward(result.logits_gradient);
        for (size_t i = layers_.size() - 1; i > 0; --i) {
            const size_t relu_idx = i - 1;
            grad = relus_[relu_idx].backward(grad);
            grad = layers_[relu_idx]->backward(grad);
        }
        for (auto& l : layers_) {
            optimizer.step(*l);
        }
        return result.loss;
    }

    void save(const fs::path& path) const {
        std::error_code ec;
        const fs::path parent = path.parent_path();
        if (!parent.empty() && !fs::exists(parent, ec)) {
            fs::create_directories(parent, ec);
            if (ec) {
                throw std::runtime_error("Failed to create directory " + parent.string() + ": " + ec.message());
            }
        }

        fs::path tmp = path;
        tmp += ".tmp";
        {
            std::ofstream file(tmp);
            if (!file) {
                throw std::runtime_error("Cannot write to path: " + tmp.string());
            }
            file << std::setprecision(std::numeric_limits<float>::max_digits10);
            file << CHECKPOINT_MAGIC << ' ' << CHECKPOINT_VERSION << '\n';
            file << "ARCHITECTURE";
            for (size_t w : arch_) file << ' ' << w;
            file << '\n';

            for (size_t i = 0; i < layers_.size(); ++i) {
                file << "LAYER fc" << (i + 1) << '\n';
                write_tensor_data(file, "WEIGHTS", layers_[i]->weights());
                write_tensor_data(file, "BIAS", layers_[i]->bias());
            }
            file.flush();
            if (!file) {
                file.close();
                fs::remove(tmp, ec);
                throw std::runtime_error("Stream error while writing " + tmp.string());
            }
        }
        fs::rename(tmp, path, ec);
        if (ec) {
            std::error_code rm_ec;
            fs::remove(tmp, rm_ec);
            throw std::runtime_error("Atomic rename failed: " + ec.message());
        }
    }

private:
    std::vector<size_t> arch_;
    std::vector<std::unique_ptr<neural::Dense>> layers_;
    std::vector<neural::ReLU> relus_;
};

Network load_checkpoint(const fs::path& path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) {
        throw std::runtime_error("Checkpoint file not found: " + path.string());
    }
    const uintmax_t size = fs::file_size(path, ec);
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Access denied or failed to open: " + path.string());
    }

    const std::vector<size_t> arch = read_checkpoint_header(in);
    if (!ec && size < calculate_params(arch) * 2) {
        throw std::runtime_error("File size is smaller than theoretical minimum for declared architecture");
    }

    Network net(arch, neural::WeightInit::He, BASE_SEED);

    for (size_t i = 0; i < net.layer_count(); ++i) {
        std::string tag, name;
        if (!(in >> tag >> name) || tag != "LAYER") {
            throw std::runtime_error("Missing LAYER tag for layer fc" + std::to_string(i + 1));
        }
        const std::string expected = "fc" + std::to_string(i + 1);
        if (name != expected) {
            throw std::runtime_error("Layer sequence mismatch: found '" + name + "', expected " + expected);
        }

        const size_t in_dim = arch[i];
        const size_t out_dim = arch[i + 1];
        const std::vector<float> w = load_tensor_block(in, "WEIGHTS", {out_dim, in_dim});
        const std::vector<float> b = load_tensor_block(in, "BIAS", {out_dim});

        neural::Dense& layer = net.layer(i);
        auto wd = layer.weights().data();
        auto bd = layer.bias().data();
        if (wd.size() != w.size() || bd.size() != b.size()) {
            throw std::runtime_error("Internal dimension mismatch in " + name);
        }
        std::copy(w.begin(), w.end(), wd.begin());
        std::copy(b.begin(), b.end(), bd.begin());
    }

    std::string leftover;
    if (in >> leftover) {
        throw std::runtime_error("Unexpected trailing content in checkpoint");
    }
    return net;
}

// --- Checkpoint Discovery ---

struct ModelEntry {
    fs::path path;
    std::string rel_path;
    std::string filename;
    bool valid = false;
    std::string error_msg;
    std::vector<size_t> arch;
    uintmax_t file_bytes = 0;
};

struct DirectoryScan {
    std::vector<ModelEntry> entries;
    size_t non_checkpoints = 0;
    size_t unreadable_files = 0;
    bool reached_limit = false;
    std::string error_message;
};

bool should_ignore_dir(const std::string& name) {
    static const std::vector<std::string> blacklist = {
        ".git", ".cache", ".idea", ".vscode", "CMakeFiles", "_deps",
        "third_party", "external", "vendor", "node_modules", "Testing", "out"
    };
    if (std::find(blacklist.begin(), blacklist.end(), name) != blacklist.end()) return true;
    return name.rfind("build", 0) == 0 || name.rfind("cmake-build", 0) == 0;
}

DirectoryScan scan_checkpoint_directory(const fs::path& root) {
    constexpr size_t MAX_TRAVERSAL = 200'000;
    constexpr int MAX_DEPTH = 8;

    DirectoryScan scan;
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        scan.error_message = "Path is not a valid directory: " + root.string();
        return scan;
    }

    fs::recursive_directory_iterator iter(root, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        scan.error_message = "Filesystem error: " + ec.message();
        return scan;
    }

    const fs::recursive_directory_iterator end;
    size_t visited = 0;

    while (iter != end) {
        if (++visited > MAX_TRAVERSAL) {
            scan.reached_limit = true;
            break;
        }
        const auto& entry = *iter;
        std::error_code entry_ec;

        if (entry.is_directory(entry_ec)) {
            if (should_ignore_dir(entry.path().filename().string()) || iter.depth() >= MAX_DEPTH) {
                iter.disable_recursion_pending();
            }
        } else if (entry.is_regular_file(entry_ec) && to_lower(entry.path().extension().string()) == ".txt") {
            std::ifstream file(entry.path());
            if (!file) {
                ++scan.unreadable_files;
            } else {
                ModelEntry item;
                item.path = entry.path();
                item.rel_path = entry.path().lexically_relative(root).generic_string();
                item.filename = entry.path().filename().string();
                item.file_bytes = entry.file_size(entry_ec);

                bool is_ckpt = true;
                try {
                    item.arch = read_checkpoint_header(file);
                    item.valid = true;
                } catch (const InvalidCheckpointError&) {
                    is_ckpt = false;
                } catch (const std::exception& e) {
                    item.valid = false;
                    item.error_msg = e.what();
                }

                if (is_ckpt) {
                    scan.entries.push_back(std::move(item));
                } else {
                    ++scan.non_checkpoints;
                }
            }
        }
        iter.increment(ec);
        if (ec) {
            scan.error_message = "Scan aborted: " + ec.message();
            break;
        }
    }

    std::sort(scan.entries.begin(), scan.entries.end(), [](const ModelEntry& a, const ModelEntry& b) {
        return to_lower(a.rel_path) < to_lower(b.rel_path);
    });
    return scan;
}

// --- Dataset Paths ---

struct DatasetBundle {
    fs::path train_images;
    fs::path train_labels;
    fs::path test_images;
    fs::path test_labels;

    bool has_train() const { return safe_is_file(train_images) && safe_is_file(train_labels); }
    bool has_test() const { return safe_is_file(test_images) && safe_is_file(test_labels); }
};

DatasetBundle resolve_dataset_files(const fs::path& dir) {
    return {
        dir / FILE_TRAIN_IMAGES,
        dir / FILE_TRAIN_LABELS,
        dir / FILE_TEST_IMAGES,
        dir / FILE_TEST_LABELS
    };
}

fs::path detect_project_root() {
    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (ec) return ".";
    fs::path probe = cwd;

    for (int i = 0; i < 5; ++i) {
        if (fs::exists(probe / "CMakeLists.txt", ec) && fs::is_directory(probe / "examples", ec)) {
            return probe;
        }
        if (!probe.has_parent_path() || probe.parent_path() == probe) break;
        probe = probe.parent_path();
    }
    return cwd;
}

fs::path find_default_dataset_dir(const fs::path& root) {
    std::error_code ec;
    const std::vector<fs::path> paths = {
        root / "data" / "mnist",
        fs::current_path(ec) / "data" / "mnist",
        fs::current_path(ec) / ".." / "data" / "mnist"
    };
    for (const auto& p : paths) {
        if (safe_is_file(p / FILE_TRAIN_IMAGES)) {
            return fs::weakly_canonical(p, ec);
        }
    }
    return root / "data" / "mnist";
}

// --- Canvas & Centering Preprocessor ---

void paint_brush_stamp(std::vector<uint8_t>& canvas, float cx, float cy, float radius) {
    const int x0 = std::max(0, static_cast<int>(std::floor(cx - radius)));
    const int x1 = std::min(CANVAS_SIZE - 1, static_cast<int>(std::ceil(cx + radius)));
    const int y0 = std::max(0, static_cast<int>(std::floor(cy - radius)));
    const int y1 = std::min(CANVAS_SIZE - 1, static_cast<int>(std::ceil(cy + radius)));

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float dx = static_cast<float>(x) + 0.5f - cx;
            const float dy = static_cast<float>(y) + 0.5f - cy;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dist > radius) continue;

            const float ratio = dist / radius;
            const float falloff = ratio < 0.6f ? 1.0f : (1.0f - ratio) / 0.4f;
            const auto intensity = static_cast<uint8_t>(std::lround(falloff * 255.0f));
            uint8_t& pixel = canvas[static_cast<size_t>(y * CANVAS_SIZE + x)];
            pixel = std::max(pixel, intensity);
        }
    }
}

void paint_brush_stroke(std::vector<uint8_t>& canvas, float x0, float y0, float x1, float y1) {
    const float d = std::hypot(x1 - x0, y1 - y0);
    const int steps = std::max(1, static_cast<int>(std::ceil(d / 2.0f)));
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        paint_brush_stamp(canvas, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, BRUSH_RADIUS);
    }
}

struct DigitSample {
    std::array<float, MNIST_INPUTS> pixels{};
    bool empty = true;
};

DigitSample preprocess_digit_image(const std::vector<uint8_t>& canvas, int size) {
    DigitSample res;
    int min_x = size, min_y = size;
    int max_x = -1, max_y = -1;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (canvas[static_cast<size_t>(y * size + x)] > 24) {
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }
    }
    if (max_x < 0) return res;

    const int bbox_w = max_x - min_x + 1;
    const int bbox_h = max_y - min_y + 1;
    const double scale = 20.0 / static_cast<double>(std::max(bbox_w, bbox_h));
    const int target_w = std::clamp(static_cast<int>(std::lround(bbox_w * scale)), 1, 20);
    const int target_h = std::clamp(static_cast<int>(std::lround(bbox_h * scale)), 1, 20);
    const int box_x = (28 - target_w) / 2;
    const int box_y = (28 - target_h) / 2;

    std::array<float, MNIST_INPUTS> frame{};
    for (int ty = 0; ty < target_h; ++ty) {
        for (int tx = 0; tx < target_w; ++tx) {
            const double src_x0 = min_x + static_cast<double>(tx) * bbox_w / target_w;
            const double src_x1 = min_x + static_cast<double>(tx + 1) * bbox_w / target_w;
            const double src_y0 = min_y + static_cast<double>(ty) * bbox_h / target_h;
            const double src_y1 = min_y + static_cast<double>(ty + 1) * bbox_h / target_h;

            const int px0 = std::clamp(static_cast<int>(std::floor(src_x0)), min_x, max_x);
            const int px1 = std::clamp(std::max(px0 + 1, static_cast<int>(std::ceil(src_x1))), min_x + 1, max_x + 1);
            const int py0 = std::clamp(static_cast<int>(std::floor(src_y0)), min_y, max_y);
            const int py1 = std::clamp(std::max(py0 + 1, static_cast<int>(std::ceil(src_y1))), min_y + 1, max_y + 1);

            double sum = 0.0;
            for (int y = py0; y < py1; ++y) {
                for (int x = px0; x < px1; ++x) {
                    sum += canvas[static_cast<size_t>(y * size + x)];
                }
            }
            const double mean = sum / (static_cast<double>(px1 - px0) * (py1 - py0));
            frame[static_cast<size_t>((box_y + ty) * 28 + box_x + tx)] = static_cast<float>(mean / 255.0);
        }
    }

    double total_mass = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const double v = frame[static_cast<size_t>(y * 28 + x)];
            total_mass += v;
            weighted_x += v * x;
            weighted_y += v * y;
        }
    }

    int shift_x = 0;
    int shift_y = 0;
    if (total_mass > 0.0) {
        shift_x = static_cast<int>(std::lround(13.5 - weighted_x / total_mass));
        shift_y = static_cast<int>(std::lround(13.5 - weighted_y / total_mass));
    }

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const int nx = x + shift_x;
            const int ny = y + shift_y;
            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                res.pixels[static_cast<size_t>(ny * 28 + nx)] =
                    std::clamp(frame[static_cast<size_t>(y * 28 + x)], 0.0f, 1.0f);
            }
        }
    }
    res.empty = false;
    return res;
}

neural::Tensor tensor_from_pixels(const std::array<float, MNIST_INPUTS>& pixels) {
    neural::Tensor t({MNIST_INPUTS}, 0.0f);
    auto data = t.data();
    for (size_t i = 0; i < MNIST_INPUTS; ++i) data[i] = pixels[i];
    return t;
}

struct ClassPrediction {
    std::array<float, MNIST_CLASSES> probabilities{};
    size_t digit = 0;
};

ClassPrediction predict_class(Network& net, const neural::Tensor& input) {
    if (!net.is_mnist_compatible()) {
        throw std::runtime_error("Network requires 784 inputs and 10 outputs for MNIST");
    }
    const neural::Tensor logits = net.forward(input);
    if (logits.size() != MNIST_CLASSES) {
        throw std::runtime_error("Unexpected logit vector dimension");
    }

    const neural::LossResult r = neural::softmax_cross_entropy(logits, 0);
    ClassPrediction pred;
    const auto p = r.probabilities.data();
    for (size_t i = 0; i < MNIST_CLASSES; ++i) pred.probabilities[i] = p[i];
    pred.digit = static_cast<size_t>(
        std::distance(pred.probabilities.begin(),
                      std::max_element(pred.probabilities.begin(), pred.probabilities.end())));
    return pred;
}

// --- Detailed Evaluation Suite ---

struct EvaluationResult {
    size_t correct = 0;
    size_t total = 0;
    double cumulative_loss = 0.0;
    bool was_cancelled = false;
    bool is_test_set = true;
    std::string model_identifier;
    std::array<std::array<size_t, MNIST_CLASSES>, MNIST_CLASSES> confusion_matrix{};
    std::vector<size_t> misclassified_indices;
    size_t miss_count = 0;

    double accuracy_percent() const {
        return total > 0 ? (100.0 * static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
    }
    double average_loss() const {
        return total > 0 ? (cumulative_loss / static_cast<double>(total)) : 0.0;
    }
};

EvaluationResult evaluate_network(Network& net,
                                 const neural::MNISTDataset& data,
                                 const std::stop_token& stop,
                                 const std::function<void(size_t, size_t)>& on_progress) {
    EvaluationResult res;
    for (size_t i = 0; i < data.size(); ++i) {
        if (stop.stop_requested()) {
            res.was_cancelled = true;
            return res;
        }
        const neural::Tensor logits = net.forward(data.image(i));
        if (logits.size() != MNIST_CLASSES) {
            throw std::runtime_error("Model forward pass produced mismatched class count");
        }
        const size_t target = data.label(i);
        const neural::LossResult lr = neural::softmax_cross_entropy(logits, target);
        const auto p = lr.probabilities.data();
        const auto best = static_cast<size_t>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));

        res.cumulative_loss += static_cast<double>(lr.loss);
        ++res.confusion_matrix[target][best];

        if (best == target) {
            ++res.correct;
        } else {
            ++res.miss_count;
            if (res.misclassified_indices.size() < MAX_MISSES) {
                res.misclassified_indices.push_back(i);
            }
        }
        ++res.total;
        if ((i + 1) % 200 == 0) on_progress(i + 1, res.correct);
    }
    on_progress(data.size(), res.correct);
    return res;
}

enum class WorkerState { Idle, Running, Complete };

class AsyncEvaluator {
public:
    bool is_active() const { return state_.load() == WorkerState::Running; }
    size_t completed() const { return done_.load(); }
    size_t total_samples() const { return total_.load(); }

    double running_accuracy() const {
        const size_t d = done_.load();
        return d > 0 ? (100.0 * static_cast<double>(correct_.load()) / static_cast<double>(d)) : 0.0;
    }

    void start(Network net,
               std::shared_ptr<const neural::MNISTDataset> dataset,
               bool test_set,
               std::string model_name) {
        if (is_active()) return;
        if (thread_.joinable()) thread_.join();

        done_ = 0;
        correct_ = 0;
        total_ = dataset->size();

        {
            const std::lock_guard<std::mutex> lock(mtx_);
            result_ = EvaluationResult{};
            error_.clear();
        }

        state_ = WorkerState::Running;
        thread_ = std::jthread([this,
                                n = std::move(net),
                                d = std::move(dataset),
                                test_set,
                                name = std::move(model_name)](std::stop_token st) mutable {
            EvaluationResult res;
            std::string err;
            try {
                res = evaluate_network(n, *d, st, [this](size_t cur, size_t num_correct) {
                    done_ = cur;
                    correct_ = num_correct;
                });
            } catch (const std::exception& e) {
                err = e.what();
            } catch (...) {
                err = "Fatal evaluation error";
            }
            res.is_test_set = test_set;
            res.model_identifier = name;

            {
                const std::lock_guard<std::mutex> lock(mtx_);
                result_ = std::move(res);
                error_ = std::move(err);
            }
            state_ = WorkerState::Complete;
        });
    }

    void cancel() {
        if (thread_.joinable()) thread_.request_stop();
    }

    bool fetch_result(EvaluationResult& out, std::string& err) {
        if (state_.load() != WorkerState::Complete) return false;
        const std::lock_guard<std::mutex> lock(mtx_);
        out = std::move(result_);
        err = error_;
        state_ = WorkerState::Idle;
        return true;
    }

private:
    std::atomic<WorkerState> state_{WorkerState::Idle};
    std::atomic<size_t> done_{0};
    std::atomic<size_t> correct_{0};
    std::atomic<size_t> total_{0};
    mutable std::mutex mtx_;
    EvaluationResult result_;
    std::string error_;
    std::jthread thread_;
};

// --- Training Worker ---

struct TrainingConfig {
    fs::path data_dir;
    fs::path save_target;
    std::optional<fs::path> resume_checkpoint;
    std::vector<size_t> architecture;
    neural::WeightInit weight_init = neural::WeightInit::He;
    float lr = 0.01f;
    size_t epoch_count = 5;
    size_t max_epoch_samples = 0;
    bool run_eval_on_completion = false;
    bool persist_on_early_stop = true;
};

enum class StepPhase { Idle, Loading, Running, Exporting, Validating, Complete, Aborted, Error };

const char* stringify_phase(StepPhase p) {
    switch (p) {
        case StepPhase::Idle: return "Idle";
        case StepPhase::Loading: return "Loading";
        case StepPhase::Running: return "Training";
        case StepPhase::Exporting: return "Saving";
        case StepPhase::Validating: return "Evaluating";
        case StepPhase::Complete: return "Complete";
        case StepPhase::Aborted: return "Stopped";
        case StepPhase::Error: return "Failed";
    }
    return "?";
}

struct TrainingStateSnapshot {
    StepPhase phase = StepPhase::Idle;
    size_t epoch = 0;
    size_t total_epochs = 0;
    size_t sample_in_epoch = 0;
    size_t epoch_size = 0;
    size_t total_processed = 0;
    size_t eval_done = 0;
    size_t eval_total = 0;
    double current_loss = 0.0;
    double test_accuracy = -1.0;
    double test_loss = 0.0;
    double elapsed_sec = 0.0;
    std::string status_msg;
    std::string error_msg;
    std::string saved_file;
    std::vector<LossPoint> loss_curve;
};

class TrainingEngine {
public:
    ~TrainingEngine() = default;

    bool is_busy() const {
        const auto p = phase_.load();
        return p == StepPhase::Loading || p == StepPhase::Running ||
               p == StepPhase::Exporting || p == StepPhase::Validating;
    }

    void start(TrainingConfig cfg) {
        if (is_busy()) return;
        if (thread_.joinable()) thread_.join();

        {
            const std::lock_guard<std::mutex> lock(mtx_);
            status_text_.clear();
            error_text_.clear();
            output_path_.clear();
            loss_points_.clear();
        }

        epoch_ = 0;
        target_epochs_ = cfg.epoch_count;
        step_in_epoch_ = 0;
        steps_per_epoch_ = 0;
        total_samples_ = 0;
        eval_done_ = 0;
        eval_total_ = 0;
        recent_loss_ = 0.0;
        test_accuracy_ = -1.0;
        test_loss_ = 0.0;
        start_time_ns_ = get_timestamp_ns();
        end_time_ns_ = 0;
        phase_ = StepPhase::Loading;

        thread_ = std::jthread([this, c = std::move(cfg)](std::stop_token st) { execute(st, c); });
    }

    void stop() {
        if (thread_.joinable()) thread_.request_stop();
    }

    TrainingStateSnapshot get_snapshot(bool copy_history) const {
        TrainingStateSnapshot s;
        s.phase = phase_.load();
        s.epoch = epoch_;
        s.total_epochs = target_epochs_;
        s.sample_in_epoch = step_in_epoch_;
        s.epoch_size = steps_per_epoch_;
        s.total_processed = total_samples_;
        s.eval_done = eval_done_;
        s.eval_total = eval_total_;
        s.current_loss = recent_loss_;
        s.test_accuracy = test_accuracy_;
        s.test_loss = test_loss_;

        const long long t0 = start_time_ns_;
        const long long t1 = end_time_ns_;
        if (t0 != 0) {
            s.elapsed_sec = static_cast<double>((t1 != 0 ? t1 : get_timestamp_ns()) - t0) * 1e-9;
        }

        const std::lock_guard<std::mutex> lock(mtx_);
        s.status_msg = status_text_;
        s.error_msg = error_text_;
        s.saved_file = output_path_;
        if (copy_history) s.loss_curve = loss_points_;
        return s;
    }

private:
    static long long get_timestamp_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void post_status(const std::string& msg) {
        const std::lock_guard<std::mutex> lock(mtx_);
        status_text_ = msg;
    }

    void conclude(StepPhase terminal_phase) {
        end_time_ns_ = get_timestamp_ns();
        phase_ = terminal_phase;
    }

    void execute(const std::stop_token& stop, const TrainingConfig& cfg) {
        try {
            post_status("Loading dataset files...");
            const DatasetBundle files = resolve_dataset_files(cfg.data_dir);
            const neural::MNISTDataset train_data(files.train_images, files.train_labels);

            std::optional<neural::MNISTDataset> test_data;
            if (cfg.run_eval_on_completion) {
                test_data.emplace(files.test_images, files.test_labels);
            }

            std::optional<Network> net;
            if (cfg.resume_checkpoint) {
                post_status("Restoring model from checkpoint...");
                net.emplace(load_checkpoint(*cfg.resume_checkpoint));
            } else {
                net.emplace(cfg.architecture, cfg.weight_init, BASE_SEED);
            }

            if (!net->is_mnist_compatible()) {
                throw std::runtime_error("Network dimensions incompatible with MNIST dataset");
            }
            if (stop.stop_requested()) {
                post_status("Cancelled prior to start");
                conclude(StepPhase::Aborted);
                return;
            }

            const neural::SGD opt(cfg.lr);
            std::vector<size_t> indices(train_data.size());
            std::iota(indices.begin(), indices.end(), size_t{0});
            std::mt19937 rng(SHUFFLE_SEED);

            const size_t samples_per_epoch = (cfg.max_epoch_samples == 0)
                ? train_data.size()
                : std::min(cfg.max_epoch_samples, train_data.size());

            steps_per_epoch_ = samples_per_epoch;
            phase_ = StepPhase::Running;
            post_status("Training in progress (SGD, batch size = 1)...");

            bool aborted = false;
            double window_sum = 0.0;
            size_t window_items = 0;

            auto emit_loss_point = [&]() {
                if (window_items == 0) return;
                const double mean_loss = window_sum / static_cast<double>(window_items);
                recent_loss_ = mean_loss;
                {
                    const std::lock_guard<std::mutex> lock(mtx_);
                    loss_points_.push_back({
                        static_cast<double>(total_samples_.load()),
                        static_cast<float>(mean_loss)
                    });
                    compress_loss_history(loss_points_, HISTORY_CAP);
                }
                window_sum = 0.0;
                window_items = 0;
            };

            for (size_t ep = 0; ep < cfg.epoch_count && !aborted; ++ep) {
                epoch_ = ep + 1;
                step_in_epoch_ = 0;
                std::shuffle(indices.begin(), indices.end(), rng);

                for (size_t step = 0; step < samples_per_epoch; ++step) {
                    if (stop.stop_requested()) {
                        aborted = true;
                        break;
                    }
                    const size_t sample_idx = indices[step];
                    const neural::Tensor img = train_data.image(sample_idx);
                    const float loss = net->train_step(img, train_data.label(sample_idx), opt);

                    if (!std::isfinite(loss)) {
                        throw std::runtime_error("Training loss diverged to NaN/Inf. Lower the learning rate.");
                    }

                    window_sum += static_cast<double>(loss);
                    ++window_items;
                    ++total_samples_;
                    step_in_epoch_ = step + 1;

                    if (window_items == 500) {
                        emit_loss_point();
                    }
                }
                emit_loss_point();
            }

            if (aborted && (!cfg.persist_on_early_stop || total_samples_ == 0)) {
                post_status("Training cancelled. Checkpoint discarded.");
                conclude(StepPhase::Aborted);
                return;
            }

            phase_ = StepPhase::Exporting;
            post_status("Writing trained parameters...");
            net->save(cfg.save_target);
            {
                const std::lock_guard<std::mutex> lock(mtx_);
                output_path_ = cfg.save_target.string();
            }

            if (aborted) {
                post_status("Training stopped. Saved current model state.");
                conclude(StepPhase::Aborted);
                return;
            }

            if (test_data) {
                phase_ = StepPhase::Validating;
                post_status("Running validation on test split...");
                eval_total_ = test_data->size();
                const EvaluationResult ev = evaluate_network(
                    *net, *test_data, stop, [this](size_t cur, size_t) { eval_done_ = cur; });
                if (ev.was_cancelled) {
                    post_status("Validation aborted. Checkpoint preserved.");
                    conclude(StepPhase::Aborted);
                    return;
                }
                test_accuracy_ = ev.accuracy_percent();
                test_loss_ = ev.average_loss();
            }

            post_status("Training run completed successfully.");
            conclude(StepPhase::Complete);
        } catch (const std::exception& ex) {
            {
                const std::lock_guard<std::mutex> lock(mtx_);
                error_text_ = ex.what();
            }
            post_status("Training failed due to exception.");
            conclude(StepPhase::Error);
        } catch (...) {
            {
                const std::lock_guard<std::mutex> lock(mtx_);
                error_text_ = "Unknown non-standard exception caught.";
            }
            conclude(StepPhase::Error);
        }
    }

    std::atomic<StepPhase> phase_{StepPhase::Idle};
    std::atomic<size_t> epoch_{0};
    std::atomic<size_t> target_epochs_{0};
    std::atomic<size_t> step_in_epoch_{0};
    std::atomic<size_t> steps_per_epoch_{0};
    std::atomic<size_t> total_samples_{0};
    std::atomic<size_t> eval_done_{0};
    std::atomic<size_t> eval_total_{0};
    std::atomic<double> recent_loss_{0.0};
    std::atomic<double> test_accuracy_{-1.0};
    std::atomic<double> test_loss_{0.0};
    std::atomic<long long> start_time_ns_{0};
    std::atomic<long long> end_time_ns_{0};

    mutable std::mutex mtx_;
    std::string status_text_;
    std::string error_text_;
    std::string output_path_;
    std::vector<LossPoint> loss_points_;
    std::jthread thread_;
};

// --- GUI Architecture & Theming ---

namespace theme {
    constexpr Color BG_DARK        = {15, 17, 22, 255};
    constexpr Color SIDEBAR_BG     = {20, 23, 30, 255};
    constexpr Color PANEL_BG       = {26, 30, 39, 255};
    constexpr Color PANEL_ELEVATED = {35, 40, 52, 255};
    constexpr Color BORDER_COLOR   = {47, 54, 70, 255};
    constexpr Color TEXT_NORMAL    = {226, 230, 239, 255};
    constexpr Color TEXT_MUTED     = {139, 148, 167, 255};
    constexpr Color TEXT_FAINT     = {85, 93, 112, 255};
    constexpr Color ACCENT_PRIMARY = {88, 166, 255, 255};
    constexpr Color ACCENT_SUBDUED = {34, 66, 106, 255};
    constexpr Color STATUS_GOOD    = {76, 201, 140, 255};
    constexpr Color STATUS_WARN    = {240, 180, 70, 255};
    constexpr Color STATUS_ERR     = {240, 98, 98, 255};
}

enum class NoticeLevel { Info, Success, Warning, Error };

struct StatusNotice {
    std::string message;
    NoticeLevel level = NoticeLevel::Info;
};

Color resolve_notice_color(NoticeLevel lvl) {
    switch (lvl) {
        case NoticeLevel::Success: return theme::STATUS_GOOD;
        case NoticeLevel::Warning: return theme::STATUS_WARN;
        case NoticeLevel::Error:   return theme::STATUS_ERR;
        default:                   return theme::TEXT_MUTED;
    }
}

Color blend_colors(Color a, Color b, float factor) {
    auto lerp_ch = [factor](unsigned char c1, unsigned char c2) {
        return static_cast<unsigned char>(std::lround(c1 + (c2 - c1) * factor));
    };
    return Color{lerp_ch(a.r, b.r), lerp_ch(a.g, b.g), lerp_ch(a.b, b.b), lerp_ch(a.a, b.a)};
}

struct RenderContext {
    Font font{};
    bool using_custom_font = false;
    Rectangle scissor_bounds{0, 0, 100000, 100000};
    bool input_blocked = false;
    int active_focus_id = -1;
    size_t caret_pos = 0;
    int pointer_style = MOUSE_CURSOR_DEFAULT;

    float text_spacing(float pt) const { return using_custom_font ? 0.4f : pt / 10.0f; }
};

void draw_ui_text(const RenderContext& ctx, const std::string& str, float x, float y, float size, Color col) {
    DrawTextEx(ctx.font, str.c_str(), Vector2{x, y}, size, ctx.text_spacing(size), col);
}

float measure_ui_text(const RenderContext& ctx, const std::string& str, float size) {
    return MeasureTextEx(ctx.font, str.c_str(), size, ctx.text_spacing(size)).x;
}

std::string truncate_middle(const RenderContext& ctx, const std::string& str, float size, float max_px) {
    if (measure_ui_text(ctx, str, size) <= max_px) return str;
    size_t len = str.size();
    while (len > 4) {
        len -= 2;
        std::string candidate = str.substr(0, len / 2) + "..." + str.substr(str.size() - len / 2);
        if (measure_ui_text(ctx, candidate, size) <= max_px) return candidate;
    }
    return "...";
}

float draw_multiline_text(const RenderContext& ctx, const std::string& text, float x, float y, float max_w, float size, Color col) {
    const float spacing = size * 1.35f;
    std::istringstream stream(text);
    std::string word, line;
    float curr_y = y;

    auto print_line = [&]() {
        if (!line.empty()) {
            draw_ui_text(ctx, line, x, curr_y, size, col);
            curr_y += spacing;
            line.clear();
        }
    };

    while (stream >> word) {
        while (measure_ui_text(ctx, word, size) > max_w && word.size() > 1) {
            size_t take = word.size();
            while (take > 1 && measure_ui_text(ctx, word.substr(0, take), size) > max_w) --take;
            print_line();
            draw_ui_text(ctx, word.substr(0, take), x, curr_y, size, col);
            curr_y += spacing;
            word = word.substr(take);
        }
        std::string test = line.empty() ? word : line + " " + word;
        if (!line.empty() && measure_ui_text(ctx, test, size) > max_w) {
            print_line();
            line = word;
        } else {
            line = test;
        }
    }
    print_line();
    return curr_y - y;
}

void render_rounded_box(Rectangle rect, float radius, Color fill) {
    const float curve = std::min(1.0f, 2.0f * radius / std::min(rect.width, rect.height));
    DrawRectangleRounded(rect, curve, 8, fill);
}

void render_outlined_box(Rectangle rect, float radius, Color border, Color fill) {
    render_rounded_box(rect, radius, border);
    render_rounded_box(Rectangle{rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2}, radius - 1, fill);
}

bool hit_test(const RenderContext& ctx, Rectangle rect) {
    if (ctx.input_blocked) return false;
    const Vector2 m = GetMousePosition();
    return CheckCollisionPointRec(m, ctx.scissor_bounds) && CheckCollisionPointRec(m, rect);
}

enum class ActionStyle { Default, Accent, Danger, Highlighted };

bool ui_button(RenderContext& ctx, Rectangle rect, const std::string& label, bool is_enabled = true,
               ActionStyle style = ActionStyle::Default) {
    const bool is_hovered = is_enabled && hit_test(ctx, rect);
    Color bg = theme::PANEL_ELEVATED;
    Color fg = theme::TEXT_NORMAL;

    switch (style) {
        case ActionStyle::Accent:      bg = blend_colors(theme::ACCENT_SUBDUED, theme::ACCENT_PRIMARY, 0.35f); break;
        case ActionStyle::Danger:      bg = Color{110, 40, 44, 255}; break;
        case ActionStyle::Highlighted: bg = theme::ACCENT_SUBDUED; break;
        default: break;
    }

    if (!is_enabled) {
        bg = theme::PANEL_BG;
        fg = theme::TEXT_FAINT;
    } else if (is_hovered) {
        bg = blend_colors(bg, theme::ACCENT_PRIMARY, IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? 0.5f : 0.25f);
        ctx.pointer_style = MOUSE_CURSOR_POINTING_HAND;
    }

    render_outlined_box(rect, 6, style == ActionStyle::Highlighted ? theme::ACCENT_PRIMARY : theme::BORDER_COLOR, bg);
    const float font_size = 17.0f;
    const float tw = measure_ui_text(ctx, label, font_size);
    draw_ui_text(ctx, label, rect.x + (rect.width - tw) / 2.0f, rect.y + (rect.height - font_size) / 2.0f, font_size, fg);

    return is_hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

bool ui_checkbox(RenderContext& ctx, Rectangle rect, const std::string& label, bool& state, bool enabled = true) {
    const bool hovered = enabled && hit_test(ctx, rect);
    const Rectangle box{rect.x, rect.y + (rect.height - 20) / 2.0f, 20, 20};
    render_outlined_box(box, 4, hovered ? theme::ACCENT_PRIMARY : theme::BORDER_COLOR, theme::PANEL_ELEVATED);

    if (state) {
        render_rounded_box(Rectangle{box.x + 5, box.y + 5, 10, 10}, 2, enabled ? theme::ACCENT_PRIMARY : theme::TEXT_FAINT);
    }
    draw_ui_text(ctx, label, rect.x + 30, rect.y + (rect.height - 16) / 2.0f, 16, enabled ? theme::TEXT_NORMAL : theme::TEXT_FAINT);

    if (hovered) ctx.pointer_style = MOUSE_CURSOR_POINTING_HAND;
    if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        state = !state;
        return true;
    }
    return false;
}

void render_meter(const RenderContext& ctx, Rectangle rect, double progress, Color fill, const std::string& text = "") {
    progress = std::clamp(progress, 0.0, 1.0);
    render_rounded_box(rect, rect.height / 2.0f, theme::PANEL_ELEVATED);
    if (progress > 0.0) {
        const float active_w = std::max(rect.height, static_cast<float>(rect.width * progress));
        render_rounded_box(Rectangle{rect.x, rect.y, active_w, rect.height}, rect.height / 2.0f, fill);
    }
    if (!text.empty()) {
        const float tw = measure_ui_text(ctx, text, 14);
        draw_ui_text(ctx, text, rect.x + (rect.width - tw) / 2.0f, rect.y + (rect.height - 14) / 2.0f, 14, theme::TEXT_NORMAL);
    }
}

bool check_key_repeat(int key) {
    static double repeat_fire_time[512] = {};
    if (IsKeyPressed(key)) {
        repeat_fire_time[key] = GetTime() + 0.4;
        return true;
    }
    if (IsKeyDown(key) && GetTime() >= repeat_fire_time[key]) {
        repeat_fire_time[key] = GetTime() + 0.035;
        return true;
    }
    return false;
}

bool ui_text_input(RenderContext& ctx, int element_id, Rectangle rect, std::string& buffer, const char* hint, bool enabled = true) {
    const bool is_hovered = enabled && hit_test(ctx, rect);
    bool submitted = false;

    if (is_hovered) ctx.pointer_style = MOUSE_CURSOR_IBEAM;
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        if (is_hovered) {
            ctx.active_focus_id = element_id;
            ctx.caret_pos = buffer.size();
        } else if (ctx.active_focus_id == element_id) {
            ctx.active_focus_id = -1;
        }
    }

    const bool has_focus = enabled && (ctx.active_focus_id == element_id);
    if (has_focus) {
        ctx.caret_pos = std::min(ctx.caret_pos, buffer.size());
        for (int ch = GetCharPressed(); ch > 0; ch = GetCharPressed()) {
            if (ch >= 32 && ch <= 126) {
                buffer.insert(buffer.begin() + static_cast<std::ptrdiff_t>(ctx.caret_pos), static_cast<char>(ch));
                ++ctx.caret_pos;
            }
        }

        const bool has_modifier = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL) ||
                                  IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_RIGHT_SUPER);

        if (has_modifier && IsKeyPressed(KEY_V)) {
            const char* clip = GetClipboardText();
            if (clip) {
                std::string sanitized;
                for (const char* p = clip; *p != '\0'; ++p) {
                    const auto c = static_cast<unsigned char>(*p);
                    if (c >= 32 && c <= 126) sanitized.push_back(static_cast<char>(c));
                }
                buffer.insert(ctx.caret_pos, sanitized);
                ctx.caret_pos += sanitized.size();
            }
        }
        if (has_modifier && IsKeyPressed(KEY_A)) {
            buffer.clear();
            ctx.caret_pos = 0;
        }
        if (check_key_repeat(KEY_BACKSPACE) && ctx.caret_pos > 0) {
            buffer.erase(ctx.caret_pos - 1, 1);
            --ctx.caret_pos;
        }
        if (check_key_repeat(KEY_DELETE) && ctx.caret_pos < buffer.size()) {
            buffer.erase(ctx.caret_pos, 1);
        }
        if (check_key_repeat(KEY_LEFT) && ctx.caret_pos > 0) --ctx.caret_pos;
        if (check_key_repeat(KEY_RIGHT) && ctx.caret_pos < buffer.size()) ++ctx.caret_pos;
        if (IsKeyPressed(KEY_HOME)) ctx.caret_pos = 0;
        if (IsKeyPressed(KEY_END)) ctx.caret_pos = buffer.size();

        if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) {
            submitted = true;
            ctx.active_focus_id = -1;
        }
        if (IsKeyPressed(KEY_ESCAPE)) ctx.active_focus_id = -1;
    }

    Color border = has_focus ? theme::ACCENT_PRIMARY
                             : (is_hovered ? blend_colors(theme::BORDER_COLOR, theme::ACCENT_PRIMARY, 0.4f)
                                           : theme::BORDER_COLOR);

    render_outlined_box(rect, 6, border, enabled ? theme::BG_DARK : theme::PANEL_BG);

    const float font_size = 17.0f;
    const float pad = 10.0f;
    const float inner_w = rect.width - 2.0f * pad;
    float offset_x = 0;

    if (has_focus) {
        const float cursor_px = measure_ui_text(ctx, buffer.substr(0, std::min(ctx.caret_pos, buffer.size())), font_size);
        if (cursor_px > inner_w) offset_x = cursor_px - inner_w;
    }

    BeginScissorMode(static_cast<int>(rect.x + pad), static_cast<int>(rect.y),
                     static_cast<int>(std::max(1.0f, inner_w)), static_cast<int>(rect.height));

    if (buffer.empty() && !has_focus) {
        draw_ui_text(ctx, hint, rect.x + pad, rect.y + (rect.height - font_size) / 2.0f, font_size, theme::TEXT_FAINT);
    } else {
        draw_ui_text(ctx, buffer, rect.x + pad - offset_x, rect.y + (rect.height - font_size) / 2.0f, font_size,
                     enabled ? theme::TEXT_NORMAL : theme::TEXT_FAINT);
        if (has_focus && std::fmod(GetTime(), 1.0) < 0.55) {
            const float cursor_px = measure_ui_text(ctx, buffer.substr(0, std::min(ctx.caret_pos, buffer.size())), font_size);
            DrawRectangleRec(Rectangle{rect.x + pad - offset_x + cursor_px, rect.y + 7, 1.5f, rect.height - 14}, theme::ACCENT_PRIMARY);
        }
    }
    EndScissorMode();
    return submitted;
}

float render_input_row(RenderContext& ctx, int id, float x, float y, float w, const char* label,
                       std::string& value, const char* placeholder, bool enabled = true) {
    draw_ui_text(ctx, label, x, y, 14, theme::TEXT_MUTED);
    ui_text_input(ctx, id, Rectangle{x, y + 18, w, 34}, value, placeholder, enabled);
    return y + 18 + 34 + 14;
}

void render_card_frame(const RenderContext& ctx, Rectangle rect, const std::string& header) {
    render_outlined_box(rect, 10, theme::BORDER_COLOR, theme::PANEL_BG);
    if (!header.empty()) {
        draw_ui_text(ctx, header, rect.x + 18, rect.y + 14, 18, theme::TEXT_NORMAL);
    }
}

// --- GPU Textures ---

class MonochromeTexture {
public:
    MonochromeTexture(int w, int h, const std::vector<uint8_t>& raw_pixels, bool bilinear)
        : width_(w), height_(h) {
        if (raw_pixels.size() < static_cast<size_t>(w * h)) {
            throw std::invalid_argument("Provided pixel buffer too small for dimensions");
        }
        Image img{};
        img.data = const_cast<uint8_t*>(raw_pixels.data());
        img.width = w;
        img.height = h;
        img.mipmaps = 1;
        img.format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE;
        handle_ = LoadTextureFromImage(img);
        SetTextureFilter(handle_, bilinear ? TEXTURE_FILTER_BILINEAR : TEXTURE_FILTER_POINT);
    }

    ~MonochromeTexture() { UnloadTexture(handle_); }
    MonochromeTexture(const MonochromeTexture&) = delete;
    MonochromeTexture& operator=(const MonochromeTexture&) = delete;

    void upload(const std::vector<uint8_t>& raw_pixels) {
        if (raw_pixels.size() >= static_cast<size_t>(width_ * height_)) {
            UpdateTexture(handle_, raw_pixels.data());
        }
    }

    const Texture2D& handle() const { return handle_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_ = 0;
    int height_ = 0;
    Texture2D handle_{};
};

// --- Application State ---

enum class ActiveTab { Checkpoints, DrawingCanvas, DataExplorer, Trainer };

enum InputElementId {
    ID_SEARCH_ROOT = 1,
    ID_FILTER_NAME,
    ID_RESUME_PATH,
    ID_SPEC_ARCH,
    ID_TRAIN_EPOCHS,
    ID_TRAIN_LR,
    ID_MAX_SAMPLES,
    ID_DATASET_DIR,
    ID_EXPORT_PATH,
    ID_DATASET_EXPLORER_DIR,
    ID_INDEX_JUMP
};

struct ApplicationState {
    RenderContext ctx;
    ActiveTab current_tab = ActiveTab::Checkpoints;
    fs::path workspace_root;

    // Checkpoints browser
    std::string root_input;
    std::string filter_input;
    DirectoryScan file_scan;
    fs::path selected_checkpoint;
    float checkpoint_scroll = 0;
    StatusNotice checkpoint_notice;

    // Active inference model
    std::optional<Network> active_model;
    fs::path loaded_checkpoint_path;

    // Drawing pad
    std::vector<uint8_t> draw_buffer = std::vector<uint8_t>(CANVAS_SIZE * CANVAS_SIZE, 0);
    std::unique_ptr<MonochromeTexture> canvas_gpu_tex;
    bool canvas_needs_sync = true;
    bool is_painting = false;
    Vector2 last_mouse_pos{0, 0};
    std::array<float, MNIST_INPUTS> active_digit_input{};
    bool digit_input_ready = false;
    std::optional<ClassPrediction> current_prediction;
    std::string inference_error;

    // Trainer state
    TrainingEngine trainer;
    bool resume_existing = false;
    std::string resume_input;
    std::string architecture_input = "784 512 256 128 64 10";
    neural::WeightInit selected_init = neural::WeightInit::He;
    std::string epoch_input = "5";
    std::string lr_input = "0.01";
    std::string max_samples_input = "0";
    std::string dataset_path_input;
    std::string save_target_input;
    bool run_eval_after = true;
    bool save_on_abort = true;
    bool use_logarithmic_loss = false;
    float trainer_scroll_y = 0;
    StatusNotice trainer_notice;
    bool test_dataset_found = false;
    bool train_dataset_found = false;
    double dataset_probe_time = -10;
    std::string verified_data_path;
    bool was_training_active = false;

    // Data Explorer
    bool inspect_test_split = true;
    bool initial_dataset_attempt = true;
    std::shared_ptr<const neural::MNISTDataset> training_split;
    std::shared_ptr<const neural::MNISTDataset> testing_split;
    StatusNotice explorer_notice;
    size_t explorer_grid_offset = 0;
    size_t active_sample_index = 0;
    std::string jump_index_input;
    std::vector<uint8_t> grid_raw_bytes;
    std::unique_ptr<MonochromeTexture> grid_gpu_tex;
    bool grid_needs_render = true;
    std::vector<uint8_t> single_sample_bytes = std::vector<uint8_t>(MNIST_INPUTS, 0);
    std::unique_ptr<MonochromeTexture> sample_gpu_tex;
    bool sample_needs_render = true;
    std::optional<ClassPrediction> sample_prediction;
    std::string sample_error_msg;

    AsyncEvaluator evaluator;
    EvaluationResult eval_summary;
    bool eval_summary_valid = false;
    std::string eval_failure_text;

    // Modal dialog
    bool confirm_overwrite_open = false;
    std::string confirmation_dialog_text;
    std::optional<TrainingConfig> queued_training_cfg;
};

const ModelEntry* lookup_checkpoint(const ApplicationState& app, const fs::path& target) {
    for (const auto& item : app.file_scan.entries) {
        if (item.path == target) return &item;
    }
    return nullptr;
}

void refresh_checkpoints(ApplicationState& app) {
    app.file_scan = scan_checkpoint_directory(fs::path(trim_whitespace(app.root_input)));
    if (!app.file_scan.error_message.empty()) {
        app.checkpoint_notice = {app.file_scan.error_message, NoticeLevel::Error};
    } else if (app.file_scan.entries.empty()) {
        app.checkpoint_notice = {"No checkpoints matching format found in directory.", NoticeLevel::Warning};
    } else {
        std::string s = "Found " + std::to_string(app.file_scan.entries.size()) + " checkpoint(s)";
        if (app.file_scan.non_checkpoints > 0) s += ", ignored " + std::to_string(app.file_scan.non_checkpoints) + " text file(s)";
        if (app.file_scan.unreadable_files > 0) s += ", " + std::to_string(app.file_scan.unreadable_files) + " unreadable";
        if (app.file_scan.reached_limit) s += " (depth limit reached)";
        app.checkpoint_notice = {s, app.file_scan.reached_limit ? NoticeLevel::Warning : NoticeLevel::Success};
    }

    if (!app.selected_checkpoint.empty() && lookup_checkpoint(app, app.selected_checkpoint) == nullptr) {
        app.selected_checkpoint.clear();
    }
}

void trigger_canvas_classification(ApplicationState& app) {
    try {
        const DigitSample sample = preprocess_digit_image(app.draw_buffer, CANVAS_SIZE);
        app.digit_input_ready = !sample.empty;
        app.active_digit_input = sample.pixels;
        app.inference_error.clear();

        if (sample.empty || !app.active_model) {
            app.current_prediction.reset();
            return;
        }
        app.current_prediction = predict_class(*app.active_model, tensor_from_pixels(sample.pixels));
    } catch (const std::exception& e) {
        app.current_prediction.reset();
        app.inference_error = e.what();
    }
}

void activate_model_checkpoint(ApplicationState& app, const fs::path& target) {
    try {
        Network net = load_checkpoint(target);
        if (!net.is_mnist_compatible()) {
            throw std::runtime_error("Architecture " + format_architecture(net.architecture()) +
                                     " is valid, but digit inference requires 784-in / 10-out.");
        }
        app.active_model.emplace(std::move(net));
        app.loaded_checkpoint_path = target;
        app.checkpoint_notice = {"Loaded " + target.filename().string() + " (" +
                                 format_architecture(app.active_model->architecture()) + ")", NoticeLevel::Success};
        app.canvas_needs_sync = true;
        app.sample_needs_render = true;
        app.eval_summary_valid = false;
        app.eval_failure_text.clear();
    } catch (const std::exception& e) {
        app.checkpoint_notice = {"Failed to load " + target.filename().string() + ": " + e.what(), NoticeLevel::Error};
        for (auto& entry : app.file_scan.entries) {
            if (entry.path == target) {
                entry.valid = false;
                entry.error_msg = e.what();
            }
        }
    }
}

std::optional<TrainingConfig> assemble_trainer_params(const ApplicationState& app, std::string& err) {
    TrainingConfig cfg;
    try {
        if (!parse_size_t(app.epoch_input, cfg.epoch_count) || cfg.epoch_count == 0 || cfg.epoch_count > 10000) {
            throw std::runtime_error("Epochs must be an integer between 1 and 10000");
        }
        if (!parse_float_val(app.lr_input, cfg.lr) || !std::isfinite(cfg.lr) || cfg.lr <= 0.0f || cfg.lr > 10.0f) {
            throw std::runtime_error("Learning rate must be in range (0, 10]");
        }
        if (!parse_size_t(app.max_samples_input, cfg.max_epoch_samples)) {
            throw std::runtime_error("Max samples must be a non-negative integer (0 = all)");
        }

        cfg.data_dir = fs::path(trim_whitespace(app.dataset_path_input));
        const DatasetBundle bundle = resolve_dataset_files(cfg.data_dir);
        std::error_code ec;

        if (!fs::is_directory(cfg.data_dir, ec)) {
            throw std::runtime_error("Specified dataset folder does not exist");
        }
        if (!bundle.has_train()) {
            throw std::runtime_error(std::string("Missing training files (expected ") + FILE_TRAIN_IMAGES + " and labels)");
        }
        cfg.run_eval_on_completion = app.run_eval_after && bundle.has_test();

        if (app.resume_existing) {
            const fs::path resume_p(trim_whitespace(app.resume_input));
            if (resume_p.empty()) throw std::runtime_error("Select a checkpoint to resume");
            std::ifstream f(resume_p);
            if (!f) throw std::runtime_error("Unable to read resume checkpoint: " + resume_p.string());
            cfg.architecture = read_checkpoint_header(f);
            if (cfg.architecture.front() != MNIST_INPUTS || cfg.architecture.back() != MNIST_CLASSES) {
                throw std::runtime_error("Resume checkpoint is not a 784 -> 10 classifier");
            }
            cfg.resume_checkpoint = resume_p;
        } else {
            cfg.architecture = parse_architecture_spec(app.architecture_input);
            if (cfg.architecture.front() != MNIST_INPUTS || cfg.architecture.back() != MNIST_CLASSES) {
                throw std::runtime_error("MNIST architecture must start at 784 and end at 10");
            }
            cfg.weight_init = app.selected_init;
        }

        std::string out_str = trim_whitespace(app.save_target_input);
        if (out_str.empty()) throw std::runtime_error("Target export file cannot be empty");
        cfg.save_target = fs::path(out_str);
        if (!cfg.save_target.has_extension()) cfg.save_target += ".txt";
        if (fs::is_directory(cfg.save_target, ec)) {
            throw std::runtime_error("Target path points to an existing directory");
        }
        cfg.persist_on_early_stop = app.save_on_abort;
    } catch (const std::exception& e) {
        err = e.what();
        return std::nullopt;
    }
    return cfg;
}

void dispatch_training(ApplicationState& app) {
    std::string err;
    std::optional<TrainingConfig> cfg = assemble_trainer_params(app, err);
    if (!cfg) {
        app.trainer_notice = {err, NoticeLevel::Error};
        return;
    }

    std::error_code ec;
    if (fs::exists(cfg->save_target, ec)) {
        app.confirmation_dialog_text = "Target file already exists:\n" + cfg->save_target.string() + "\nOverwrite upon completion?";
        if (cfg->resume_checkpoint && fs::equivalent(*cfg->resume_checkpoint, cfg->save_target, ec)) {
            app.confirmation_dialog_text += "\n(Warning: Overwriting the checkpoint you are resuming from)";
        }
        app.queued_training_cfg = std::move(cfg);
        app.confirm_overwrite_open = true;
        app.ctx.active_focus_id = -1;
        return;
    }
    app.trainer_notice = {"Training initiated.", NoticeLevel::Success};
    app.trainer.start(std::move(*cfg));
}

std::shared_ptr<const neural::MNISTDataset> get_selected_dataset(const ApplicationState& app) {
    return app.inspect_test_split ? app.testing_split : app.training_split;
}

void load_dataset_partition(ApplicationState& app, bool use_test) {
    app.explorer_grid_offset = 0;
    app.active_sample_index = 0;
    app.grid_needs_render = true;
    app.sample_needs_render = true;

    try {
        const fs::path dir(trim_whitespace(app.dataset_path_input));
        const DatasetBundle bundle = resolve_dataset_files(dir);
        const fs::path& img_file = use_test ? bundle.test_images : bundle.train_images;
        const fs::path& lbl_file = use_test ? bundle.test_labels : bundle.train_labels;

        if (!safe_is_file(img_file) || !safe_is_file(lbl_file)) {
            throw std::runtime_error(std::string(use_test ? "Test" : "Train") + " files not found in " + dir.string());
        }

        auto data = std::make_shared<neural::MNISTDataset>(img_file, lbl_file);
        const size_t total = data->size();
        (use_test ? app.testing_split : app.training_split) = std::move(data);
        app.explorer_notice = {
            std::string("Loaded ") + (use_test ? "test" : "train") + " set (" + std::to_string(total) + " images)",
            NoticeLevel::Success
        };
    } catch (const std::exception& e) {
        (use_test ? app.testing_split : app.training_split).reset();
        app.explorer_notice = {e.what(), NoticeLevel::Error};
    }
}

void reload_all_data(ApplicationState& app) {
    app.training_split.reset();
    app.testing_split.reset();
    app.eval_summary_valid = false;
    load_dataset_partition(app, app.inspect_test_split);
}

void switch_partition(ApplicationState& app, bool test) {
    if (app.inspect_test_split == test) return;
    app.inspect_test_split = test;
    app.explorer_grid_offset = 0;
    app.active_sample_index = 0;
    app.grid_needs_render = true;
    app.sample_needs_render = true;
    if (!get_selected_dataset(app)) load_dataset_partition(app, test);
}

void bake_grid_bitmap(std::vector<uint8_t>& buffer, const neural::MNISTDataset& data, size_t start, size_t num_rows) {
    const size_t stride = GRID_COLS * 28;
    buffer.assign(stride * num_rows * 28, 0);

    for (size_t k = 0; k < GRID_COLS * num_rows; ++k) {
        const size_t idx = start + k;
        if (idx >= data.size()) break;

        const neural::Tensor img = data.image(idx);
        const auto v = img.data();
        const size_t cell_x = (k % GRID_COLS) * 28;
        const size_t cell_y = (k / GRID_COLS) * 28;

        for (size_t y = 0; y < 28; ++y) {
            for (size_t x = 0; x < 28; ++x) {
                const float px = std::clamp(v[y * 28 + x], 0.0f, 1.0f);
                buffer[(cell_y + y) * stride + cell_x + x] = static_cast<uint8_t>(std::lround(px * 255.0f));
            }
        }
    }
}

void sync_single_sample_preview(ApplicationState& app, const neural::MNISTDataset& data) {
    app.sample_needs_render = false;
    app.sample_prediction.reset();
    app.sample_error_msg.clear();

    try {
        if (data.size() == 0) return;
        const size_t idx = std::min(app.active_sample_index, data.size() - 1);
        const neural::Tensor img = data.image(idx);
        const auto v = img.data();

        for (size_t i = 0; i < MNIST_INPUTS; ++i) {
            app.single_sample_bytes[i] = static_cast<uint8_t>(std::lround(std::clamp(v[i], 0.0f, 1.0f) * 255.0f));
        }

        if (!app.sample_gpu_tex) {
            app.sample_gpu_tex = std::make_unique<MonochromeTexture>(28, 28, app.single_sample_bytes, false);
        } else {
            app.sample_gpu_tex->upload(app.single_sample_bytes);
        }

        if (app.active_model) {
            app.sample_prediction = predict_class(*app.active_model, img);
        }
    } catch (const std::exception& e) {
        app.sample_error_msg = e.what();
    }
}

void trigger_background_evaluation(ApplicationState& app) {
    const auto ds = get_selected_dataset(app);
    if (!app.active_model || !ds || app.evaluator.is_active()) return;
    app.eval_failure_text.clear();
    try {
        app.evaluator.start(app.active_model->clone(), ds, app.inspect_test_split, app.loaded_checkpoint_path.string());
    } catch (const std::exception& e) {
        app.eval_failure_text = e.what();
    }
}

void locate_worst_performing_digit(const EvaluationResult& r, size_t& class_idx, double& lowest_acc) {
    class_idx = 0;
    lowest_acc = 101.0;

    for (size_t row = 0; row < MNIST_CLASSES; ++row) {
        size_t row_total = 0;
        for (size_t col = 0; col < MNIST_CLASSES; ++col) row_total += r.confusion_matrix[row][col];
        if (row_total == 0) continue;

        const double acc = 100.0 * static_cast<double>(r.confusion_matrix[row][row]) / static_cast<double>(row_total);
        if (acc < lowest_acc) {
            lowest_acc = acc;
            class_idx = row;
        }
    }
}

// --- Visual Components & Charts ---

void render_loss_plot(const RenderContext& ctx, Rectangle rect, const std::vector<LossPoint>& history,
                      double max_x_domain, size_t epoch_samples, size_t total_epochs, bool log_scale) {
    render_outlined_box(rect, 8, theme::BORDER_COLOR, theme::BG_DARK);
    if (history.size() < 2 || !(max_x_domain > 0.0)) {
        draw_ui_text(ctx, "Loss profile will plot as training executes", rect.x + 12, rect.y + rect.height / 2.0f - 8, 14, theme::TEXT_FAINT);
        return;
    }

    const float ml = 46.0f, mr = 12.0f, mt = 12.0f, mb = 24.0f;
    const Rectangle area{rect.x + ml, rect.y + mt, rect.width - ml - mr, rect.height - mt - mb};
    if (area.width < 40 || area.height < 30) return;

    double min_y = std::numeric_limits<double>::max();
    double max_y = 0.0;
    for (const auto& pt : history) {
        const double v = static_cast<double>(pt.loss);
        if (std::isfinite(v)) {
            min_y = std::min(min_y, v);
            max_y = std::max(max_y, v);
        }
    }
    if (min_y > max_y) return;

    double floor_y = 0.0;
    double ceil_y = max_y > 0.0 ? max_y * 1.08 : 1.0;
    if (log_scale) {
        min_y = std::max(min_y, 1e-4);
        max_y = std::max(max_y, min_y * 1.5);
        floor_y = min_y * 0.8;
        ceil_y = max_y * 1.25;
    }

    auto to_plot_y = [&](double v) -> float {
        double factor = 0.0;
        if (log_scale) {
            v = std::max(v, floor_y);
            factor = (std::log(v) - std::log(floor_y)) / (std::log(ceil_y) - std::log(floor_y));
        } else {
            factor = (v - floor_y) / (ceil_y - floor_y);
        }
        return area.y + area.height * static_cast<float>(1.0 - std::clamp(factor, 0.0, 1.0));
    };

    auto to_plot_x = [&](double s) -> float {
        return area.x + area.width * static_cast<float>(std::clamp(s / max_x_domain, 0.0, 1.0));
    };

    const Color line_col = Fade(theme::BORDER_COLOR, 0.7f);

    std::vector<double> y_ticks;
    if (log_scale) {
        const int p0 = static_cast<int>(std::floor(std::log10(floor_y)));
        const int p1 = static_cast<int>(std::ceil(std::log10(ceil_y)));
        for (int p = p0; p <= p1; ++p) {
            for (double multiplier : {1.0, 2.0, 5.0}) {
                const double v = multiplier * std::pow(10.0, p);
                if (v >= floor_y && v <= ceil_y) y_ticks.push_back(v);
            }
        }
    } else {
        const double step = compute_nice_step(ceil_y - floor_y, 4);
        for (double v = std::ceil(floor_y / step) * step; v <= ceil_y + 1e-12; v += step) {
            y_ticks.push_back(v);
        }
    }

    float last_rendered_y = -9999.0f;
    for (double val : y_ticks) {
        const float py = to_plot_y(val);
        DrawLineEx(Vector2{area.x, py}, Vector2{area.x + area.width, py}, 1.0f, line_col);
        if (std::fabs(py - last_rendered_y) >= 14.0f) {
            const std::string s = format_tick(val);
            draw_ui_text(ctx, s, area.x - 6 - measure_ui_text(ctx, s, 11), py - 6, 11, theme::TEXT_FAINT);
            last_rendered_y = py;
        }
    }

    if (epoch_samples > 0 && total_epochs > 0) {
        const double ep_step = std::max(1.0, compute_nice_step(static_cast<double>(total_epochs), 5));
        for (double ep = 0.0; ep <= static_cast<double>(total_epochs) + 1e-9; ep += ep_step) {
            const float px = to_plot_x(ep * static_cast<double>(epoch_samples));
            DrawLineEx(Vector2{px, area.y}, Vector2{px, area.y + area.height}, 1.0f, line_col);
            const std::string lbl = format_tick(ep);
            draw_ui_text(ctx, lbl, px - measure_ui_text(ctx, lbl, 11) / 2.0f, area.y + area.height + 5, 11, theme::TEXT_FAINT);
        }
    }
    draw_ui_text(ctx, "epoch", rect.x + 6, rect.y + rect.height - 16, 11, theme::TEXT_FAINT);

    std::vector<double> smoothed_curve(history.size());
    smoothed_curve[0] = static_cast<double>(history[0].loss);
    constexpr double EMA_ALPHA = 0.15;
    for (size_t i = 1; i < history.size(); ++i) {
        smoothed_curve[i] = smoothed_curve[i - 1] + EMA_ALPHA * (static_cast<double>(history[i].loss) - smoothed_curve[i - 1]);
    }

    BeginScissorMode(static_cast<int>(area.x), static_cast<int>(area.y) - 2,
                     static_cast<int>(area.width) + 2, static_cast<int>(area.height) + 4);

    for (size_t i = 1; i < history.size(); ++i) {
        DrawLineEx(Vector2{to_plot_x(history[i - 1].samples), to_plot_y(static_cast<double>(history[i - 1].loss))},
                   Vector2{to_plot_x(history[i].samples), to_plot_y(static_cast<double>(history[i].loss))},
                   1.0f, Fade(theme::ACCENT_PRIMARY, 0.30f));
    }
    for (size_t i = 1; i < history.size(); ++i) {
        DrawLineEx(Vector2{to_plot_x(history[i - 1].samples), to_plot_y(smoothed_curve[i - 1])},
                   Vector2{to_plot_x(history[i].samples), to_plot_y(smoothed_curve[i])},
                   2.5f, theme::ACCENT_PRIMARY);
    }
    DrawCircleV(Vector2{to_plot_x(history.back().samples), to_plot_y(smoothed_curve.back())}, 3.5f, theme::TEXT_NORMAL);
    EndScissorMode();

    const std::string cur_val = "EMA: " + format_fixed(smoothed_curve.back(), 4);
    draw_ui_text(ctx, cur_val, area.x + area.width - measure_ui_text(ctx, cur_val, 12) - 4, area.y + 2, 12, theme::TEXT_MUTED);
}

void render_confusion_grid(const RenderContext& ctx, Rectangle rect, const EvaluationResult& eval) {
    const float label_w = 14.0f;
    const float cell_sz = (std::min(rect.width, rect.height) - label_w) / static_cast<float>(MNIST_CLASSES);
    if (cell_sz < 4.0f) return;

    const float ox = rect.x + label_w;
    const float oy = rect.y + label_w;

    for (size_t i = 0; i < MNIST_CLASSES; ++i) {
        const std::string s = std::to_string(i);
        draw_ui_text(ctx, s, ox + static_cast<float>(i) * cell_sz + (cell_sz - 7.0f) / 2.0f, rect.y, 10, theme::TEXT_FAINT);
        draw_ui_text(ctx, s, rect.x + 3, oy + static_cast<float>(i) * cell_sz + (cell_sz - 10.0f) / 2.0f, 10, theme::TEXT_FAINT);
    }

    for (size_t r = 0; r < MNIST_CLASSES; ++r) {
        size_t row_samples = 0;
        for (size_t c = 0; c < MNIST_CLASSES; ++c) row_samples += eval.confusion_matrix[r][c];

        for (size_t c = 0; c < MNIST_CLASSES; ++c) {
            const double ratio = row_samples > 0
                ? static_cast<double>(eval.confusion_matrix[r][c]) / static_cast<double>(row_samples)
                : 0.0;
            const float heat = static_cast<float>(std::sqrt(ratio));
            const Color base = (r == c) ? theme::STATUS_GOOD : theme::STATUS_ERR;
            DrawRectangleRec(Rectangle{ox + static_cast<float>(c) * cell_sz, oy + static_cast<float>(r) * cell_sz, cell_sz - 1, cell_sz - 1},
                             blend_colors(theme::PANEL_ELEVATED, base, heat));
        }
    }

    const Rectangle bounds{ox, oy, cell_sz * MNIST_CLASSES, cell_sz * MNIST_CLASSES};
    if (hit_test(ctx, bounds)) {
        const Vector2 m = GetMousePosition();
        const auto c = static_cast<size_t>(std::clamp((m.x - ox) / cell_sz, 0.0f, 9.0f));
        const auto r = static_cast<size_t>(std::clamp((m.y - oy) / cell_sz, 0.0f, 9.0f));

        const std::string tip = "Truth: " + std::to_string(r) + " -> Pred: " + std::to_string(c) +
                                " (" + std::to_string(eval.confusion_matrix[r][c]) + " samples)";
        const float w = measure_ui_text(ctx, tip, 13) + 16;
        const float x = std::min(m.x + 14, static_cast<float>(GetScreenWidth()) - w - 4);

        render_outlined_box(Rectangle{x, m.y + 14, w, 24}, 5, theme::BORDER_COLOR, theme::BG_DARK);
        draw_ui_text(ctx, tip, x + 8, m.y + 19, 13, theme::TEXT_NORMAL);
    }
}

// --- Layouts and Views ---

void draw_page_heading(const RenderContext& ctx, Rectangle bounds, const char* title, const char* subtitle) {
    draw_ui_text(ctx, title, bounds.x, bounds.y, 30, theme::TEXT_NORMAL);
    draw_ui_text(ctx, subtitle, bounds.x, bounds.y + 38, 16, theme::TEXT_MUTED);
}

void render_app_sidebar(ApplicationState& app, float view_height) {
    RenderContext& ctx = app.ctx;
    const float sidebar_width = 210.0f;
    DrawRectangleRec(Rectangle{0, 0, sidebar_width, view_height}, theme::SIDEBAR_BG);
    DrawRectangleRec(Rectangle{sidebar_width - 1, 0, 1, view_height}, theme::BORDER_COLOR);

    draw_ui_text(ctx, "NeuralEngine", 22, 24, 22, theme::TEXT_NORMAL);
    draw_ui_text(ctx, "STUDIO", 22, 50, 14, theme::ACCENT_PRIMARY);

    struct NavItem { ActiveTab tab; const char* name; };
    const NavItem items[] = {
        {ActiveTab::Checkpoints,   "Models"},
        {ActiveTab::DrawingCanvas, "Draw & Predict"},
        {ActiveTab::DataExplorer,  "MNIST Data"},
        {ActiveTab::Trainer,       "Train"}
    };

    float btn_y = 100.0f;
    for (const auto& item : items) {
        const Rectangle area{12, btn_y, sidebar_width - 24, 42};
        const bool active = (app.current_tab == item.tab);
        const bool hovered = hit_test(ctx, area);

        if (active) {
            render_rounded_box(area, 8, theme::ACCENT_SUBDUED);
            DrawRectangleRec(Rectangle{12, btn_y + 8, 3, 26}, theme::ACCENT_PRIMARY);
        } else if (hovered) {
            render_rounded_box(area, 8, theme::PANEL_ELEVATED);
        }

        if (hovered) ctx.pointer_style = MOUSE_CURSOR_POINTING_HAND;
        draw_ui_text(ctx, item.name, 32, btn_y + 12, 18, active ? theme::TEXT_NORMAL : theme::TEXT_MUTED);

        if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            app.current_tab = item.tab;
            ctx.active_focus_id = -1;
        }
        btn_y += 48.0f;
    }

    const float footer_y = view_height - 120.0f;
    draw_ui_text(ctx, "ACTIVE MODEL", 22, footer_y, 12, theme::TEXT_FAINT);
    if (app.active_model) {
        draw_ui_text(ctx, truncate_middle(ctx, app.loaded_checkpoint_path.filename().string(), 15, sidebar_width - 44), 22, footer_y + 18, 15, theme::TEXT_NORMAL);
        draw_ui_text(ctx, truncate_middle(ctx, format_architecture(app.active_model->architecture()), 13, sidebar_width - 44), 22, footer_y + 38, 13, theme::TEXT_MUTED);
    } else {
        draw_ui_text(ctx, "None", 22, footer_y + 18, 15, theme::TEXT_MUTED);
    }

    const auto snap = app.trainer.get_snapshot(false);
    draw_ui_text(ctx, "ENGINE STATUS", 22, footer_y + 64, 12, theme::TEXT_FAINT);
    draw_ui_text(ctx, stringify_phase(snap.phase), 22, footer_y + 80, 15,
                 app.trainer.is_busy() ? theme::ACCENT_PRIMARY : theme::TEXT_MUTED);
}

void render_models_tab(ApplicationState& app, Rectangle layout) {
    RenderContext& ctx = app.ctx;
    draw_page_heading(ctx, layout, "Model Checkpoints", "Manage serialized neural network weights and load parameters into the runtime.");

    const float y_offset = layout.y + 80;
    const float scan_btn_w = 110;

    ui_text_input(ctx, ID_SEARCH_ROOT, Rectangle{layout.x, y_offset + 18, layout.width - scan_btn_w - 12, 34}, app.root_input, "Directory to traverse");
    draw_ui_text(ctx, "SEARCH PATH", layout.x, y_offset, 14, theme::TEXT_MUTED);

    if (ui_button(ctx, Rectangle{layout.x + layout.width - scan_btn_w, y_offset + 18, scan_btn_w, 34}, "Refresh", true, ActionStyle::Accent)) {
        refresh_checkpoints(app);
    }

    const float filter_y = y_offset + 66;
    draw_ui_text(ctx, "FILTER", layout.x, filter_y, 14, theme::TEXT_MUTED);
    ui_text_input(ctx, ID_FILTER_NAME, Rectangle{layout.x, filter_y + 18, layout.width, 34}, app.filter_input, "Filter by substring or relative path");

    const std::string query = to_lower(trim_whitespace(app.filter_input));
    std::vector<const ModelEntry*> matched_files;
    for (const auto& item : app.file_scan.entries) {
        if (query.empty() || to_lower(item.rel_path).find(query) != std::string::npos) {
            matched_files.push_back(&item);
        }
    }

    const float main_y = filter_y + 66;
    const float footer_h = 26;
    const float main_h = std::max(120.0f, layout.y + layout.height - main_y - footer_h - 8);
    const float list_w = std::max(320.0f, layout.width * 0.56f);

    const Rectangle list_bounds{layout.x, main_y, list_w, main_h};
    const Rectangle details_bounds{layout.x + list_w + 16, main_y, layout.width - list_w - 16, main_h};

    if (ctx.active_focus_id == -1 && !ctx.input_blocked && !matched_files.empty()) {
        size_t cursor = matched_files.size();
        for (size_t i = 0; i < matched_files.size(); ++i) {
            if (matched_files[i]->path == app.selected_checkpoint) cursor = i;
        }
        if (IsKeyPressed(KEY_DOWN)) cursor = (cursor >= matched_files.size()) ? 0 : std::min(cursor + 1, matched_files.size() - 1);
        if (IsKeyPressed(KEY_UP)) cursor = (cursor >= matched_files.size() || cursor == 0) ? 0 : cursor - 1;
        if (cursor < matched_files.size()) app.selected_checkpoint = matched_files[cursor]->path;
        if (IsKeyPressed(KEY_ENTER) && cursor < matched_files.size() && matched_files[cursor]->valid) {
            activate_model_checkpoint(app, matched_files[cursor]->path);
        }
    }

    render_outlined_box(list_bounds, 10, theme::BORDER_COLOR, theme::PANEL_BG);
    const float row_h = 58.0f;
    const float inner_content_h = static_cast<float>(matched_files.size()) * row_h;
    const Rectangle inner_clip{list_bounds.x + 6, list_bounds.y + 6, list_bounds.width - 12, list_bounds.height - 12};

    if (hit_test(ctx, list_bounds)) {
        app.checkpoint_scroll -= GetMouseWheelMove() * 40.0f;
    }
    app.checkpoint_scroll = std::clamp(app.checkpoint_scroll, 0.0f, std::max(0.0f, inner_content_h - inner_clip.height));

    if (matched_files.empty()) {
        const char* msg = app.file_scan.entries.empty()
            ? "No compatible model files discovered. Check search root and click Refresh."
            : "No records match search parameters.";
        draw_multiline_text(ctx, msg, list_bounds.x + 20, list_bounds.y + 20, list_bounds.width - 40, 16, theme::TEXT_MUTED);
    } else {
        BeginScissorMode(static_cast<int>(inner_clip.x), static_cast<int>(inner_clip.y),
                         static_cast<int>(inner_clip.width), static_cast<int>(inner_clip.height));
        const Rectangle prev_clip = ctx.scissor_bounds;
        ctx.scissor_bounds = inner_clip;

        for (size_t i = 0; i < matched_files.size(); ++i) {
            const float item_y = inner_clip.y + static_cast<float>(i) * row_h - app.checkpoint_scroll;
            if (item_y + row_h < inner_clip.y || item_y > inner_clip.y + inner_clip.height) continue;

            const ModelEntry& item = *matched_files[i];
            const Rectangle row_rect{inner_clip.x, item_y, inner_clip.width - 8, row_h - 4};
            const bool is_selected = (item.path == app.selected_checkpoint);
            const bool is_hovered = hit_test(ctx, row_rect);

            if (is_selected) {
                render_rounded_box(row_rect, 6, theme::ACCENT_SUBDUED);
            } else if (is_hovered) {
                render_rounded_box(row_rect, 6, theme::PANEL_ELEVATED);
            }

            if (is_hovered) {
                ctx.pointer_style = MOUSE_CURSOR_POINTING_HAND;
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    app.selected_checkpoint = item.path;
                    ctx.active_focus_id = -1;
                }
            }

            const bool is_loaded = (item.path == app.loaded_checkpoint_path && app.active_model.has_value());
            draw_ui_text(ctx, truncate_middle(ctx, item.filename, 17, row_rect.width - 150), row_rect.x + 12, row_rect.y + 7, 17, theme::TEXT_NORMAL);
            draw_ui_text(ctx, truncate_middle(ctx, item.rel_path, 13, row_rect.width - 24), row_rect.x + 12, row_rect.y + 30, 13, theme::TEXT_MUTED);

            const std::string tag = !item.valid ? "CORRUPT" : (is_loaded ? "ACTIVE" : "");
            if (!tag.empty()) {
                const float tw = measure_ui_text(ctx, tag, 12);
                draw_ui_text(ctx, tag, row_rect.x + row_rect.width - tw - 12, row_rect.y + 9, 12, item.valid ? theme::STATUS_GOOD : theme::STATUS_ERR);
            }
            if (item.valid && !is_loaded) {
                const std::string arch_str = format_architecture(item.arch);
                const float aw = std::min(measure_ui_text(ctx, arch_str, 12), 140.0f);
                draw_ui_text(ctx, truncate_middle(ctx, arch_str, 12, 140), row_rect.x + row_rect.width - aw - 12, row_rect.y + 9, 12, theme::TEXT_FAINT);
            }
        }
        ctx.scissor_bounds = prev_clip;
        EndScissorMode();
    }

    render_card_frame(ctx, details_bounds, "Parameters & Meta");
    const ModelEntry* inspected = lookup_checkpoint(app, app.selected_checkpoint);
    float dy = details_bounds.y + 50.0f;
    const float dx = details_bounds.x + 18.0f;
    const float dw = details_bounds.width - 36.0f;

    if (!inspected) {
        draw_multiline_text(ctx, "Pick a checkpoint to inspect its topology.", dx, dy, dw, 16, theme::TEXT_MUTED);
    } else {
        draw_ui_text(ctx, truncate_middle(ctx, inspected->filename, 20, dw), dx, dy, 20, theme::TEXT_NORMAL);
        dy += 30.0f;

        draw_ui_text(ctx, "FILE LOCATION", dx, dy, 12, theme::TEXT_FAINT);
        dy += 16.0f;
        dy += draw_multiline_text(ctx, inspected->path.string(), dx, dy, dw, 14, theme::TEXT_MUTED) + 8.0f;

        draw_ui_text(ctx, "SIZE ON DISK", dx, dy, 12, theme::TEXT_FAINT);
        draw_ui_text(ctx, format_bytes(inspected->file_bytes), dx + 90.0f, dy - 2.0f, 15, theme::TEXT_NORMAL);
        dy += 24.0f;

        if (inspected->valid) {
            draw_ui_text(ctx, "TOPOLOGY", dx, dy, 12, theme::TEXT_FAINT);
            dy += 16.0f;
            dy += draw_multiline_text(ctx, format_architecture(inspected->arch), dx, dy, dw, 16, theme::TEXT_NORMAL) + 4.0f;

            draw_ui_text(ctx, "PARAMETERS", dx, dy, 12, theme::TEXT_FAINT);
            draw_ui_text(ctx, std::to_string(calculate_params(inspected->arch)), dx + 100.0f, dy - 2.0f, 15, theme::TEXT_NORMAL);
            dy += 28.0f;

            const bool compatible = inspected->arch.front() == MNIST_INPUTS && inspected->arch.back() == MNIST_CLASSES;
            if (!compatible) {
                dy += draw_multiline_text(ctx, "Incompatible with MNIST (must map 784 to 10)", dx, dy, dw, 14, theme::STATUS_WARN) + 6.0f;
            }

            if (ui_button(ctx, Rectangle{dx, dy, dw, 38}, "Load into Memory", compatible, ActionStyle::Accent)) {
                activate_model_checkpoint(app, inspected->path);
            }
            dy += 46.0f;

            if (ui_button(ctx, Rectangle{dx, dy, dw, 38}, "Resume Training Run", compatible)) {
                app.resume_existing = true;
                app.resume_input = inspected->path.string();
                app.save_target_input = (inspected->path.parent_path() / (inspected->path.stem().string() + "_resumed.txt")).string();
                app.current_tab = ActiveTab::Trainer;
                ctx.active_focus_id = -1;
            }
        } else {
            draw_ui_text(ctx, "DESERIALIZATION ERROR", dx, dy, 12, theme::TEXT_FAINT);
            dy += 16.0f;
            draw_multiline_text(ctx, inspected->error_msg.empty() ? "Unspecified parsing failure" : inspected->error_msg,
                                dx, dy, dw, 15, theme::STATUS_ERR);
        }
    }

    draw_ui_text(ctx, truncate_middle(ctx, app.checkpoint_notice.message, 15, layout.width),
                 layout.x, layout.y + layout.height - footer_h, 15, resolve_notice_color(app.checkpoint_notice.level));
}

void render_drawing_tab(ApplicationState& app, Rectangle layout) {
    RenderContext& ctx = app.ctx;
    draw_page_heading(ctx, layout, "Inference Canvas", "Draw a character below. The preprocessor will crop and center mass before prediction.");

    const float canvas_dim = std::clamp(std::min(layout.width * 0.46f, layout.height - 190.0f), 200.0f, 520.0f);
    const Rectangle canvas_area{layout.x, layout.y + 84, canvas_dim, canvas_dim};

    const Vector2 mouse = GetMousePosition();
    const bool cursor_inside = hit_test(ctx, canvas_area);
    if (cursor_inside) ctx.pointer_style = MOUSE_CURSOR_CROSSHAIR;

    auto map_to_canvas = [&](Vector2 pt) {
        return Vector2{
            (pt.x - canvas_area.x) * static_cast<float>(CANVAS_SIZE) / canvas_area.width,
            (pt.y - canvas_area.y) * static_cast<float>(CANVAS_SIZE) / canvas_area.height
        };
    };

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && cursor_inside) {
        app.is_painting = true;
        app.last_mouse_pos = map_to_canvas(mouse);
        paint_brush_stamp(app.draw_buffer, app.last_mouse_pos.x, app.last_mouse_pos.y, BRUSH_RADIUS);
        app.canvas_needs_sync = true;
        ctx.active_focus_id = -1;
    } else if (app.is_painting && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        const Vector2 pt = map_to_canvas(mouse);
        if (pt.x != app.last_mouse_pos.x || pt.y != app.last_mouse_pos.y) {
            paint_brush_stroke(app.draw_buffer, app.last_mouse_pos.x, app.last_mouse_pos.y, pt.x, pt.y);
            app.last_mouse_pos = pt;
            app.canvas_needs_sync = true;
        }
    }
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT)) app.is_painting = false;

    const float actions_y = canvas_area.y + canvas_dim + 14.0f;
    const bool hotkey_c = (ctx.active_focus_id == -1 && !ctx.input_blocked && IsKeyPressed(KEY_C));
    const bool hotkey_enter = (ctx.active_focus_id == -1 && !ctx.input_blocked && IsKeyPressed(KEY_ENTER));
    bool run_inference = hotkey_enter;

    if (ui_button(ctx, Rectangle{canvas_area.x, actions_y, 130, 38}, "Clear (C)") || hotkey_c) {
        std::fill(app.draw_buffer.begin(), app.draw_buffer.end(), uint8_t{0});
        app.canvas_needs_sync = true;
    }
    if (ui_button(ctx, Rectangle{canvas_area.x + 142, actions_y, 160, 38}, "Evaluate (Enter)", app.active_model.has_value(), ActionStyle::Accent)) {
        run_inference = true;
    }

    if (app.canvas_needs_sync) {
        app.canvas_gpu_tex->upload(app.draw_buffer);
        trigger_canvas_classification(app);
        app.canvas_needs_sync = false;
    } else if (run_inference) {
        trigger_canvas_classification(app);
    }

    render_outlined_box(Rectangle{canvas_area.x - 3, canvas_area.y - 3, canvas_dim + 6, canvas_dim + 6}, 8, theme::BORDER_COLOR, theme::BORDER_COLOR);
    DrawTexturePro(app.canvas_gpu_tex->handle(),
                   Rectangle{0, 0, static_cast<float>(CANVAS_SIZE), static_cast<float>(CANVAS_SIZE)},
                   canvas_area, Vector2{0, 0}, 0.0f, WHITE);

    if (!app.active_model) {
        draw_ui_text(ctx, "No network loaded. Switch to Checkpoints tab to load weights.",
                     canvas_area.x, actions_y + 52, 15, theme::STATUS_WARN);
    } else {
        draw_ui_text(ctx, truncate_middle(ctx, "Loaded: " + app.loaded_checkpoint_path.filename().string(), 15, canvas_dim),
                     canvas_area.x, actions_y + 52, 15, theme::TEXT_MUTED);
    }

    const float panel_x = canvas_area.x + canvas_dim + 36.0f;
    const float panel_w = layout.x + layout.width - panel_x;
    if (panel_w < 260) return;

    const Rectangle pred_panel{panel_x, canvas_area.y - 3, panel_w, canvas_dim + 6 + 52};
    render_card_frame(ctx, pred_panel, "Inference Output");

    const float px = panel_x + 20.0f;
    const auto& pred = app.current_prediction;
    const std::string label_str = pred ? std::to_string(pred->digit) : "-";

    DrawTextEx(ctx.font, label_str.c_str(), Vector2{px, canvas_area.y + 40}, 84, ctx.text_spacing(84),
               pred ? theme::ACCENT_PRIMARY : theme::TEXT_FAINT);

    if (pred) {
        draw_ui_text(ctx, format_fixed(100.0 * static_cast<double>(pred->probabilities[pred->digit]), 1) + "% confidence",
                     px + 90, canvas_area.y + 70, 18, theme::TEXT_NORMAL);
    } else if (!app.inference_error.empty()) {
        draw_multiline_text(ctx, app.inference_error, px + 90, canvas_area.y + 60, panel_w - 130, 14, theme::STATUS_ERR);
    } else if (!app.digit_input_ready) {
        draw_ui_text(ctx, "Canvas is empty", px + 90, canvas_area.y + 70, 16, theme::TEXT_MUTED);
    } else if (!app.active_model) {
        draw_ui_text(ctx, "Ready for model", px + 90, canvas_area.y + 70, 16, theme::TEXT_MUTED);
    }

    const float bar_start_y = canvas_area.y + 150.0f;
    const float row_h = 22.0f;
    const float gap = 8.0f;
    const float bar_x = px + 26.0f;
    const float bar_w = panel_w - 40.0f - 26.0f - 64.0f;

    for (size_t i = 0; i < MNIST_CLASSES; ++i) {
        const float curr_y = bar_start_y + static_cast<float>(i) * (row_h + gap);
        const bool is_chosen = pred && pred->digit == i;
        const double prob = pred ? static_cast<double>(pred->probabilities[i]) : 0.0;

        draw_ui_text(ctx, std::to_string(i), px, curr_y + 2, 17, is_chosen ? theme::ACCENT_PRIMARY : theme::TEXT_MUTED);
        render_meter(ctx, Rectangle{bar_x, curr_y, bar_w, row_h}, prob,
                     is_chosen ? theme::ACCENT_PRIMARY : blend_colors(theme::ACCENT_SUBDUED, theme::TEXT_FAINT, 0.5f));

        draw_ui_text(ctx, pred ? format_fixed(100.0 * prob, 1) + "%" : "-",
                     bar_x + bar_w + 10, curr_y + 3, 15, is_chosen ? theme::TEXT_NORMAL : theme::TEXT_MUTED);
    }

    const float pixel_size = 4.0f;
    const float preview_y = bar_start_y + 10.0f * (row_h + gap) + 6.0f;
    if (preview_y + 28 * pixel_size < pred_panel.y + pred_panel.height) {
        draw_ui_text(ctx, "NORMALIZED INPUT (28x28)", px, preview_y, 12, theme::TEXT_FAINT);
        for (int y = 0; y < 28; ++y) {
            for (int x = 0; x < 28; ++x) {
                const float intensity = app.digit_input_ready ? app.active_digit_input[static_cast<size_t>(y * 28 + x)] : 0.0f;
                const auto gray = static_cast<unsigned char>(std::lround(intensity * 255.0f));
                DrawRectangleRec(Rectangle{px + static_cast<float>(x) * pixel_size, preview_y + 18 + static_cast<float>(y) * pixel_size, pixel_size, pixel_size},
                                 Color{gray, gray, gray, 255});
            }
        }
    }
}

void render_data_tab(ApplicationState& app, Rectangle layout) {
    RenderContext& ctx = app.ctx;
    draw_page_heading(ctx, layout, "Dataset Visualizer", "Explore IDX archives, inspect raw images, and evaluate inference batches.");

    if (app.initial_dataset_attempt) {
        app.initial_dataset_attempt = false;
        load_dataset_partition(app, app.inspect_test_split);
    }

    const float top_y = layout.y + 80.0f;
    draw_ui_text(ctx, "DATASET DIRECTORY", layout.x, top_y, 14, theme::TEXT_MUTED);

    constexpr float SPLIT_BTN_W = 80.0f;
    constexpr float LOAD_BTN_W = 90.0f;
    const float input_w = std::max(120.0f, layout.width - (2.0f * SPLIT_BTN_W + LOAD_BTN_W + 30.0f));

    const bool submitted_path = ui_text_input(ctx, ID_DATASET_EXPLORER_DIR, Rectangle{layout.x, top_y + 18, input_w, 34},
                                              app.dataset_path_input, "Directory with IDX binary archives");

    float btn_x = layout.x + input_w + 10.0f;
    if (ui_button(ctx, Rectangle{btn_x, top_y + 18, SPLIT_BTN_W, 34}, "Test", true,
                  app.inspect_test_split ? ActionStyle::Highlighted : ActionStyle::Default)) {
        switch_partition(app, true);
    }

    btn_x += SPLIT_BTN_W + 10.0f;
    if (ui_button(ctx, Rectangle{btn_x, top_y + 18, SPLIT_BTN_W, 34}, "Train", true,
                  !app.inspect_test_split ? ActionStyle::Highlighted : ActionStyle::Default)) {
        switch_partition(app, false);
    }

    btn_x += SPLIT_BTN_W + 10.0f;
    if (ui_button(ctx, Rectangle{btn_x, top_y + 18, LOAD_BTN_W, 34}, "Load", true, ActionStyle::Accent) || submitted_path) {
        reload_all_data(app);
    }

    const auto dataset = get_selected_dataset(app);
    const float body_y = top_y + 66.0f;
    const float bottom_bar_h = 28.0f;
    const float view_h = std::max(260.0f, layout.y + layout.height - body_y - bottom_bar_h);
    const float grid_panel_w = std::clamp(layout.width * 0.5f, 300.0f, 520.0f);

    const Rectangle grid_bounds{layout.x, body_y, grid_panel_w, view_h};
    const Rectangle right_bounds{layout.x + grid_panel_w + 16, body_y, layout.width - grid_panel_w - 16, view_h};

    render_card_frame(ctx, grid_bounds, "");

    if (!dataset) {
        draw_multiline_text(ctx, "No dataset currently loaded. Provide path and click Load.",
                            grid_bounds.x + 20, grid_bounds.y + 20, grid_bounds.width - 40, 16, theme::TEXT_MUTED);
        render_card_frame(ctx, right_bounds, "");
        const auto& note = !app.eval_failure_text.empty()
            ? StatusNotice{"Evaluation terminated: " + app.eval_failure_text, NoticeLevel::Error}
            : app.explorer_notice;
        draw_ui_text(ctx, truncate_middle(ctx, note.message, 15, layout.width), layout.x, layout.y + layout.height - 24, 15, resolve_notice_color(note.level));
        return;
    }

    const size_t total_samples = dataset->size();
    if (app.active_sample_index >= total_samples) {
        app.active_sample_index = total_samples - 1;
        app.sample_needs_render = true;
    }

    const float cell_dim = std::floor((grid_panel_w - 24.0f) / static_cast<float>(GRID_COLS));
    const auto rows_to_display = static_cast<size_t>(std::clamp(std::floor((view_h - 96.0f) / cell_dim), 1.0f, 10.0f));
    const size_t samples_per_view = GRID_COLS * rows_to_display;

    auto set_inspected_sample = [&](size_t idx) {
        idx = std::min(idx, total_samples - 1);
        if (idx != app.active_sample_index) {
            app.active_sample_index = idx;
            app.sample_needs_render = true;
        }
    };

    auto adjust_index = [&](long long delta) {
        const long long target = static_cast<long long>(app.active_sample_index) + delta;
        set_inspected_sample(static_cast<size_t>(std::clamp<long long>(target, 0, static_cast<long long>(total_samples) - 1)));
    };

    if (ctx.active_focus_id == -1 && !ctx.input_blocked) {
        if (check_key_repeat(KEY_RIGHT)) adjust_index(1);
        if (check_key_repeat(KEY_LEFT)) adjust_index(-1);
        if (check_key_repeat(KEY_DOWN)) adjust_index(static_cast<long long>(GRID_COLS));
        if (check_key_repeat(KEY_UP)) adjust_index(-static_cast<long long>(GRID_COLS));
        if (IsKeyPressed(KEY_PAGE_DOWN)) adjust_index(static_cast<long long>(samples_per_view));
        if (IsKeyPressed(KEY_PAGE_UP)) adjust_index(-static_cast<long long>(samples_per_view));
        if (IsKeyPressed(KEY_HOME)) set_inspected_sample(0);
        if (IsKeyPressed(KEY_END)) set_inspected_sample(total_samples - 1);
    }

    const float pager_y = grid_bounds.y + 12.0f;
    if (ui_button(ctx, Rectangle{grid_bounds.x + 12, pager_y, 64, 32}, "Prev", app.explorer_grid_offset > 0)) {
        adjust_index(-static_cast<long long>(samples_per_view));
    }
    if (ui_button(ctx, Rectangle{grid_bounds.x + 84, pager_y, 64, 32}, "Next", app.explorer_grid_offset + samples_per_view < total_samples)) {
        adjust_index(static_cast<long long>(samples_per_view));
    }

    if (ui_text_input(ctx, ID_INDEX_JUMP, Rectangle{grid_bounds.x + grid_bounds.width - 12 - 110, pager_y, 110, 32}, app.jump_index_input, "Go to #")) {
        size_t parsed_idx = 0;
        if (parse_size_t(app.jump_index_input, parsed_idx) && parsed_idx < total_samples) {
            set_inspected_sample(parsed_idx);
            app.jump_index_input.clear();
        } else {
            app.explorer_notice = {"Index out of bounds [0, " + std::to_string(total_samples - 1) + "]", NoticeLevel::Warning};
        }
    }

    {
        const size_t final_row = (total_samples - 1) / GRID_COLS;
        const size_t max_page_start = final_row + 1 > rows_to_display ? final_row + 1 - rows_to_display : 0;
        size_t current_row_page = app.explorer_grid_offset / GRID_COLS;
        const size_t selected_row = app.active_sample_index / GRID_COLS;

        if (selected_row < current_row_page) {
            current_row_page = selected_row;
        } else if (selected_row >= current_row_page + rows_to_display) {
            current_row_page = selected_row + 1 - rows_to_display;
        }
        current_row_page = std::min(current_row_page, max_page_start);

        if (current_row_page * GRID_COLS != app.explorer_grid_offset) {
            app.explorer_grid_offset = current_row_page * GRID_COLS;
            app.grid_needs_render = true;
        }
    }

    const int surface_w = static_cast<int>(GRID_COLS * 28);
    const int surface_h = static_cast<int>(rows_to_display * 28);

    if (!app.grid_gpu_tex || app.grid_gpu_tex->width() != surface_w || app.grid_gpu_tex->height() != surface_h) {
        app.grid_raw_bytes.assign(static_cast<size_t>(surface_w * surface_h), 0);
        app.grid_gpu_tex = std::make_unique<MonochromeTexture>(surface_w, surface_h, app.grid_raw_bytes, true);
        app.grid_needs_render = true;
    }

    if (app.grid_needs_render) {
        try {
            bake_grid_bitmap(app.grid_raw_bytes, *dataset, app.explorer_grid_offset, rows_to_display);
            app.grid_gpu_tex->upload(app.grid_raw_bytes);
        } catch (const std::exception& e) {
            app.explorer_notice = {e.what(), NoticeLevel::Error};
        }
        app.grid_needs_render = false;
    }

    const Rectangle grid_rect{
        grid_bounds.x + 12,
        grid_bounds.y + 56,
        cell_dim * static_cast<float>(GRID_COLS),
        cell_dim * static_cast<float>(rows_to_display)
    };

    DrawTexturePro(app.grid_gpu_tex->handle(),
                   Rectangle{0, 0, static_cast<float>(surface_w), static_cast<float>(surface_h)},
                   grid_rect, Vector2{0, 0}, 0.0f, WHITE);

    for (size_t c = 0; c <= GRID_COLS; ++c) {
        DrawRectangleRec(Rectangle{grid_rect.x + static_cast<float>(c) * cell_dim - 0.5f, grid_rect.y, 1.0f, grid_rect.height}, theme::BORDER_COLOR);
    }
    for (size_t r = 0; r <= rows_to_display; ++r) {
        DrawRectangleRec(Rectangle{grid_rect.x, grid_rect.y + static_cast<float>(r) * cell_dim - 0.5f, grid_rect.width, 1.0f}, theme::BORDER_COLOR);
    }

    auto get_cell_box = [&](size_t idx) {
        const size_t offset = idx - app.explorer_grid_offset;
        return Rectangle{
            grid_rect.x + static_cast<float>(offset % GRID_COLS) * cell_dim,
            grid_rect.y + static_cast<float>(offset / GRID_COLS) * cell_dim,
            cell_dim,
            cell_dim
        };
    };

    const bool render_error_outlines = app.eval_summary_valid && (app.eval_summary.is_test_set == app.inspect_test_split);
    if (render_error_outlines) {
        for (size_t k = 0; k < samples_per_view && app.explorer_grid_offset + k < total_samples; ++k) {
            if (std::binary_search(app.eval_summary.misclassified_indices.begin(),
                                   app.eval_summary.misclassified_indices.end(),
                                   app.explorer_grid_offset + k)) {
                const Rectangle c_rect = get_cell_box(app.explorer_grid_offset + k);
                DrawRectangleLinesEx(Rectangle{c_rect.x + 1, c_rect.y + 1, c_rect.width - 2, c_rect.height - 2}, 2.0f, theme::STATUS_ERR);
            }
        }
    }

    if (hit_test(ctx, grid_rect)) {
        const Vector2 m = GetMousePosition();
        const auto col = static_cast<size_t>((m.x - grid_rect.x) / cell_dim);
        const auto row = static_cast<size_t>((m.y - grid_rect.y) / cell_dim);
        const size_t target_idx = app.explorer_grid_offset + row * GRID_COLS + col;

        if (col < GRID_COLS && row < rows_to_display && target_idx < total_samples) {
            ctx.pointer_style = MOUSE_CURSOR_POINTING_HAND;
            DrawRectangleLinesEx(get_cell_box(target_idx), 1.5f, theme::TEXT_MUTED);
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                set_inspected_sample(target_idx);
                ctx.active_focus_id = -1;
            }
        }
    }

    if (app.active_sample_index >= app.explorer_grid_offset &&
        app.active_sample_index < app.explorer_grid_offset + samples_per_view) {
        DrawRectangleLinesEx(get_cell_box(app.active_sample_index), 2.5f, theme::ACCENT_PRIMARY);
    }

    const float sub_y = grid_rect.y + grid_rect.height + 8.0f;
    draw_ui_text(ctx, "Displaying " + std::to_string(app.explorer_grid_offset) + " - " +
                      std::to_string(std::min(app.explorer_grid_offset + samples_per_view, total_samples) - 1) +
                      " of " + std::to_string(total_samples) + (app.inspect_test_split ? " (test)" : " (train)"),
                 grid_rect.x, sub_y, 14, theme::TEXT_MUTED);

    if (render_error_outlines) {
        draw_ui_text(ctx, "Red border indicates batch misclassification", grid_rect.x, sub_y + 18, 12, theme::TEXT_FAINT);
    }

    if (app.sample_needs_render) {
        sync_single_sample_preview(app, *dataset);
    }

    const float sample_view_h = std::clamp(view_h * 0.5f, 224.0f, 310.0f);
    const Rectangle sample_panel_bounds{right_bounds.x, right_bounds.y, right_bounds.width, sample_view_h};
    const Rectangle eval_panel_bounds{right_bounds.x, right_bounds.y + sample_view_h + 12, right_bounds.width, std::max(120.0f, view_h - sample_view_h - 12)};

    render_card_frame(ctx, sample_panel_bounds, "Sample #" + std::to_string(app.active_sample_index));

    const float sx = sample_panel_bounds.x + 18.0f;
    const float preview_box_w = std::clamp(sample_panel_bounds.width * 0.34f, 100.0f, std::min(170.0f, sample_view_h - 112.0f));
    const float preview_box_y = sample_panel_bounds.y + 44.0f;
    const size_t truth_label = dataset->label(std::min(app.active_sample_index, total_samples - 1));

    if (app.sample_gpu_tex) {
        DrawTexturePro(app.sample_gpu_tex->handle(), Rectangle{0, 0, 28, 28},
                       Rectangle{sx, preview_box_y, preview_box_w, preview_box_w},
                       Vector2{0, 0}, 0.0f, WHITE);
        DrawRectangleLinesEx(Rectangle{sx - 1, preview_box_y - 1, preview_box_w + 2, preview_box_w + 2}, 1.0f, theme::BORDER_COLOR);
    }

    float meta_y = preview_box_y + preview_box_w + 8.0f;
    draw_ui_text(ctx, "Ground Truth: " + std::to_string(truth_label), sx, meta_y, 15, theme::TEXT_NORMAL);
    meta_y += 20.0f;

    const auto& s_pred = app.sample_prediction;
    if (s_pred) {
        const bool matched = (s_pred->digit == truth_label);
        draw_ui_text(ctx, "Predicted: " + std::to_string(s_pred->digit) + " (" +
                          format_fixed(100.0 * static_cast<double>(s_pred->probabilities[s_pred->digit]), 1) + "%)",
                     sx, meta_y, 15, matched ? theme::STATUS_GOOD : theme::STATUS_ERR);
    } else if (!app.active_model) {
        draw_ui_text(ctx, "Awaiting model load", sx, meta_y, 14, theme::STATUS_WARN);
    } else if (!app.sample_error_msg.empty()) {
        draw_ui_text(ctx, truncate_middle(ctx, app.sample_error_msg, 13, preview_box_w + 10), sx, meta_y, 13, theme::STATUS_ERR);
    }

    const float bar_origin_x = sx + preview_box_w + 18.0f;
    const float bar_total_w = sample_panel_bounds.x + sample_panel_bounds.width - 18.0f - bar_origin_x;
    const float bar_height = std::clamp((sample_view_h - 56.0f) / 10.0f - 4.0f, 10.0f, 18.0f);

    if (bar_total_w > 90.0f) {
        for (size_t i = 0; i < MNIST_CLASSES; ++i) {
            const float curr_y = sample_panel_bounds.y + 44.0f + static_cast<float>(i) * (bar_height + 4.0f);
            const bool is_pred = s_pred && s_pred->digit == i;
            const bool is_true = (i == truth_label);
            const double prob = s_pred ? static_cast<double>(s_pred->probabilities[i]) : 0.0;

            draw_ui_text(ctx, std::to_string(i), bar_origin_x, curr_y + (bar_height - 13.0f) / 2.0f, 13,
                         is_true ? theme::STATUS_GOOD : (is_pred ? theme::ACCENT_PRIMARY : theme::TEXT_MUTED));

            const Rectangle meter_box{bar_origin_x + 16, curr_y, bar_total_w - 16 - 52, bar_height};
            render_meter(ctx, meter_box, prob, is_pred ? (is_true ? theme::STATUS_GOOD : theme::STATUS_ERR)
                                                       : blend_colors(theme::ACCENT_SUBDUED, theme::TEXT_FAINT, 0.5f));

            if (is_true) {
                DrawRectangleLinesEx(Rectangle{meter_box.x - 1, meter_box.y - 1, meter_box.width + 2, meter_box.height + 2}, 1.5f, theme::STATUS_GOOD);
            }
            draw_ui_text(ctx, s_pred ? format_fixed(100.0 * prob, 1) + "%" : "-",
                         meter_box.x + meter_box.width + 8, curr_y + (bar_height - 12.0f) / 2.0f, 12,
                         is_pred ? theme::TEXT_NORMAL : theme::TEXT_MUTED);
        }
    }

    render_card_frame(ctx, eval_panel_bounds, std::string("Partition Assessment (") + (app.inspect_test_split ? "Test)" : "Train)"));
    const float ex = eval_panel_bounds.x + 18.0f;
    const float ew = eval_panel_bounds.width - 36.0f;
    const float matrix_side = std::clamp(std::min(eval_panel_bounds.height - 100.0f, ew * 0.4f), 0.0f, 190.0f);
    const bool show_matrix = (matrix_side >= 100.0f && app.eval_summary_valid);
    const float col_left_w = show_matrix ? (ew - matrix_side - 16.0f) : ew;
    const bool is_evaluating = app.evaluator.is_active();

    float ey = eval_panel_bounds.y + 46.0f;
    if (ui_button(ctx, Rectangle{ex, ey, col_left_w * 0.58f - 4, 34}, "Run Batch",
                  app.active_model.has_value() && !is_evaluating, ActionStyle::Accent)) {
        trigger_background_evaluation(app);
    }
    if (ui_button(ctx, Rectangle{ex + col_left_w * 0.58f + 4, ey, col_left_w * 0.42f - 4, 34}, "Abort", is_evaluating, ActionStyle::Danger)) {
        app.evaluator.cancel();
    }
    ey += 44.0f;

    if (is_evaluating) {
        const double frac = app.evaluator.total_samples() > 0
            ? static_cast<double>(app.evaluator.completed()) / static_cast<double>(app.evaluator.total_samples())
            : 0.0;
        render_meter(ctx, Rectangle{ex, ey, col_left_w, 16}, frac, theme::ACCENT_PRIMARY,
                     std::to_string(app.evaluator.completed()) + " / " + std::to_string(app.evaluator.total_samples()));
        ey += 26.0f;
        draw_ui_text(ctx, "Current accuracy: " + format_fixed(app.evaluator.running_accuracy(), 2) + "%", ex, ey, 14, theme::TEXT_MUTED);
    } else if (!app.active_model) {
        draw_multiline_text(ctx, "Load weights from Checkpoints tab to evaluate.", ex, ey, col_left_w, 14, theme::STATUS_WARN);
    } else if (app.eval_summary_valid) {
        render_meter(ctx, Rectangle{ex, ey, col_left_w, 16}, 1.0, blend_colors(theme::ACCENT_SUBDUED, theme::STATUS_GOOD, 0.6f),
                     app.eval_summary.was_cancelled ? "Incomplete" : "Validated");
        ey += 26.0f;
        const std::string split_label = app.eval_summary.is_test_set ? "Test" : "Train";
        draw_ui_text(ctx, (app.eval_summary.was_cancelled ? "Partial " : "") + split_label + " Accuracy: " + format_fixed(app.eval_summary.accuracy_percent(), 2) + "%",
                     ex, ey, 15, app.eval_summary.was_cancelled ? theme::STATUS_WARN : theme::STATUS_GOOD);
        ey += 21.0f;

        draw_ui_text(ctx, truncate_middle(ctx, std::to_string(app.eval_summary.correct) + " / " + std::to_string(app.eval_summary.total) +
                                               " (Loss: " + format_fixed(app.eval_summary.average_loss(), 4) + ")", 13, col_left_w),
                     ex, ey, 13, theme::TEXT_MUTED);
        ey += 18.0f;

        size_t weakest_digit = 0;
        double weakest_val = 0.0;
        locate_worst_performing_digit(app.eval_summary, weakest_digit, weakest_val);
        if (ey + 14 < eval_panel_bounds.y + eval_panel_bounds.height - 52) {
            draw_ui_text(ctx, "Lowest Class: " + std::to_string(weakest_digit) + " (" + format_fixed(weakest_val, 1) + "%)", ex, ey, 13, theme::TEXT_MUTED);
        }
    } else {
        draw_multiline_text(ctx, "Runs asynchronously on a decoupled model clone.", ex, ey, col_left_w, 13, theme::TEXT_MUTED);
    }

    if (show_matrix) {
        render_confusion_grid(ctx, Rectangle{ex + col_left_w + 16, eval_panel_bounds.y + 46, matrix_side, matrix_side}, app.eval_summary);
    }

    const float nav_y = eval_panel_bounds.y + eval_panel_bounds.height - 44.0f;
    const bool can_navigate_errors = render_error_outlines && !app.eval_summary.misclassified_indices.empty();

    if (ui_button(ctx, Rectangle{ex, nav_y, ew / 2 - 4, 32}, "Prev Error", can_navigate_errors)) {
        const auto& miss_list = app.eval_summary.misclassified_indices;
        const auto it = std::lower_bound(miss_list.begin(), miss_list.end(), app.active_sample_index);
        set_inspected_sample(it != miss_list.begin() ? *(it - 1) : miss_list.back());
    }
    if (ui_button(ctx, Rectangle{ex + ew / 2 + 4, nav_y, ew / 2 - 4, 32}, "Next Error", can_navigate_errors)) {
        const auto& miss_list = app.eval_summary.misclassified_indices;
        const auto it = std::upper_bound(miss_list.begin(), miss_list.end(), app.active_sample_index);
        set_inspected_sample(it != miss_list.end() ? *it : miss_list.front());
    }

    const auto& footer_notice = !app.eval_failure_text.empty()
        ? StatusNotice{"Evaluation terminated: " + app.eval_failure_text, NoticeLevel::Error}
        : app.explorer_notice;
    draw_ui_text(ctx, truncate_middle(ctx, footer_notice.message, 15, layout.width), layout.x, layout.y + layout.height - 24, 15, resolve_notice_color(footer_notice.level));
}

void verify_dataset_presence(ApplicationState& app) {
    const double now = GetTime();
    if (app.verified_data_path == app.dataset_path_input && now - app.dataset_probe_time < 1.5) return;
    const DatasetBundle bundle = resolve_dataset_files(fs::path(trim_whitespace(app.dataset_path_input)));
    app.train_dataset_found = bundle.has_train();
    app.test_dataset_found = bundle.has_test();
    app.verified_data_path = app.dataset_path_input;
    app.dataset_probe_time = now;
}

void render_trainer_tab(ApplicationState& app, Rectangle layout) {
    RenderContext& ctx = app.ctx;
    verify_dataset_presence(app);
    const bool busy = app.trainer.is_busy();

    const float form_w = std::clamp(layout.width * 0.5f, 380.0f, 560.0f);
    const Rectangle form_view_area{layout.x, layout.y + 84, form_w, layout.y + layout.height - (layout.y + 84)};
    const float right_x = layout.x + form_w + 24.0f;
    const float right_w = layout.x + layout.width - right_x;

    if (hit_test(ctx, form_view_area)) {
        app.trainer_scroll_y -= GetMouseWheelMove() * 50.0f;
    }

    const Rectangle prior_clip = ctx.scissor_bounds;
    ctx.scissor_bounds = form_view_area;

    const bool enable_edits = !busy;
    float fy = form_view_area.y - app.trainer_scroll_y;
    const float fx = form_view_area.x;
    const float fw = form_view_area.width - 14.0f;

    draw_ui_text(ctx, "CONFIG TYPE", fx, fy, 14, theme::TEXT_MUTED);
    fy += 20.0f;

    if (ui_button(ctx, Rectangle{fx, fy, fw / 2 - 4, 36}, "Scratch Model", enable_edits,
                  !app.resume_existing ? ActionStyle::Highlighted : ActionStyle::Default)) {
        app.resume_existing = false;
    }
    if (ui_button(ctx, Rectangle{fx + fw / 2 + 4, fy, fw / 2 - 4, 36}, "Resume Weights", enable_edits,
                  app.resume_existing ? ActionStyle::Highlighted : ActionStyle::Default)) {
        app.resume_existing = true;
        if (app.resume_input.empty() && app.active_model) {
            app.resume_input = app.loaded_checkpoint_path.string();
        }
    }
    fy += 52.0f;

    if (app.resume_existing) {
        fy = render_input_row(ctx, ID_RESUME_PATH, fx, fy, fw, "CHECKPOINT SOURCE", app.resume_input,
                              "Path to serialized weights (.txt)", enable_edits);
        fy += draw_multiline_text(ctx, "Tensors will be mapped directly; no random initialization.",
                                  fx, fy - 6, fw, 13, theme::TEXT_MUTED) + 8.0f;
    } else {
        fy = render_input_row(ctx, ID_SPEC_ARCH, fx, fy, fw, "LAYER TOPOLOGY (784 ... 10)", app.architecture_input,
                              "784 512 256 128 64 10", enable_edits);
        draw_ui_text(ctx, "HIDDEN LAYER INITIALIZER", fx, fy - 4, 14, theme::TEXT_MUTED);
        fy += 18.0f;

        if (ui_button(ctx, Rectangle{fx, fy, fw / 2 - 4, 34}, "He (Normal)", enable_edits,
                      app.selected_init == neural::WeightInit::He ? ActionStyle::Highlighted : ActionStyle::Default)) {
            app.selected_init = neural::WeightInit::He;
        }
        if (ui_button(ctx, Rectangle{fx + fw / 2 + 4, fy, fw / 2 - 4, 34}, "Xavier (Glorot)", enable_edits,
                      app.selected_init == neural::WeightInit::Xavier ? ActionStyle::Highlighted : ActionStyle::Default)) {
            app.selected_init = neural::WeightInit::Xavier;
        }
        fy += 42.0f;
        fy += draw_multiline_text(ctx, "Output dense projection defaults to Xavier scaling.",
                                  fx, fy, fw, 13, theme::TEXT_MUTED) + 12.0f;
    }

    const float field_half_w = fw / 2.0f - 6.0f;
    const float dual_row_y = fy;
    render_input_row(ctx, ID_TRAIN_EPOCHS, fx, dual_row_y, field_half_w, "EPOCHS", app.epoch_input, "5", enable_edits);
    fy = render_input_row(ctx, ID_TRAIN_LR, fx + field_half_w + 12, dual_row_y, field_half_w, "LEARNING RATE", app.lr_input, "0.01", enable_edits);
    fy = render_input_row(ctx, ID_MAX_SAMPLES, fx, fy, field_half_w, "MAX SAMPLES / EPOCH (0 = ALL)", app.max_samples_input, "0", enable_edits);

    draw_ui_text(ctx, "Stochastic Gradient Descent (Step Size = 1)", fx, fy - 8, 14, theme::TEXT_MUTED);
    fy += 24.0f;

    fy = render_input_row(ctx, ID_DATASET_DIR, fx, fy, fw, "MNIST DATASET DIRECTORY", app.dataset_path_input,
                          "Path to IDX file bundle", enable_edits);

    draw_ui_text(ctx, std::string("Training files: ") + (app.train_dataset_found ? "found" : "MISSING"),
                 fx, fy - 8, 14, app.train_dataset_found ? theme::STATUS_GOOD : theme::STATUS_ERR);
    draw_ui_text(ctx, std::string("Test split: ") + (app.test_dataset_found ? "found" : "absent (optional)"),
                 fx + 200, fy - 8, 14, app.test_dataset_found ? theme::STATUS_GOOD : theme::TEXT_MUTED);
    fy += 22.0f;

    if (!app.test_dataset_found) app.run_eval_after = false;
    ui_checkbox(ctx, Rectangle{fx, fy, fw, 28}, "Evaluate test accuracy upon run completion", app.run_eval_after,
                enable_edits && app.test_dataset_found);
    fy += 32.0f;

    ui_checkbox(ctx, Rectangle{fx, fy, fw, 28}, "Preserve weights to disk on manual stop", app.save_on_abort, enable_edits);
    fy += 40.0f;

    fy = render_input_row(ctx, ID_EXPORT_PATH, fx, fy, fw, "EXPORT PATH", app.save_target_input,
                          "Target checkpoint file (.txt)", enable_edits);

    const float total_rendered_h = (fy - (form_view_area.y - app.trainer_scroll_y)) + 20.0f;
    ctx.scissor_bounds = prior_clip;

    const float max_scroll = std::max(0.0f, total_rendered_h - form_view_area.height);
    app.trainer_scroll_y = std::clamp(app.trainer_scroll_y, 0.0f, max_scroll);

    DrawRectangleRec(Rectangle{layout.x - 4, 0, form_w + 12, form_view_area.y - 4}, theme::BG_DARK);
    draw_page_heading(ctx, layout, "Trainer Subsystem", "Configure hyperparameters and train the network asynchronously on a worker thread.");

    const auto snap = app.trainer.get_snapshot(true);
    const Rectangle right_panel{right_x, layout.y + 84, right_w, layout.y + layout.height - (layout.y + 84)};
    render_card_frame(ctx, right_panel, "Runtime Metrics");

    const float rx = right_panel.x + 18.0f;
    const float rw = right_panel.width - 36.0f;
    float ry = right_panel.y + 50.0f;

    const bool can_dispatch = !busy && app.train_dataset_found;
    if (ui_button(ctx, Rectangle{rx, ry, rw / 2 - 4, 40}, "Begin Training", can_dispatch, ActionStyle::Accent)) {
        dispatch_training(app);
    }
    if (ui_button(ctx, Rectangle{rx + rw / 2 + 4, ry, rw / 2 - 4, 40}, "Abort", busy, ActionStyle::Danger)) {
        app.trainer.stop();
    }
    ry += 56.0f;

    Color p_color = theme::TEXT_MUTED;
    if (busy) p_color = theme::ACCENT_PRIMARY;
    if (snap.phase == StepPhase::Complete) p_color = theme::STATUS_GOOD;
    if (snap.phase == StepPhase::Error)    p_color = theme::STATUS_ERR;
    if (snap.phase == StepPhase::Aborted)  p_color = theme::STATUS_WARN;

    draw_ui_text(ctx, std::string("Engine: ") + stringify_phase(snap.phase), rx, ry, 18, p_color);
    ry += 26.0f;

    if (!snap.status_msg.empty()) ry += draw_multiline_text(ctx, snap.status_msg, rx, ry, rw, 14, theme::TEXT_MUTED) + 4.0f;
    if (!snap.error_msg.empty())  ry += draw_multiline_text(ctx, snap.error_msg, rx, ry, rw, 14, theme::STATUS_ERR) + 4.0f;

    if (!app.trainer_notice.message.empty() && snap.phase == StepPhase::Idle) {
        ry += draw_multiline_text(ctx, app.trainer_notice.message, rx, ry, rw, 14, resolve_notice_color(app.trainer_notice.level)) + 4.0f;
    } else if (app.trainer_notice.level == NoticeLevel::Error && !busy) {
        ry += draw_multiline_text(ctx, app.trainer_notice.message, rx, ry, rw, 14, theme::STATUS_ERR) + 4.0f;
    }
    ry += 6.0f;

    const double ep_fraction = snap.total_epochs > 0
        ? (static_cast<double>(snap.epoch > 0 ? snap.epoch - 1 : 0) +
          (snap.epoch_size > 0 ? static_cast<double>(snap.sample_in_epoch) / static_cast<double>(snap.epoch_size) : 0.0)) /
          static_cast<double>(snap.total_epochs)
        : 0.0;

    const double net_progress = (snap.phase == StepPhase::Complete) ? 1.0 : std::min(1.0, ep_fraction);
    draw_ui_text(ctx, "Epoch: " + std::to_string(snap.epoch) + " / " + std::to_string(snap.total_epochs), rx, ry, 15, theme::TEXT_NORMAL);
    ry += 22.0f;

    render_meter(ctx, Rectangle{rx, ry, rw, 20}, net_progress, theme::ACCENT_PRIMARY, format_fixed(100.0 * net_progress, 1) + "%");
    ry += 30.0f;

    draw_ui_text(ctx, "Current Epoch Progress: " + std::to_string(snap.sample_in_epoch) + " / " + std::to_string(snap.epoch_size), rx, ry, 14, theme::TEXT_MUTED);
    ry += 20.0f;

    render_meter(ctx, Rectangle{rx, ry, rw, 10}, snap.epoch_size > 0 ? static_cast<double>(snap.sample_in_epoch) / static_cast<double>(snap.epoch_size) : 0.0,
                 blend_colors(theme::ACCENT_SUBDUED, theme::ACCENT_PRIMARY, 0.6f));
    ry += 24.0f;

    const double throughput = snap.elapsed_sec > 0.5 ? static_cast<double>(snap.total_processed) / snap.elapsed_sec : 0.0;
    draw_ui_text(ctx, "Processed Samples: " + std::to_string(snap.total_processed), rx, ry, 14, theme::TEXT_NORMAL);
    ry += 20.0f;

    draw_ui_text(ctx, "Mean Batch Loss: " + (snap.total_processed > 0 ? format_fixed(snap.current_loss, 4) : std::string("--")), rx, ry, 14, theme::TEXT_NORMAL);
    ui_checkbox(ctx, Rectangle{rx + rw - 100, ry - 3, 100, 22}, "Log Scale", app.use_logarithmic_loss);
    ry += 20.0f;

    std::string perf_line = "Elapsed: " + format_duration(snap.elapsed_sec);
    if (snap.phase == StepPhase::Running && throughput > 0.0 && snap.epoch_size > 0) {
        const double remaining_samples = static_cast<double>(snap.total_epochs * snap.epoch_size) - static_cast<double>(snap.total_processed);
        perf_line += " | " + format_fixed(throughput, 0) + " samples/sec | Remaining: " + format_duration(remaining_samples / throughput);
    }
    draw_ui_text(ctx, truncate_middle(ctx, perf_line, 14, rw), rx, ry, 14, theme::TEXT_MUTED);
    ry += 26.0f;

    const double full_domain = static_cast<double>(snap.total_epochs) * static_cast<double>(snap.epoch_size);
    const float chart_height = std::clamp(right_panel.y + right_panel.height - ry - 150.0f, 90.0f, 200.0f);
    render_loss_plot(ctx, Rectangle{rx, ry, rw, chart_height}, snap.loss_curve, full_domain, snap.epoch_size, snap.total_epochs, app.use_logarithmic_loss);
    ry += chart_height + 14.0f;

    if (snap.phase == StepPhase::Validating) {
        render_meter(ctx, Rectangle{rx, ry, rw, 14}, snap.eval_total > 0 ? static_cast<double>(snap.eval_done) / static_cast<double>(snap.eval_total) : 0.0,
                     theme::STATUS_WARN, "Validating Test Partition");
        ry += 26.0f;
    }
    if (snap.test_accuracy >= 0.0) {
        draw_ui_text(ctx, "Validation Accuracy: " + format_fixed(snap.test_accuracy, 2) + "%", rx, ry, 18, theme::STATUS_GOOD);
        ry += 24.0f;
        draw_ui_text(ctx, "Validation Loss: " + format_fixed(snap.test_loss, 4), rx, ry, 14, theme::TEXT_MUTED);
        ry += 22.0f;
    }
    if (!snap.saved_file.empty()) {
        draw_multiline_text(ctx, "Saved Checkpoint: " + snap.saved_file, rx, ry, rw, 13, theme::STATUS_GOOD);
    }
}

void render_confirmation_modal(ApplicationState& app, float w, float h) {
    RenderContext& ctx = app.ctx;
    DrawRectangleRec(Rectangle{0, 0, w, h}, Fade(BLACK, 0.6f));

    const float box_w = std::min(520.0f, w - 60.0f);
    const Rectangle dialog_box{(w - box_w) / 2.0f, h / 2.0f - 120.0f, box_w, 230.0f};

    render_outlined_box(dialog_box, 12, theme::BORDER_COLOR, theme::PANEL_BG);
    draw_ui_text(ctx, "Overwrite Existing File?", dialog_box.x + 22, dialog_box.y + 18, 20, theme::TEXT_NORMAL);

    std::string single_line_msg = app.confirmation_dialog_text;
    for (char& c : single_line_msg) {
        if (c == '\n') c = ' ';
    }
    draw_multiline_text(ctx, single_line_msg, dialog_box.x + 22, dialog_box.y + 56, dialog_box.width - 44, 15, theme::TEXT_MUTED);

    const bool overwrite = ui_button(ctx, Rectangle{dialog_box.x + 22, dialog_box.y + dialog_box.height - 60, 150, 40}, "Overwrite", true, ActionStyle::Danger);
    const bool cancel = ui_button(ctx, Rectangle{dialog_box.x + 184, dialog_box.y + dialog_box.height - 60, 120, 40}, "Cancel") || IsKeyPressed(KEY_ESCAPE);

    if (overwrite && app.queued_training_cfg) {
        app.trainer_notice = {"Training initiated.", NoticeLevel::Success};
        app.trainer.start(std::move(*app.queued_training_cfg));
        app.queued_training_cfg.reset();
        app.confirm_overwrite_open = false;
    } else if (cancel) {
        app.queued_training_cfg.reset();
        app.confirm_overwrite_open = false;
    }
}

void render_application_frame(ApplicationState& app) {
    RenderContext& ctx = app.ctx;
    ctx.pointer_style = MOUSE_CURSOR_DEFAULT;
    ctx.scissor_bounds = Rectangle{0, 0, 100000, 100000};
    ctx.input_blocked = app.confirm_overwrite_open;

    const float view_w = static_cast<float>(GetScreenWidth());
    const float view_h = static_cast<float>(GetScreenHeight());

    BeginDrawing();
    ClearBackground(theme::BG_DARK);
    render_app_sidebar(app, view_h);

    const float padding = 28.0f;
    const Rectangle main_layout{210.0f + padding, padding, view_w - 210.0f - 2.0f * padding, view_h - 2.0f * padding};

    switch (app.current_tab) {
        case ActiveTab::Checkpoints:   render_models_tab(app, main_layout); break;
        case ActiveTab::DrawingCanvas: render_drawing_tab(app, main_layout); break;
        case ActiveTab::DataExplorer:  render_data_tab(app, main_layout); break;
        case ActiveTab::Trainer:       render_trainer_tab(app, main_layout); break;
    }

    if (app.confirm_overwrite_open) {
        ctx.input_blocked = false;
        render_confirmation_modal(app, view_w, view_h);
    }
    EndDrawing();

    SetMouseCursor(ctx.pointer_style);
    if (ctx.active_focus_id == -1) {
        while (GetCharPressed() != 0) {}
    }

    const bool currently_training = app.trainer.is_busy();
    if (app.was_training_active && !currently_training) {
        const auto snap = app.trainer.get_snapshot(false);
        if (!snap.saved_file.empty()) {
            refresh_checkpoints(app);
            app.selected_checkpoint = fs::path(snap.saved_file);
        }
    }
    app.was_training_active = currently_training;

    EvaluationResult eval_res;
    std::string eval_err;
    if (app.evaluator.fetch_result(eval_res, eval_err)) {
        if (!eval_err.empty()) {
            app.eval_failure_text = eval_err;
            app.eval_summary_valid = false;
        } else if (app.active_model && eval_res.model_identifier == app.loaded_checkpoint_path.string()) {
            app.eval_summary = std::move(eval_res);
            app.eval_summary_valid = true;
        } else {
            app.eval_summary_valid = false;
        }
    }
}

// --- Headless Test Driver ---

void synthesize_idx_fixtures(const fs::path& img_target, const fs::path& lbl_target, uint32_t num_samples) {
    auto write_be32 = [](std::ofstream& stream, uint32_t val) {
        const std::array<char, 4> bytes = {
            static_cast<char>((val >> 24U) & 0xFFU),
            static_cast<char>((val >> 16U) & 0xFFU),
            static_cast<char>((val >> 8U) & 0xFFU),
            static_cast<char>(val & 0xFFU)
        };
        stream.write(bytes.data(), 4);
    };

    std::ofstream img_file(img_target, std::ios::binary);
    std::ofstream lbl_file(lbl_target, std::ios::binary);

    write_be32(img_file, 2051U);
    write_be32(img_file, num_samples);
    write_be32(img_file, 28U);
    write_be32(img_file, 28U);

    write_be32(lbl_file, 2049U);
    write_be32(lbl_file, num_samples);

    for (uint32_t i = 0; i < num_samples; ++i) {
        for (uint32_t p = 0; p < 784U; ++p) {
            img_file.put(((p + i * 13U) % 9U == 0U) ? static_cast<char>(255) : static_cast<char>(0));
        }
        lbl_file.put(static_cast<char>(i % 10U));
    }
}

int execute_self_tests() {
    int errors = 0;
    auto evaluate_condition = [&](bool pass, const std::string& description) {
        std::cout << (pass ? "[PASS] " : "[FAIL] ") << description << '\n';
        if (!pass) ++errors;
    };

    auto expect_exception = [](const std::function<void()>& fn) {
        try { fn(); } catch (const std::exception&) { return true; }
        return false;
    };

    const fs::path sandbox = fs::temp_directory_path() /
        ("neural_studio_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(sandbox);

    try {
        // 1. Image preprocessing checks
        {
            std::vector<uint8_t> empty_canvas(CANVAS_SIZE * CANVAS_SIZE, 0);
            evaluate_condition(preprocess_digit_image(empty_canvas, CANVAS_SIZE).empty, "Empty canvas produces empty digit");

            paint_brush_stroke(empty_canvas, 100, 40, 160, 240);
            const DigitSample sample = preprocess_digit_image(empty_canvas, CANVAS_SIZE);
            evaluate_condition(!sample.empty && sample.pixels.size() == 784, "Rendered digit has 784 pixels");

            const auto max_p = *std::max_element(sample.pixels.begin(), sample.pixels.end());
            const auto min_p = *std::min_element(sample.pixels.begin(), sample.pixels.end());
            evaluate_condition(min_p >= 0.0f && max_p <= 1.0f && max_p > 0.5f, "Pixel normalization within [0, 1]");

            double mass = 0.0, cx = 0.0, cy = 0.0;
            for (int y = 0; y < 28; ++y) {
                for (int x = 0; x < 28; ++x) {
                    const double v = sample.pixels[static_cast<size_t>(y * 28 + x)];
                    mass += v;
                    cx += v * x;
                    cy += v * y;
                }
            }
            evaluate_condition(std::abs(cx / mass - 13.5) < 1.5 && std::abs(cy / mass - 13.5) < 1.5, "Mass centered near middle pixel");
            evaluate_condition(tensor_from_pixels(sample.pixels).size() == 784, "Tensor matches shape 784");
        }

        // 2. Model serialization verification
        const std::vector<size_t> test_arch = {784, 24, 12, 10};
        Network net_a(test_arch, neural::WeightInit::He, BASE_SEED);
        const fs::path ckpt_path = sandbox / "models" / "valid_model.txt";
        net_a.save(ckpt_path);

        evaluate_condition(safe_is_file(ckpt_path) && !safe_is_file(fs::path(ckpt_path.string() + ".tmp")),
                           "Atomic checkpoint output creates file and cleans tmp");

        Network net_b = load_checkpoint(ckpt_path);
        bool match = (net_a.layer_count() == net_b.layer_count());
        for (size_t i = 0; match && i < net_a.layer_count(); ++i) {
            const auto wa = net_a.layer(i).weights().data();
            const auto wb = net_b.layer(i).weights().data();
            const auto ba = net_a.layer(i).bias().data();
            const auto bb = net_b.layer(i).bias().data();
            match = std::equal(wa.begin(), wa.end(), wb.begin(), wb.end()) &&
                    std::equal(ba.begin(), ba.end(), bb.begin(), bb.end());
        }
        evaluate_condition(match, "Serialized and deserialized weights/biases match exactly");

        {
            DigitSample sample;
            for (size_t i = 0; i < 784; ++i) sample.pixels[i] = static_cast<float>(i % 7) / 7.0f;
            const ClassPrediction pa = predict_class(net_a, tensor_from_pixels(sample.pixels));
            const ClassPrediction pb = predict_class(net_b, tensor_from_pixels(sample.pixels));
            evaluate_condition(pa.digit == pb.digit && pa.probabilities == pb.probabilities, "Model outputs identical inferences post-load");

            const float sum_prob = std::accumulate(pa.probabilities.begin(), pa.probabilities.end(), 0.0f);
            evaluate_condition(std::abs(sum_prob - 1.0f) < 1e-4f, "Probabilities sum to 1.0");

            Network net_c = net_a.clone();
            const ClassPrediction pc = predict_class(net_c, tensor_from_pixels(sample.pixels));
            evaluate_condition(pa.probabilities == pc.probabilities, "Network clone() produces identical activations");
        }

        // 3. Validation of corrupted/malformed inputs
        std::ifstream file_in(ckpt_path);
        const std::string valid_text((std::istreambuf_iterator<char>(file_in)), std::istreambuf_iterator<char>());

        auto write_corrupt_file = [&](const char* name, const std::string& content) {
            const fs::path p = sandbox / "corrupt" / name;
            fs::create_directories(p.parent_path());
            std::ofstream(p) << content;
            return p;
        };

        evaluate_condition(expect_exception([&] { (void)load_checkpoint(write_corrupt_file("plain.txt", "arbitrary text")); }), "Reject non-checkpoint plain text");

        std::string v2_text = valid_text;
        v2_text.replace(v2_text.find(" 1\n"), 3, " 2\n");
        evaluate_condition(expect_exception([&] { (void)load_checkpoint(write_corrupt_file("v2.txt", v2_text)); }), "Reject unhandled checkpoint version");

        evaluate_condition(expect_exception([&] { (void)load_checkpoint(write_corrupt_file("partial.txt", valid_text.substr(0, valid_text.size() / 2))); }), "Reject truncated checkpoint");

        std::string dim_mismatch = valid_text;
        dim_mismatch.replace(dim_mismatch.find("784 24 12 10"), 12, "784 25 12 10");
        evaluate_condition(expect_exception([&] { (void)load_checkpoint(write_corrupt_file("mismatch.txt", dim_mismatch)); }), "Reject shape mismatch with declared architecture");

        // Renamed variable to avoid macro/function conflict with <cmath> nan
        std::string bad_nan_text = valid_text;
        const size_t insert_idx = bad_nan_text.find("2 24 784\n") + 9;
        bad_nan_text.insert(insert_idx, "nan ");
        evaluate_condition(expect_exception([&] { (void)load_checkpoint(write_corrupt_file("nan.txt", bad_nan_text)); }), "Reject NaN weights during deserialization");

        // 4. Directory scanner validation
        net_a.save(sandbox / "build" / "ignored_by_dir.txt");
        const DirectoryScan scan = scan_checkpoint_directory(sandbox);
        size_t valid_cnt = 0, invalid_cnt = 0;
        for (const auto& item : scan.entries) {
            (item.valid ? valid_cnt : invalid_cnt)++;
        }
        evaluate_condition(valid_cnt == 4 && invalid_cnt == 1, "Directory scanner identifies valid vs invalid files");
        evaluate_condition(scan.non_checkpoints == 1, "Non-checkpoint text files ignored");
        evaluate_condition(std::none_of(scan.entries.begin(), scan.entries.end(), [](const ModelEntry& e) {
            return e.rel_path.rfind("build/", 0) == 0;
        }), "Build subdirectories ignored by scan");

        // 5. Optimization step sanity
        {
            Network net({784, 32, 10}, neural::WeightInit::He, BASE_SEED);
            neural::Tensor in_tensor({MNIST_INPUTS}, 0.0f);
            auto data = in_tensor.data();
            for (size_t i = 0; i < 784; ++i) data[i] = (i % 5 == 0) ? 1.0f : 0.0f;

            const neural::SGD opt(0.05f);
            const float initial_loss = net.train_step(in_tensor, 3, opt);
            float final_loss = initial_loss;
            for (int i = 0; i < 30; ++i) final_loss = net.train_step(in_tensor, 3, opt);
            evaluate_condition(final_loss < initial_loss, "SGD steps consistently reduce loss on single sample");
        }

        // 6. Dataset and evaluator unit validation
        {
            const fs::path img_fixture = sandbox / "synthetic-images-idx3-ubyte";
            const fs::path lbl_fixture = sandbox / "synthetic-labels-idx1-ubyte";
            synthesize_idx_fixtures(img_fixture, lbl_fixture, 30);

            const auto ds = std::make_shared<neural::MNISTDataset>(img_fixture, lbl_fixture);
            evaluate_condition(ds->size() == 30 && ds->label(13) == 3, "Synthetic dataset loads binary IDX payload correctly");

            Network eval_net({784, 16, 10}, neural::WeightInit::He, BASE_SEED);
            std::stop_source idle_stop;
            const EvaluationResult res = evaluate_network(eval_net, *ds, idle_stop.get_token(), [](size_t, size_t) {});

            size_t matrix_sum = 0;
            for (const auto& row : res.confusion_matrix) {
                for (size_t val : row) matrix_sum += val;
            }
            evaluate_condition(res.total == 30 && matrix_sum == 30, "Confusion matrix tallies equal sample count");
            evaluate_condition(res.correct + res.miss_count == 30 && res.misclassified_indices.size() == res.miss_count,
                               "Correct + error counts match total");

            std::stop_source cancel_stop;
            cancel_stop.request_stop();
            const EvaluationResult cancelled_res = evaluate_network(eval_net, *ds, cancel_stop.get_token(), [](size_t, size_t) {});
            evaluate_condition(cancelled_res.was_cancelled && cancelled_res.total == 0, "Evaluator acknowledges stop_token");

            AsyncEvaluator async_worker;
            async_worker.start(eval_net.clone(), ds, true, "synthetic");
            for (int i = 0; i < 500 && async_worker.is_active(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            EvaluationResult async_res;
            std::string async_err;
            const bool got_result = async_worker.fetch_result(async_res, async_err);
            evaluate_condition(got_result && async_err.empty() && async_res.total == 30 && async_res.correct == res.correct,
                               "Background AsyncEvaluator matches synchronous evaluation");
        }
    } catch (const std::exception& e) {
        std::cout << "[FAIL] Unexpected test exception: " << e.what() << '\n';
        ++errors;
    }

    std::error_code ec;
    fs::remove_all(sandbox, ec);
    std::cout << (errors == 0 ? "All verification suites passed.\n" : "Validation failures encountered.\n");
    return errors == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--self-test") {
        return execute_self_tests();
    }

    try {
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "NeuralEngine Studio");
        SetWindowMinSize(1000, 680);
        SetExitKey(KEY_NULL);
        SetTargetFPS(60);

        Font active_font = GetFontDefault();
        bool has_loaded_font = false;

        const char* font_candidates[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        };

        for (const char* path : font_candidates) {
            if (FileExists(path)) {
                const Font loaded = LoadFontEx(path, 40, nullptr, 0);
                if (loaded.texture.id != 0) {
                    SetTextureFilter(loaded.texture, TEXTURE_FILTER_BILINEAR);
                    active_font = loaded;
                    has_loaded_font = true;
                    break;
                }
            }
        }

        {
            ApplicationState app;
            app.ctx.font = active_font;
            app.ctx.using_custom_font = has_loaded_font;
            app.workspace_root = detect_project_root();
            app.root_input = app.workspace_root.string();
            app.dataset_path_input = find_default_dataset_dir(app.workspace_root).string();
            app.save_target_input = (app.workspace_root / "models" / "mnist_studio.txt").string();
            app.canvas_gpu_tex = std::make_unique<MonochromeTexture>(CANVAS_SIZE, CANVAS_SIZE, app.draw_buffer, true);

            refresh_checkpoints(app);

            while (!WindowShouldClose()) {
                render_application_frame(app);
            }

            app.trainer.stop();
            app.evaluator.cancel();
        }

        if (has_loaded_font) {
            UnloadFont(active_font);
        }
        CloseWindow();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal engine termination: " << e.what() << '\n';
        return 1;
    }
}
