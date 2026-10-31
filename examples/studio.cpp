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

// Clear platform macro conflicts
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
#include <ctime>
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
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ============================================================================
// Constants
// ============================================================================

constexpr const char* kCheckpointMagic = "NEURAL_ENGINE_CHECKPOINT";
constexpr long kCheckpointVersion = 1;
constexpr std::size_t kMaxLayerWidth = 65536;
constexpr std::size_t kMaxLayerCount = 32;
constexpr std::size_t kMaxParameters = 60'000'000;
constexpr std::size_t kMnistInputs = 28U * 28U;
constexpr std::size_t kMnistClasses = 10;
constexpr std::uint32_t kBaseSeed = 100;
constexpr std::uint32_t kShuffleSeed = 42;
constexpr std::size_t kHistoryCap = 4000;
constexpr std::size_t kMaxMisses = 20000;
constexpr std::size_t kGridCols = 8;
constexpr std::size_t kOptimiserCount = 4;
constexpr std::size_t kMaxBatchSize = 4096;

constexpr int kCanvas = 280;
constexpr float kBrushRadius = 11.0F;

constexpr const char* kTrainImages = "train-images-idx3-ubyte";
constexpr const char* kTrainLabels = "train-labels-idx1-ubyte";
constexpr const char* kTestImages = "t10k-images-idx3-ubyte";
constexpr const char* kTestLabels = "t10k-labels-idx1-ubyte";

// Sensible batch-size-1 defaults, indexed by OptimiserType (SGD, Momentum, RMSprop, Adam).
constexpr std::array<float, kOptimiserCount> kDefaultLrs = {0.01F, 0.002F, 0.0001F, 0.0005F};
constexpr std::array<const char*, kOptimiserCount> kDefaultLrTexts = {"0.01", "0.002", "0.0001", "0.0005"};
constexpr float kMinLr = 1e-6F;
constexpr float kMaxLr = 1.0F;

// ============================================================================
// UI Theme
// ============================================================================

namespace theme {
constexpr Color bg{15, 17, 22, 255};
constexpr Color sidebar{20, 23, 30, 255};
constexpr Color panel{26, 30, 39, 255};
constexpr Color panel_alt{35, 40, 52, 255};
constexpr Color border{47, 54, 70, 255};
constexpr Color text{226, 230, 239, 255};
constexpr Color muted{139, 148, 167, 255};
constexpr Color faint{85, 93, 112, 255};
constexpr Color accent{88, 166, 255, 255};
constexpr Color accent_dim{34, 66, 106, 255};
constexpr Color good{76, 201, 140, 255};
constexpr Color warn{240, 180, 70, 255};
constexpr Color bad{240, 98, 98, 255};

constexpr Color opt_sgd{240, 98, 98, 255};
constexpr Color opt_momentum{240, 180, 70, 255};
constexpr Color opt_rmsprop{76, 201, 140, 255};
constexpr Color opt_adam{88, 166, 255, 255};
}  // namespace theme

enum class Tone { Info, Good, Warn, Bad };
struct Status {
    std::string text;
    Tone tone = Tone::Info;
};

Color tone_color(Tone t) {
    switch (t) {
        case Tone::Good: return theme::good;
        case Tone::Warn: return theme::warn;
        case Tone::Bad:  return theme::bad;
        default:         return theme::muted;
    }
}

Rectangle rect(float x, float y, float w, float h) { return Rectangle{x, y, w, h}; }
Vector2 vec(float x, float y) { return Vector2{x, y}; }

Color mix(Color a, Color b, float t) {
    auto l = [t](unsigned char x, unsigned char y) {
        return static_cast<unsigned char>(std::lround(static_cast<float>(x) + (static_cast<float>(y) - static_cast<float>(x)) * t));
    };
    return Color{l(a.r, b.r), l(a.g, b.g), l(a.b, b.b), l(a.a, b.a)};
}

// ============================================================================
// Formatting & Math Helpers
// ============================================================================

std::string lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim(const std::string& s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) --e;
    return s.substr(b, e - b);
}

std::string fixed(double v, int precision) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
    return buf;
}

std::string human_bytes(std::uintmax_t b) {
    const double v = static_cast<double>(b);
    if (v >= 1024.0 * 1024.0) return fixed(v / (1024.0 * 1024.0), 1) + " MB";
    if (v >= 1024.0) return fixed(v / 1024.0, 1) + " KB";
    return std::to_string(b) + " B";
}

std::string format_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0) return "--";
    const auto s = static_cast<long long>(seconds);
    char buf[48];
    if (s >= 3600) {
        std::snprintf(buf, sizeof(buf), "%lldh %02lldm", s / 3600, (s / 60) % 60);
    } else {
        std::snprintf(buf, sizeof(buf), "%lldm %02llds", s / 60, s % 60);
    }
    return buf;
}

std::string tick_label(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3g", v);
    return buf;
}

bool is_regular_file_checked(const fs::path& p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

bool parse_size(const std::string& text, std::size_t& out) {
    const std::string t = trim(text);
    if (t.empty()) return false;
    std::size_t value = 0;
    const auto result = std::from_chars(t.data(), t.data() + t.size(), value);
    if (result.ec != std::errc{} || result.ptr != t.data() + t.size()) return false;
    out = value;
    return true;
}

bool parse_float(const std::string& text, float& out) {
    const std::string t = trim(text);
    if (t.empty()) return false;
    char* end = nullptr;
    const float v = std::strtof(t.c_str(), &end);
    if (end != t.c_str() + t.size()) return false;
    out = v;
    return true;
}

double nice_step(double range, int target_ticks) {
    if (!(range > 0.0) || !std::isfinite(range)) return 1.0;
    const double raw = range / static_cast<double>(std::max(1, target_ticks));
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double n = raw / mag;
    const double step = n < 1.5 ? 1.0 : (n < 3.0 ? 2.0 : (n < 7.0 ? 5.0 : 10.0));
    return step * mag;
}

// ============================================================================
// History Tracking
// ============================================================================

struct LossPoint {
    double samples = 0.0;
    float loss = 0.0F;
};

void decimate_history(std::vector<LossPoint>& pts, std::size_t cap) {
    if (pts.size() <= cap) return;
    std::vector<LossPoint> out;
    out.reserve(pts.size() / 2 + 1);
    for (std::size_t i = 0; i + 1 < pts.size(); i += 2) {
        out.push_back({(pts[i].samples + pts[i + 1].samples) * 0.5,
                       (pts[i].loss + pts[i + 1].loss) * 0.5F});
    }
    if (pts.size() % 2 == 1) out.push_back(pts.back());
    pts = std::move(out);
}

// ============================================================================
// Checkpoint Serialization
// ============================================================================

class NotACheckpoint : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string read_line_limited(std::istream& in, std::size_t limit = 4096) {
    std::string line;
    char c = 0;
    while (line.size() < limit && in.get(c)) {
        if (c == '\n') break;
        line.push_back(c);
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

std::size_t parameter_count(const std::vector<std::size_t>& arch) {
    std::size_t total = 0;
    for (std::size_t i = 0; i + 1 < arch.size(); ++i) {
        total += arch[i] * arch[i + 1] + arch[i + 1];
    }
    return total;
}

std::string arch_to_string(const std::vector<std::size_t>& arch) {
    std::string s;
    for (std::size_t i = 0; i < arch.size(); ++i) {
        if (i != 0) s += '-';
        s += std::to_string(arch[i]);
    }
    return s;
}

std::vector<std::size_t> parse_architecture(const std::string& text) {
    std::string cleaned = text;
    for (char& c : cleaned) {
        if (c == ',' || c == '-' || c == '>' || c == '\t') c = ' ';
    }
    std::istringstream in(cleaned);
    std::vector<std::size_t> arch;
    std::string token;
    while (in >> token) {
        const bool digits = std::all_of(token.begin(), token.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        });
        if (!digits) {
            throw std::runtime_error("Architecture contains non-numeric value '" + token + "'");
        }
        if (token.size() > 6) {
            throw std::runtime_error("Architecture layer width is too large: " + token);
        }
        arch.push_back(static_cast<std::size_t>(std::stoull(token)));
    }
    if (arch.size() < 2) {
        throw std::runtime_error("Architecture needs at least an input and an output size");
    }
    if (arch.size() > kMaxLayerCount + 1) {
        throw std::runtime_error("Architecture has too many layers");
    }
    for (const std::size_t width : arch) {
        if (width == 0 || width > kMaxLayerWidth) {
            throw std::runtime_error("Layer widths must be between 1 and " + std::to_string(kMaxLayerWidth));
        }
    }
    if (parameter_count(arch) > kMaxParameters) {
        throw std::runtime_error("Architecture has too many parameters");
    }
    return arch;
}

std::vector<std::size_t> read_checkpoint_header(std::istream& in) {
    std::string line = read_line_limited(in);
    std::istringstream first(line);
    std::string magic;
    if (!(first >> magic) || magic != kCheckpointMagic) {
        throw NotACheckpoint("Missing NEURAL_ENGINE_CHECKPOINT header");
    }
    long version = 0;
    if (!(first >> version)) {
        throw std::runtime_error("Checkpoint version number is missing or invalid");
    }
    if (version != kCheckpointVersion) {
        throw std::runtime_error("Unsupported checkpoint version " + std::to_string(version));
    }

    line = read_line_limited(in);
    const std::string tag = "ARCHITECTURE";
    if (line.rfind(tag, 0) != 0) {
        throw std::runtime_error("Missing ARCHITECTURE line");
    }
    return parse_architecture(line.substr(tag.size()));
}

std::vector<float> read_tensor_block(std::istream& in,
                                     const char* expected_tag,
                                     const std::vector<std::size_t>& expected_shape) {
    std::string tag;
    if (!(in >> tag) || tag != expected_tag) {
        throw std::runtime_error(std::string("Expected '") + expected_tag + "' section");
    }
    std::size_t rank = 0;
    if (!(in >> rank) || rank != expected_shape.size()) {
        throw std::runtime_error(std::string(expected_tag) + " has the wrong rank");
    }
    std::size_t count = 1;
    for (const std::size_t expected : expected_shape) {
        std::size_t dim = 0;
        if (!(in >> dim) || dim != expected) {
            throw std::runtime_error(std::string(expected_tag) + " shape mismatch");
        }
        count *= dim;
    }
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        float v = 0.0F;
        if (!(in >> v)) {
            throw std::runtime_error(std::string(expected_tag) + " truncated or invalid data");
        }
        if (!std::isfinite(v)) {
            throw std::runtime_error(std::string(expected_tag) + " contains non-finite value");
        }
        values[i] = v;
    }
    return values;
}

void write_tensor(std::ostream& file, const char* name, const neural::Tensor& tensor) {
    file << name << '\n';
    file << tensor.rank();
    for (const std::size_t dimension : tensor.shape()) {
        file << ' ' << dimension;
    }
    file << '\n';

    std::size_t count = 0;
    for (const float value : tensor.data()) {
        file << value << ' ';
        if (++count % 8 == 0) file << '\n';
    }
    if (count % 8 != 0) file << '\n';
}

// ============================================================================
// Network Architecture
// ============================================================================

class Network {
public:
    Network(std::vector<std::size_t> architecture,
            neural::WeightInit hidden_init,
            std::uint32_t seed)
        : architecture_(std::move(architecture)) {
        if (architecture_.size() < 2) {
            throw std::invalid_argument("Network needs at least two sizes");
        }
        for (std::size_t i = 0; i + 1 < architecture_.size(); ++i) {
            const bool last = i + 2 == architecture_.size();
            const neural::WeightInit init = last ? neural::WeightInit::Xavier : hidden_init;
            layers_.push_back(std::make_unique<neural::Dense>(
                architecture_[i], architecture_[i + 1], init,
                seed + 1U + static_cast<std::uint32_t>(i)));
        }
        relus_ = std::vector<neural::ReLU>(layers_.size() - 1);
    }

    Network(Network&&) noexcept = default;
    Network& operator=(Network&&) noexcept = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    [[nodiscard]] const std::vector<std::size_t>& architecture() const noexcept { return architecture_; }
    [[nodiscard]] std::size_t layer_count() const noexcept { return layers_.size(); }
    neural::Dense& layer(std::size_t i) { return *layers_.at(i); }
    [[nodiscard]] const neural::Dense& layer(std::size_t i) const { return *layers_.at(i); }

    [[nodiscard]] bool is_mnist_compatible() const noexcept {
        return architecture_.front() == kMnistInputs && architecture_.back() == kMnistClasses;
    }

    [[nodiscard]] Network clone() const {
        Network copy(architecture_, neural::WeightInit::He, kBaseSeed);
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const neural::Dense& src = *layers_[i];
            neural::Dense& dst = *copy.layers_[i];
            const auto sw = src.weights().data();
            const auto sb = src.bias().data();
            auto dw = dst.weights().data();
            auto db = dst.bias().data();
            std::copy(sw.begin(), sw.end(), dw.begin());
            std::copy(sb.begin(), sb.end(), db.begin());
        }
        return copy;
    }

    neural::Tensor forward(const neural::Tensor& input) {
        neural::Tensor x = layers_[0]->forward(input);
        for (std::size_t i = 1; i < layers_.size(); ++i) {
            x = relus_[i - 1].forward(x);
            x = layers_[i]->forward(x);
        }
        return x;
    }

    // One forward/backward/update step on a single sample (batch size 1). If `correct` is given
    // it is set to whether the prediction made *before* the update matched the target.
    template <typename Opt>
    float step(const neural::Tensor& input, std::size_t target, Opt& optimizer, bool* correct = nullptr) {
        const float loss = forward_backward(input, target, correct);
        for (auto& l : layers_) {
            optimizer.step(*l);
        }
        return loss;
    }

    // Mini-batch support: call accumulate() once per sample of the batch (forward + backward, the
    // per-sample gradients are summed), then apply_accumulated() once with the batch size. The
    // optimiser receives the AVERAGED gradient and performs a single update.
    float accumulate(const neural::Tensor& input, std::size_t target, bool* correct = nullptr) {
        const float loss = forward_backward(input, target, correct);
        ensure_accumulators();
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const auto gw = layers_[i]->weight_gradients().data();
            const auto gb = layers_[i]->bias_gradients().data();
            for (std::size_t j = 0; j < gw.size(); ++j) acc_w_[i][j] += gw[j];
            for (std::size_t j = 0; j < gb.size(); ++j) acc_b_[i][j] += gb[j];
        }
        return loss;
    }

    template <typename Opt>
    void apply_accumulated(Opt& optimizer, std::size_t count) {
        if (count == 0) return;
        ensure_accumulators();
        const float inv = 1.0F / static_cast<float>(count);
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            for (float& v : acc_w_[i]) v *= inv;
            for (float& v : acc_b_[i]) v *= inv;
            optimizer.apply(*layers_[i], acc_w_[i], acc_b_[i]);
            std::fill(acc_w_[i].begin(), acc_w_[i].end(), 0.0F);
            std::fill(acc_b_[i].begin(), acc_b_[i].end(), 0.0F);
        }
    }

    float train_step(const neural::Tensor& input, std::size_t target, const neural::SGD& optimizer) {
        return step(input, target, const_cast<neural::SGD&>(optimizer));
    }

    void save(const fs::path& path) const {
        std::error_code ec;
        const fs::path parent = path.parent_path();
        if (!parent.empty() && !fs::exists(parent, ec)) {
            fs::create_directories(parent, ec);
            if (ec) {
                throw std::runtime_error("Cannot create directory " + parent.string() + ": " + ec.message());
            }
        }
        fs::path tmp = path;
        tmp += ".tmp";
        {
            std::ofstream file(tmp);
            if (!file) {
                throw std::runtime_error("Could not open for writing: " + tmp.string());
            }
            file << std::setprecision(std::numeric_limits<float>::max_digits10);
            file << kCheckpointMagic << ' ' << kCheckpointVersion << '\n';
            file << "ARCHITECTURE";
            for (const std::size_t w : architecture_) file << ' ' << w;
            file << '\n';
            for (std::size_t i = 0; i < layers_.size(); ++i) {
                file << "LAYER fc" << (i + 1) << '\n';
                write_tensor(file, "WEIGHTS", layers_[i]->weights());
                write_tensor(file, "BIAS", layers_[i]->bias());
            }
            file.flush();
            if (!file) {
                file.close();
                fs::remove(tmp, ec);
                throw std::runtime_error("Failed while writing checkpoint " + tmp.string());
            }
        }
        fs::rename(tmp, path, ec);
        if (ec) {
            std::error_code ignore;
            fs::remove(tmp, ignore);
            throw std::runtime_error("Could not move checkpoint into place: " + ec.message());
        }
    }

private:
    float forward_backward(const neural::Tensor& input, std::size_t target, bool* correct) {
        const neural::Tensor logits = forward(input);
        if (correct != nullptr) {
            const auto d = logits.data();
            const auto best = static_cast<std::size_t>(std::distance(d.begin(), std::max_element(d.begin(), d.end())));
            *correct = (best == target);
        }
        const neural::LossResult result = neural::softmax_cross_entropy(logits, target);

        neural::Tensor gradient = layers_.back()->backward(result.logits_gradient);
        for (std::size_t i = layers_.size() - 1; i > 0; --i) {
            const std::size_t k = i - 1;
            gradient = relus_[k].backward(gradient);
            gradient = layers_[k]->backward(gradient);
        }
        return result.loss;
    }

    void ensure_accumulators() {
        if (acc_w_.size() == layers_.size()) return;
        acc_w_.assign(layers_.size(), std::vector<float>{});
        acc_b_.assign(layers_.size(), std::vector<float>{});
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            acc_w_[i].assign(layers_[i]->weight_gradients().data().size(), 0.0F);
            acc_b_[i].assign(layers_[i]->bias_gradients().data().size(), 0.0F);
        }
    }

    std::vector<std::size_t> architecture_;
    std::vector<std::unique_ptr<neural::Dense>> layers_;
    std::vector<neural::ReLU> relus_;
    std::vector<std::vector<float>> acc_w_, acc_b_;  // summed per-sample gradients (mini-batch)
};

Network load_checkpoint(const fs::path& path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) {
        throw std::runtime_error("Checkpoint file not found: " + path.string());
    }
    std::error_code size_ec;
    const std::uintmax_t size = fs::file_size(path, size_ec);
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open checkpoint: " + path.string());
    }

    const std::vector<std::size_t> arch = read_checkpoint_header(in);
    if (!size_ec && size < parameter_count(arch) * 2) {
        throw std::runtime_error("File is too small for the declared architecture");
    }

    Network net(arch, neural::WeightInit::He, kBaseSeed);

    for (std::size_t i = 0; i < net.layer_count(); ++i) {
        std::string word, name;
        if (!(in >> word >> name) || word != "LAYER") {
            throw std::runtime_error("Expected 'LAYER fc" + std::to_string(i + 1) + "'");
        }
        const std::string expected = "fc" + std::to_string(i + 1);
        if (name != expected) {
            throw std::runtime_error("Unexpected layer name '" + name + "' (expected " + expected + ")");
        }
        const std::size_t inputs = arch[i];
        const std::size_t outputs = arch[i + 1];

        const std::vector<float> w = read_tensor_block(in, "WEIGHTS", {outputs, inputs});
        const std::vector<float> b = read_tensor_block(in, "BIAS", {outputs});

        neural::Dense& layer = net.layer(i);
        auto wd = layer.weights().data();
        auto bd = layer.bias().data();
        if (wd.size() != w.size() || bd.size() != b.size()) {
            throw std::runtime_error("Parameter count mismatch in " + name);
        }
        std::copy(w.begin(), w.end(), wd.begin());
        std::copy(b.begin(), b.end(), bd.begin());
    }

    std::string extra;
    if (in >> extra) {
        throw std::runtime_error("Unexpected trailing data after last layer");
    }
    return net;
}

// ============================================================================
// Optimisers (shared by the single trainer and the arena)
// ============================================================================

enum class OptimiserType { SGD = 0, Momentum, RMSprop, Adam, Count };

const char* optimiser_name(OptimiserType t) {
    switch (t) {
        case OptimiserType::SGD:      return "SGD";
        case OptimiserType::Momentum: return "Momentum";
        case OptimiserType::RMSprop:  return "RMSprop";
        case OptimiserType::Adam:     return "Adam";
        default:                      return "?";
    }
}

Color optimiser_color(OptimiserType t) {
    switch (t) {
        case OptimiserType::SGD:      return theme::opt_sgd;
        case OptimiserType::Momentum: return theme::opt_momentum;
        case OptimiserType::RMSprop:  return theme::opt_rmsprop;
        case OptimiserType::Adam:     return theme::opt_adam;
        default:                      return WHITE;
    }
}

class ArenaSGD {
public:
    explicit ArenaSGD(float lr) : lr_(lr) {}
    void set_lr(float lr) noexcept { lr_ = lr; }
    void step(neural::Dense& layer) const {
        apply(layer, layer.weight_gradients().data(), layer.bias_gradients().data());
    }
    template <typename GW, typename GB>
    void apply(neural::Dense& layer, const GW& gw, const GB& gb) const {
        auto w = layer.weights().data();
        auto b = layer.bias().data();
        for (std::size_t i = 0; i < w.size(); ++i) w[i] -= lr_ * gw[i];
        for (std::size_t i = 0; i < b.size(); ++i) b[i] -= lr_ * gb[i];
    }
private:
    float lr_;
};

class ArenaMomentum {
public:
    ArenaMomentum(float lr, float momentum) : lr_(lr), momentum_(momentum) {}
    void set_lr(float lr) noexcept { lr_ = lr; }
    void set_momentum(float m) noexcept { momentum_ = m; }
    void step(neural::Dense& layer) {
        apply(layer, layer.weight_gradients().data(), layer.bias_gradients().data());
    }
    template <typename GW, typename GB>
    void apply(neural::Dense& layer, const GW& gw, const GB& gb) {
        auto w = layer.weights().data();
        auto b = layer.bias().data();
        auto& s = states_[&layer];
        if (s.vw.size() != gw.size()) {
            s.vw.assign(gw.size(), 0.0F);
            s.vb.assign(gb.size(), 0.0F);
        }
        for (std::size_t i = 0; i < gw.size(); ++i) {
            s.vw[i] = momentum_ * s.vw[i] + gw[i];
            w[i] -= lr_ * s.vw[i];
        }
        for (std::size_t i = 0; i < gb.size(); ++i) {
            s.vb[i] = momentum_ * s.vb[i] + gb[i];
            b[i] -= lr_ * s.vb[i];
        }
    }
private:
    struct State { std::vector<float> vw, vb; };
    float lr_, momentum_;
    std::unordered_map<neural::Dense*, State> states_;
};

class ArenaRMSprop {
public:
    ArenaRMSprop(float lr, float decay, float eps = 1e-8F)
        : lr_(lr), decay_(decay), eps_(eps) {}
    void set_lr(float lr) noexcept { lr_ = lr; }
    void set_decay(float d) noexcept { decay_ = d; }
    void set_eps(float e) noexcept { eps_ = e; }
    void step(neural::Dense& layer) {
        apply(layer, layer.weight_gradients().data(), layer.bias_gradients().data());
    }
    template <typename GW, typename GB>
    void apply(neural::Dense& layer, const GW& gw, const GB& gb) {
        auto w = layer.weights().data();
        auto b = layer.bias().data();
        auto& s = states_[&layer];
        if (s.vw.size() != gw.size()) {
            s.vw.assign(gw.size(), 0.0F);
            s.vb.assign(gb.size(), 0.0F);
        }
        for (std::size_t i = 0; i < gw.size(); ++i) {
            s.vw[i] = decay_ * s.vw[i] + (1.0F - decay_) * gw[i] * gw[i];
            w[i] -= lr_ * gw[i] / (std::sqrt(s.vw[i]) + eps_);
        }
        for (std::size_t i = 0; i < gb.size(); ++i) {
            s.vb[i] = decay_ * s.vb[i] + (1.0F - decay_) * gb[i] * gb[i];
            b[i] -= lr_ * gb[i] / (std::sqrt(s.vb[i]) + eps_);  // bias uses its OWN accumulator
        }
    }
private:
    struct State { std::vector<float> vw, vb; };
    float lr_, decay_, eps_;
    std::unordered_map<neural::Dense*, State> states_;
};

class ArenaAdam {
public:
    explicit ArenaAdam(float lr, float b1 = 0.9F, float b2 = 0.999F, float eps = 1e-8F)
        : lr_(lr), b1_(b1), b2_(b2), eps_(eps) {}
    void set_lr(float lr) noexcept { lr_ = lr; }
    void set_b1(float b1) noexcept { b1_ = b1; }
    void set_b2(float b2) noexcept { b2_ = b2; }
    void set_eps(float e) noexcept { eps_ = e; }
    void step(neural::Dense& layer) {
        apply(layer, layer.weight_gradients().data(), layer.bias_gradients().data());
    }
    template <typename GW, typename GB>
    void apply(neural::Dense& layer, const GW& gw, const GB& gb) {
        auto w = layer.weights().data();
        auto b = layer.bias().data();
        auto& s = states_[&layer];
        if (s.mw.size() != gw.size()) {
            s.mw.assign(gw.size(), 0.0F);
            s.vw.assign(gw.size(), 0.0F);
            s.mb.assign(gb.size(), 0.0F);
            s.vb.assign(gb.size(), 0.0F);
            s.step_count = 0;
        }
        ++s.step_count;
        const float c1 = 1.0F - std::pow(b1_, static_cast<float>(s.step_count));
        const float c2 = 1.0F - std::pow(b2_, static_cast<float>(s.step_count));

        for (std::size_t i = 0; i < gw.size(); ++i) {
            s.mw[i] = b1_ * s.mw[i] + (1.0F - b1_) * gw[i];
            s.vw[i] = b2_ * s.vw[i] + (1.0F - b2_) * gw[i] * gw[i];
            const float m_hat = s.mw[i] / c1;
            const float v_hat = s.vw[i] / c2;
            w[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
        for (std::size_t i = 0; i < gb.size(); ++i) {
            s.mb[i] = b1_ * s.mb[i] + (1.0F - b1_) * gb[i];
            s.vb[i] = b2_ * s.vb[i] + (1.0F - b2_) * gb[i] * gb[i];
            const float m_hat = s.mb[i] / c1;
            const float v_hat = s.vb[i] / c2;
            b[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
private:
    struct State {
        std::vector<float> mw, vw, mb, vb;
        std::size_t step_count{0};
    };
    float lr_, b1_, b2_, eps_;
    std::unordered_map<neural::Dense*, State> states_;
};

// ============================================================================
// Model Discovery System
// ============================================================================

struct ModelEntry {
    fs::path path;
    std::string rel;
    std::string name;
    bool valid = false;
    std::string error;
    std::vector<std::size_t> arch;
    std::uintmax_t bytes = 0;
};

struct ScanResult {
    std::vector<ModelEntry> entries;
    std::size_t ignored_text_files = 0;
    std::size_t unreadable = 0;
    bool truncated = false;
    std::string message;
};

bool is_ignored_directory(const std::string& name) {
    static const std::vector<std::string> exact = {
        ".git", ".cache", ".idea", ".vscode", "CMakeFiles", "_deps", "third_party",
        "external", "vendor", "node_modules", "Testing", "out"};
    if (std::find(exact.begin(), exact.end(), name) != exact.end()) return true;
    return name.rfind("build", 0) == 0 || name.rfind("cmake-build", 0) == 0;
}

ScanResult scan_models(const fs::path& root) {
    constexpr std::size_t kMaxVisited = 200000;
    constexpr int kMaxDepth = 8;

    ScanResult result;
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        result.message = "Not a directory: " + root.string();
        return result;
    }

    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        result.message = "Cannot read directory: " + ec.message();
        return result;
    }
    const fs::recursive_directory_iterator end;

    std::size_t visited = 0;
    while (it != end) {
        if (++visited > kMaxVisited) {
            result.truncated = true;
            break;
        }
        const fs::directory_entry& de = *it;
        std::error_code e2;
        if (de.is_directory(e2)) {
            if (is_ignored_directory(de.path().filename().string()) || it.depth() >= kMaxDepth) {
                it.disable_recursion_pending();
            }
        } else if (de.is_regular_file(e2) && lower(de.path().extension().string()) == ".txt") {
            std::ifstream f(de.path());
            if (!f) {
                ++result.unreadable;
            } else {
                ModelEntry entry;
                entry.path = de.path();
                entry.rel = de.path().lexically_relative(root).generic_string();
                entry.name = de.path().filename().string();
                std::error_code e3;
                entry.bytes = de.file_size(e3);
                bool is_checkpoint = true;
                try {
                    entry.arch = read_checkpoint_header(f);
                    entry.valid = true;
                } catch (const NotACheckpoint&) {
                    is_checkpoint = false;
                } catch (const std::exception& ex) {
                    entry.valid = false;
                    entry.error = ex.what();
                }
                if (is_checkpoint) {
                    result.entries.push_back(std::move(entry));
                } else {
                    ++result.ignored_text_files;
                }
            }
        }
        it.increment(ec);
        if (ec) {
            result.message = "Scan stopped early: " + ec.message();
            break;
        }
    }

    std::sort(result.entries.begin(), result.entries.end(),
              [](const ModelEntry& a, const ModelEntry& b) { return lower(a.rel) < lower(b.rel); });
    return result;
}

// ============================================================================
// Dataset Paths
// ============================================================================

struct DatasetFiles {
    fs::path train_images, train_labels, test_images, test_labels;
    [[nodiscard]] bool has_train() const { return is_regular_file_checked(train_images) && is_regular_file_checked(train_labels); }
    [[nodiscard]] bool has_test() const { return is_regular_file_checked(test_images) && is_regular_file_checked(test_labels); }
};

DatasetFiles dataset_files(const fs::path& dir) {
    return {dir / kTrainImages, dir / kTrainLabels, dir / kTestImages, dir / kTestLabels};
}

fs::path find_project_root() {
    std::error_code ec;
    fs::path p = fs::current_path(ec);
    if (ec) return ".";
    const fs::path start = p;
    for (int i = 0; i < 5; ++i) {
        if (fs::exists(p / "CMakeLists.txt", ec) && fs::is_directory(p / "examples", ec)) return p;
        if (!p.has_parent_path() || p.parent_path() == p) break;
        p = p.parent_path();
    }
    return start;
}

fs::path default_dataset_dir(const fs::path& root) {
    std::error_code ec;
    const std::vector<fs::path> candidates = {root / "data" / "mnist",
                                              fs::current_path(ec) / "data" / "mnist",
                                              fs::current_path(ec) / ".." / "data" / "mnist"};
    for (const fs::path& c : candidates) {
        if (is_regular_file_checked(c / kTrainImages)) return fs::weakly_canonical(c, ec);
    }
    return root / "data" / "mnist";
}

// ============================================================================
// Drawing Canvas & Preprocessing
// ============================================================================

void stamp(std::vector<std::uint8_t>& canvas, float cx, float cy, float radius) {
    const int x0 = std::max(0, static_cast<int>(std::floor(cx - radius)));
    const int x1 = std::min(kCanvas - 1, static_cast<int>(std::ceil(cx + radius)));
    const int y0 = std::max(0, static_cast<int>(std::floor(cy - radius)));
    const int y1 = std::min(kCanvas - 1, static_cast<int>(std::ceil(cy + radius)));
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float dx = static_cast<float>(x) + 0.5F - cx;
            const float dy = static_cast<float>(y) + 0.5F - cy;
            const float d = std::sqrt(dx * dx + dy * dy);
            if (d > radius) continue;
            const float t = d / radius;
            const float v = t < 0.6F ? 1.0F : (1.0F - t) / 0.4F;
            const auto value = static_cast<std::uint8_t>(std::lround(v * 255.0F));
            std::uint8_t& px = canvas[static_cast<std::size_t>(y * kCanvas + x)];
            px = std::max(px, value);
        }
    }
}

void stroke(std::vector<std::uint8_t>& canvas, float x0, float y0, float x1, float y1) {
    const float dist = std::hypot(x1 - x0, y1 - y0);
    const int steps = std::max(1, static_cast<int>(std::ceil(dist / 2.0F)));
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        stamp(canvas, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, kBrushRadius);
    }
}

struct Preprocessed {
    std::array<float, kMnistInputs> pixels{};
    bool empty = true;
};

Preprocessed preprocess_digit(const std::vector<std::uint8_t>& canvas, int size) {
    Preprocessed out;
    int minx = size, miny = size, maxx = -1, maxy = -1;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (canvas[static_cast<std::size_t>(y * size + x)] > 24) {
                minx = std::min(minx, x);
                maxx = std::max(maxx, x);
                miny = std::min(miny, y);
                maxy = std::max(maxy, y);
            }
        }
    }
    if (maxx < 0) return out;

    const int bw = maxx - minx + 1;
    const int bh = maxy - miny + 1;
    const double scale = 20.0 / static_cast<double>(std::max(bw, bh));
    const int tw = std::clamp(static_cast<int>(std::lround(bw * scale)), 1, 20);
    const int th = std::clamp(static_cast<int>(std::lround(bh * scale)), 1, 20);
    const int offx = (28 - tw) / 2;
    const int offy = (28 - th) / 2;

    std::array<float, kMnistInputs> box{};
    for (int oy = 0; oy < th; ++oy) {
        for (int ox = 0; ox < tw; ++ox) {
            const double sx0 = minx + static_cast<double>(ox) * bw / tw;
            const double sx1 = minx + static_cast<double>(ox + 1) * bw / tw;
            const double sy0 = miny + static_cast<double>(oy) * bh / th;
            const double sy1 = miny + static_cast<double>(oy + 1) * bh / th;
            const int ix0 = std::clamp(static_cast<int>(std::floor(sx0)), minx, maxx);
            const int ix1 = std::clamp(std::max(ix0 + 1, static_cast<int>(std::ceil(sx1))), minx + 1, maxx + 1);
            const int iy0 = std::clamp(static_cast<int>(std::floor(sy0)), miny, maxy);
            const int iy1 = std::clamp(std::max(iy0 + 1, static_cast<int>(std::ceil(sy1))), miny + 1, maxy + 1);
            double sum = 0.0;
            for (int y = iy0; y < iy1; ++y) {
                for (int x = ix0; x < ix1; ++x) {
                    sum += canvas[static_cast<std::size_t>(y * size + x)];
                }
            }
            const double avg = sum / (static_cast<double>(ix1 - ix0) * (iy1 - iy0));
            box[static_cast<std::size_t>((offy + oy) * 28 + offx + ox)] = static_cast<float>(avg / 255.0);
        }
    }

    double mass = 0.0, mx = 0.0, my = 0.0;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const double v = box[static_cast<std::size_t>(y * 28 + x)];
            mass += v;
            mx += v * x;
            my += v * y;
        }
    }
    int shift_x = 0, shift_y = 0;
    if (mass > 0.0) {
        shift_x = static_cast<int>(std::lround(13.5 - mx / mass));
        shift_y = static_cast<int>(std::lround(13.5 - my / mass));
    }
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const int nx = x + shift_x;
            const int ny = y + shift_y;
            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                out.pixels[static_cast<std::size_t>(ny * 28 + nx)] =
                    std::clamp(box[static_cast<std::size_t>(y * 28 + x)], 0.0F, 1.0F);
            }
        }
    }
    out.empty = false;
    return out;
}

neural::Tensor to_tensor(const std::array<float, kMnistInputs>& pixels) {
    neural::Tensor t({kMnistInputs}, 0.0F);
    auto d = t.data();
    for (std::size_t i = 0; i < kMnistInputs; ++i) d[i] = pixels[i];
    return t;
}

struct Prediction {
    std::array<float, kMnistClasses> probabilities{};
    std::size_t digit = 0;
};

Prediction predict(Network& net, const neural::Tensor& input) {
    if (!net.is_mnist_compatible()) {
        throw std::runtime_error("Model is not 784-input / 10-output");
    }
    const neural::Tensor logits = net.forward(input);
    if (logits.size() != kMnistClasses) {
        throw std::runtime_error("Model produced an unexpected number of logits");
    }
    const neural::LossResult r = neural::softmax_cross_entropy(logits, 0);
    Prediction p;
    const auto probs = r.probabilities.data();
    for (std::size_t i = 0; i < kMnistClasses; ++i) p.probabilities[i] = probs[i];
    p.digit = static_cast<std::size_t>(
        std::distance(p.probabilities.begin(),
                      std::max_element(p.probabilities.begin(), p.probabilities.end())));
    return p;
}

// ============================================================================
// Single Model Evaluation Subsystem
// ============================================================================

struct EvalSummary {
    std::size_t correct = 0;
    std::size_t total = 0;
    double loss = 0.0;
    bool cancelled = false;
    bool test_split = true;
    std::string model_name;
    std::array<std::array<std::size_t, kMnistClasses>, kMnistClasses> confusion{};
    std::vector<std::size_t> misses;
    std::size_t miss_count = 0;

    [[nodiscard]] double accuracy() const {
        return total ? 100.0 * static_cast<double>(correct) / static_cast<double>(total) : 0.0;
    }
    [[nodiscard]] double mean_loss() const { return total ? loss / static_cast<double>(total) : 0.0; }
};

EvalSummary evaluate_detailed(Network& net,
                              const neural::MNISTDataset& data,
                              const std::stop_token& stop,
                              const std::function<void(std::size_t, std::size_t)>& progress) {
    EvalSummary r;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (stop.stop_requested()) {
            r.cancelled = true;
            return r;
        }
        const neural::Tensor logits = net.forward(data.image(i));
        if (logits.size() != kMnistClasses) {
            throw std::runtime_error("Model produced an unexpected number of logits");
        }
        const std::size_t target = data.label(i);
        if (target >= kMnistClasses) {
            throw std::runtime_error("Dataset contains an out-of-range label");
        }
        const neural::LossResult lr = neural::softmax_cross_entropy(logits, target);
        const auto p = lr.probabilities.data();
        const auto best = static_cast<std::size_t>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
        r.loss += static_cast<double>(lr.loss);
        ++r.confusion[target][best];
        if (best == target) {
            ++r.correct;
        } else {
            ++r.miss_count;
            if (r.misses.size() < kMaxMisses) r.misses.push_back(i);
        }
        ++r.total;
        if ((i + 1) % 200 == 0) progress(i + 1, r.correct);
    }
    progress(data.size(), r.correct);
    return r;
}

void weakest_class(const EvalSummary& s, std::size_t& cls, double& acc) {
    cls = 0;
    acc = 101.0;
    for (std::size_t r = 0; r < kMnistClasses; ++r) {
        std::size_t row_total = 0;
        for (std::size_t c = 0; c < kMnistClasses; ++c) row_total += s.confusion[r][c];
        if (row_total == 0) continue;
        const double a = 100.0 * static_cast<double>(s.confusion[r][r]) / static_cast<double>(row_total);
        if (a < acc) {
            acc = a;
            cls = r;
        }
    }
    if (acc > 100.0) acc = 0.0;
}

class Evaluator {
public:
    [[nodiscard]] bool running() const { return state_.load() == 1; }
    [[nodiscard]] std::size_t done() const { return done_.load(); }
    [[nodiscard]] std::size_t total() const { return total_.load(); }
    [[nodiscard]] double running_accuracy() const {
        const std::size_t d = done_.load();
        return d ? 100.0 * static_cast<double>(correct_.load()) / static_cast<double>(d) : 0.0;
    }

    void start(Network net,
               std::shared_ptr<const neural::MNISTDataset> data,
               bool test_split,
               std::string model_name) {
        if (running()) return;
        if (worker_.joinable()) worker_.join();
        done_ = 0;
        correct_ = 0;
        total_ = data->size();
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            summary_ = EvalSummary{};
            error_.clear();
        }
        state_ = 1;
        worker_ = std::jthread([this, n = std::move(net), d = std::move(data), test_split,
                                name = std::move(model_name)](std::stop_token st) mutable {
            run(st, n, *d, test_split, name);
        });
    }

    void request_stop() {
        if (worker_.joinable()) worker_.request_stop();
    }

    bool take_result(EvalSummary& out, std::string& error) {
        if (state_.load() != 2) return false;
        const std::lock_guard<std::mutex> lock(mutex_);
        out = std::move(summary_);
        error = error_;
        state_ = 0;
        return true;
    }

private:
    void run(const std::stop_token& st, Network& net, const neural::MNISTDataset& data,
             bool test_split, const std::string& name) {
        EvalSummary summary;
        std::string error;
        try {
            summary = evaluate_detailed(net, data, st, [this](std::size_t d, std::size_t c) {
                done_ = d;
                correct_ = c;
            });
        } catch (const std::exception& e) {
            error = e.what();
        } catch (...) {
            error = "Unknown error during evaluation";
        }
        summary.test_split = test_split;
        summary.model_name = name;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            summary_ = std::move(summary);
            error_ = std::move(error);
        }
        state_ = 2;
    }

    std::atomic<int> state_{0};
    std::atomic<std::size_t> done_{0}, correct_{0}, total_{0};
    mutable std::mutex mutex_;
    EvalSummary summary_;
    std::string error_;
    std::jthread worker_;
};

// ============================================================================
// Single Model Trainer (any optimiser)
// ============================================================================

struct TrainConfig {
    fs::path dataset_dir;
    fs::path output;
    std::optional<fs::path> continue_from;
    std::vector<std::size_t> arch;
    neural::WeightInit init = neural::WeightInit::He;
    OptimiserType optimiser = OptimiserType::SGD;
    float learning_rate = 0.01F;
    float momentum = 0.9F;  // momentum / RMSprop decay / Adam beta1
    std::size_t batch_size = 1;
    std::size_t epochs = 5;
    std::size_t max_samples = 0;
    bool evaluate_test = false;
    bool save_on_stop = true;
};

enum class TrainPhase { Idle, Loading, Training, Saving, Evaluating, Done, Cancelled, Failed };

const char* phase_name(TrainPhase p) {
    switch (p) {
        case TrainPhase::Idle: return "Idle";
        case TrainPhase::Loading: return "Loading";
        case TrainPhase::Training: return "Training";
        case TrainPhase::Saving: return "Saving";
        case TrainPhase::Evaluating: return "Evaluating";
        case TrainPhase::Done: return "Finished";
        case TrainPhase::Cancelled: return "Stopped";
        case TrainPhase::Failed: return "Failed";
    }
    return "?";
}

struct TrainSnapshot {
    TrainPhase phase = TrainPhase::Idle;
    std::size_t epoch = 0, epochs = 0, in_epoch = 0, per_epoch = 0;
    std::size_t total_samples = 0, eval_done = 0, eval_total = 0;
    double recent_loss = 0.0, accuracy = -1.0, test_loss = 0.0, elapsed = 0.0;
    std::string message, error, saved_path;
    std::vector<LossPoint> history;
};

class Trainer {
public:
    ~Trainer() = default;

    [[nodiscard]] bool running() const {
        const TrainPhase p = phase_.load();
        return p == TrainPhase::Loading || p == TrainPhase::Training ||
               p == TrainPhase::Saving || p == TrainPhase::Evaluating;
    }

    void start(TrainConfig cfg) {
        if (running()) return;
        if (worker_.joinable()) worker_.join();
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            message_.clear();
            error_.clear();
            saved_.clear();
            history_.clear();
        }
        epoch_ = 0; epochs_ = cfg.epochs; in_epoch_ = 0; per_epoch_ = 0;
        total_samples_ = 0; eval_done_ = 0; eval_total_ = 0;
        recent_loss_ = 0.0; accuracy_ = -1.0; test_loss_ = 0.0;
        started_ns_ = now_ns();
        finished_ns_ = 0;
        phase_ = TrainPhase::Loading;
        worker_ = std::jthread([this, c = std::move(cfg)](std::stop_token st) { run(st, c); });
    }

    void request_stop() {
        if (worker_.joinable()) worker_.request_stop();
    }

    [[nodiscard]] TrainSnapshot snapshot(bool with_history) const {
        TrainSnapshot s;
        s.phase = phase_.load();
        s.epoch = epoch_; s.epochs = epochs_; s.in_epoch = in_epoch_; s.per_epoch = per_epoch_;
        s.total_samples = total_samples_; s.eval_done = eval_done_; s.eval_total = eval_total_;
        s.recent_loss = recent_loss_; s.accuracy = accuracy_; s.test_loss = test_loss_;
        const long long start = started_ns_;
        const long long fin = finished_ns_;
        if (start != 0) s.elapsed = static_cast<double>((fin != 0 ? fin : now_ns()) - start) * 1e-9;
        const std::lock_guard<std::mutex> lock(mutex_);
        s.message = message_;
        s.error = error_;
        s.saved_path = saved_;
        if (with_history) s.history = history_;
        return s;
    }

private:
    static long long now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void set_message(const std::string& m) {
        const std::lock_guard<std::mutex> lock(mutex_);
        message_ = m;
    }

    void finish(TrainPhase p) {
        finished_ns_ = now_ns();
        phase_ = p;
    }

    void run(const std::stop_token& st, const TrainConfig& cfg) {
        try {
            set_message("Loading datasets...");
            const DatasetFiles files = dataset_files(cfg.dataset_dir);
            const neural::MNISTDataset training(files.train_images, files.train_labels);
            if (training.size() == 0) {
                throw std::runtime_error("Training dataset is empty");
            }
            std::optional<neural::MNISTDataset> testing;
            if (cfg.evaluate_test) testing.emplace(files.test_images, files.test_labels);

            std::optional<Network> net;
            if (cfg.continue_from) {
                set_message("Loading checkpoint...");
                net.emplace(load_checkpoint(*cfg.continue_from));
            } else {
                net.emplace(cfg.arch, cfg.init, kBaseSeed);
            }
            if (!net->is_mnist_compatible()) {
                throw std::runtime_error("Model must have 784 inputs and 10 outputs for MNIST");
            }
            if (st.stop_requested()) {
                set_message("Stopped before training started.");
                finish(TrainPhase::Cancelled);
                return;
            }

            std::vector<std::size_t> indices(training.size());
            std::iota(indices.begin(), indices.end(), std::size_t{0});
            std::mt19937 rng(kShuffleSeed);

            const std::size_t per_epoch =
                cfg.max_samples == 0 ? training.size() : std::min(cfg.max_samples, training.size());
            per_epoch_ = per_epoch;
            phase_ = TrainPhase::Training;
            set_message("Training (batch size " + std::to_string(cfg.batch_size) + ", " + optimiser_name(cfg.optimiser) + ")...");

            bool stopped = false;
            const std::size_t batch = std::max<std::size_t>(1, cfg.batch_size);
            std::size_t in_batch = 0;
            double window_loss = 0.0;
            std::size_t window_count = 0;

            auto publish = [&]() {
                if (window_count == 0) return;
                const double mean = window_loss / static_cast<double>(window_count);
                recent_loss_ = mean;
                {
                    const std::lock_guard<std::mutex> lock(mutex_);
                    history_.push_back({static_cast<double>(total_samples_.load()),
                                        static_cast<float>(mean)});
                    decimate_history(history_, kHistoryCap);
                }
                window_loss = 0.0;
                window_count = 0;
            };

            auto train_loop = [&](auto& opt) {
                for (std::size_t epoch = 0; epoch < cfg.epochs && !stopped; ++epoch) {
                    epoch_ = epoch + 1;
                    in_epoch_ = 0;
                    std::shuffle(indices.begin(), indices.end(), rng);
                    for (std::size_t step = 0; step < per_epoch; ++step) {
                        if (st.stop_requested()) {
                            stopped = true;
                            break;
                        }
                        const std::size_t index = indices[step];
                        const neural::Tensor input = training.image(index);
                        float loss = 0.0F;
                        if (batch == 1) {
                            loss = net->step(input, training.label(index), opt);
                        } else {
                            loss = net->accumulate(input, training.label(index));
                            ++in_batch;
                        }
                        if (!std::isfinite(loss)) {
                            throw std::runtime_error("Loss became non-finite; learning rate is too high for this optimiser");
                        }
                        window_loss += static_cast<double>(loss);
                        ++window_count;
                        ++total_samples_;
                        in_epoch_ = step + 1;
                        if (batch > 1 && (in_batch == batch || step + 1 == per_epoch)) {
                            net->apply_accumulated(opt, in_batch);
                            in_batch = 0;
                        }
                        if (window_count == 500) publish();
                    }
                    publish();
                }
            };

            switch (cfg.optimiser) {
                case OptimiserType::SGD: {
                    ArenaSGD opt(cfg.learning_rate);
                    train_loop(opt);
                    break;
                }
                case OptimiserType::Momentum: {
                    ArenaMomentum opt(cfg.learning_rate, cfg.momentum);
                    train_loop(opt);
                    break;
                }
                case OptimiserType::RMSprop: {
                    ArenaRMSprop opt(cfg.learning_rate, cfg.momentum);
                    train_loop(opt);
                    break;
                }
                case OptimiserType::Adam: {
                    ArenaAdam opt(cfg.learning_rate, cfg.momentum);
                    train_loop(opt);
                    break;
                }
                default:
                    throw std::runtime_error("Unknown optimiser");
            }

            if (stopped && (!cfg.save_on_stop || total_samples_ == 0)) {
                set_message("Training stopped. Nothing was saved.");
                finish(TrainPhase::Cancelled);
                return;
            }

            phase_ = TrainPhase::Saving;
            set_message("Saving checkpoint...");
            net->save(cfg.output);
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                saved_ = cfg.output.string();
            }

            if (stopped) {
                set_message("Training stopped; checkpoint saved.");
                finish(TrainPhase::Cancelled);
                return;
            }

            if (testing) {
                phase_ = TrainPhase::Evaluating;
                set_message("Evaluating on the test set...");
                eval_total_ = testing->size();
                const EvalSummary ev = evaluate_detailed(
                    *net, *testing, st, [this](std::size_t done, std::size_t) { eval_done_ = done; });
                if (ev.cancelled) {
                    set_message("Evaluation cancelled; checkpoint was saved.");
                    finish(TrainPhase::Cancelled);
                    return;
                }
                accuracy_ = ev.accuracy();
                test_loss_ = ev.mean_loss();
            }
            set_message("Training complete.");
            finish(TrainPhase::Done);
        } catch (const std::exception& e) {
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                error_ = e.what();
            }
            set_message("Training failed.");
            finish(TrainPhase::Failed);
        } catch (...) {
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                error_ = "Unknown error";
            }
            set_message("Training failed.");
            finish(TrainPhase::Failed);
        }
    }

    std::atomic<TrainPhase> phase_{TrainPhase::Idle};
    std::atomic<std::size_t> epoch_{0}, epochs_{0}, in_epoch_{0}, per_epoch_{0};
    std::atomic<std::size_t> total_samples_{0}, eval_done_{0}, eval_total_{0};
    std::atomic<double> recent_loss_{0.0}, accuracy_{-1.0}, test_loss_{0.0};
    std::atomic<long long> started_ns_{0}, finished_ns_{0};
    mutable std::mutex mutex_;
    std::string message_, error_, saved_;
    std::vector<LossPoint> history_;
    std::jthread worker_;
};

// ============================================================================
// Multi-Optimiser Comparison Arena
// ============================================================================

struct ArenaMetrics {
    std::atomic<bool> enabled{false};
    std::atomic<bool> completed{false};
    std::atomic<bool> diverged{false};
    std::atomic<std::size_t> current_epoch{0};
    std::atomic<std::size_t> samples_processed{0};
    std::atomic<float> recent_loss{0.0F};
    std::atomic<float> train_acc{0.0F};
    std::atomic<float> val_acc{0.0F};
    std::atomic<float> val_loss{0.0F};
    std::atomic<double> elapsed_sec{0.0};

    std::atomic<float> live_lr{0.01F};
    std::atomic<float> live_momentum{0.9F};  // momentum / RMSprop decay / Adam beta1
    std::atomic<float> live_beta2{0.999F};
    std::atomic<float> live_eps{1e-8F};

    mutable std::mutex curve_mutex;
    std::vector<LossPoint> loss_curve;

    void add_point(double sample_idx, float val) {
        const std::lock_guard<std::mutex> lock(curve_mutex);
        loss_curve.push_back({sample_idx, val});
        decimate_history(loss_curve, kHistoryCap);
    }
};

class MultiTrainerArena {
public:
    using OptFlags = std::array<bool, kOptimiserCount>;
    using OptLrs = std::array<float, kOptimiserCount>;

    MultiTrainerArena() = default;
    ~MultiTrainerArena() { stop(); }
    MultiTrainerArena(const MultiTrainerArena&) = delete;
    MultiTrainerArena& operator=(const MultiTrainerArena&) = delete;

    [[nodiscard]] bool is_busy() const { return running_.load(); }
    [[nodiscard]] bool is_paused() const { return paused_.load(); }

    void pause() {
        if (running_.load()) paused_.store(true);
    }

    void resume() { paused_.store(false); }

    void stop() {
        paused_.store(false);
        if (coordinator_.joinable()) {
            coordinator_.request_stop();
            coordinator_.join();
        }
        running_.store(false);
    }

    // Returns false (and does nothing) if already running or nothing is enabled.
    bool start(const fs::path& data_dir,
               const std::vector<std::size_t>& arch,
               std::size_t epochs,
               std::size_t max_samples,
               std::size_t batch_size,
               const OptFlags& active_opts,
               const OptLrs& lrs) {
        if (is_busy()) return false;
        stop();  // reaps a previously finished coordinator thread

        bool any = false;
        for (const bool b : active_opts) any = any || b;
        if (!any) return false;

        epochs_.store(epochs);
        max_samples_.store(max_samples);
        batch_size_.store(std::max<std::size_t>(1, batch_size));
        samples_per_epoch_.store(0);
        clear_error();

        for (std::size_t i = 0; i < kOptimiserCount; ++i) {
            ArenaMetrics& m = metrics_[i];
            m.enabled.store(active_opts[i]);
            m.completed.store(false);
            m.diverged.store(false);
            m.current_epoch.store(0);
            m.samples_processed.store(0);
            m.recent_loss.store(0.0F);
            m.train_acc.store(0.0F);
            m.val_acc.store(0.0F);
            m.val_loss.store(0.0F);
            m.elapsed_sec.store(0.0);
            m.live_lr.store(lrs[i]);
            m.live_momentum.store(0.9F);
            m.live_beta2.store(0.999F);
            m.live_eps.store(1e-8F);
            const std::lock_guard<std::mutex> lock(m.curve_mutex);
            m.loss_curve.clear();
        }

        running_.store(true);
        paused_.store(false);

        coordinator_ = std::jthread([this, data_dir, arch, active_opts](std::stop_token st) {
            run_coordinator(st, data_dir, arch, active_opts);
        });
        return true;
    }

    ArenaMetrics& metrics(OptimiserType t) { return metrics_[static_cast<std::size_t>(t)]; }

    [[nodiscard]] std::size_t target_epochs() const { return epochs_.load(); }
    [[nodiscard]] std::size_t samples_per_epoch() const { return samples_per_epoch_.load(); }

    [[nodiscard]] std::string last_error() const {
        const std::lock_guard<std::mutex> lock(err_mutex_);
        return error_;
    }

private:
    void report_error(const std::string& msg) {
        const std::lock_guard<std::mutex> lock(err_mutex_);
        error_ = msg;
    }

    void clear_error() {
        const std::lock_guard<std::mutex> lock(err_mutex_);
        error_.clear();
    }

    // The coordinator owns the worker threads (they are local to this function), so the UI
    // thread never touches them. Workers watch the coordinator's stop token.
    void run_coordinator(std::stop_token st, fs::path data_dir, std::vector<std::size_t> arch, OptFlags enabled) {
        std::vector<std::jthread> workers;
        try {
            const DatasetFiles files = dataset_files(data_dir);
            if (!files.has_train()) {
                throw std::runtime_error("Training files not found in " + data_dir.string());
            }
            auto train = std::make_shared<neural::MNISTDataset>(files.train_images, files.train_labels);
            if (train->size() == 0) {
                throw std::runtime_error("Training dataset is empty");
            }
            std::shared_ptr<neural::MNISTDataset> test;
            if (files.has_test()) {
                test = std::make_shared<neural::MNISTDataset>(files.test_images, files.test_labels);
            }

            const std::size_t max_s = max_samples_.load();
            const std::size_t spe = (max_s == 0) ? train->size() : std::min(max_s, train->size());
            samples_per_epoch_.store(spe);

            Network master(arch, neural::WeightInit::He, kBaseSeed);

            for (std::size_t idx = 0; idx < kOptimiserCount; ++idx) {
                if (!enabled[idx]) continue;
                workers.emplace_back([this, idx, st, model = master.clone(), train, test, spe]() mutable {
                    run_worker(st, static_cast<OptimiserType>(idx), std::move(model), train, test, spe);
                });
            }
            for (std::jthread& t : workers) {
                if (t.joinable()) t.join();
            }
        } catch (const std::exception& e) {
            report_error(e.what());
        } catch (...) {
            report_error("Unknown arena error");
        }
        workers.clear();
        running_.store(false);
        paused_.store(false);
    }

    void run_worker(std::stop_token st,
                    OptimiserType type,
                    Network model,
                    std::shared_ptr<const neural::MNISTDataset> train,
                    std::shared_ptr<const neural::MNISTDataset> test,
                    std::size_t spe) {
        ArenaMetrics& met = metrics_[static_cast<std::size_t>(type)];
        try {
            ArenaSGD sgd(met.live_lr.load());
            ArenaMomentum momentum(met.live_lr.load(), met.live_momentum.load());
            ArenaRMSprop rmsprop(met.live_lr.load(), met.live_momentum.load(), met.live_eps.load());
            ArenaAdam adam(met.live_lr.load(), met.live_momentum.load(), met.live_beta2.load(), met.live_eps.load());

            // Each worker shuffles its own copy with the same seed, so every optimiser sees
            // exactly the same sample order without any shared mutable state.
            std::vector<std::size_t> order(train->size());
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::mt19937 rng(kShuffleSeed);

            const std::size_t epochs = epochs_.load();
            const std::size_t bs = batch_size_.load();
            const auto t0 = std::chrono::steady_clock::now();
            std::size_t total = 0;

            for (std::size_t epoch = 0; epoch < epochs && !st.stop_requested(); ++epoch) {
                met.current_epoch.store(epoch + 1);
                std::shuffle(order.begin(), order.end(), rng);

                double window_loss = 0.0;
                std::size_t window_hits = 0;
                std::size_t correct_train = 0;
                std::size_t seen = 0;
                std::size_t in_batch = 0;

                auto flush = [&]() {
                    if (window_hits == 0) return;
                    const auto mean = static_cast<float>(window_loss / static_cast<double>(window_hits));
                    met.recent_loss.store(mean);
                    met.add_point(static_cast<double>(total), mean);
                    window_loss = 0.0;
                    window_hits = 0;
                    met.elapsed_sec.store(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                };

                for (std::size_t step = 0; step < spe && !st.stop_requested(); ++step) {
                    while (paused_.load() && !st.stop_requested()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                    if (st.stop_requested()) break;

                    switch (type) {
                        case OptimiserType::SGD:
                            sgd.set_lr(met.live_lr.load());
                            break;
                        case OptimiserType::Momentum:
                            momentum.set_lr(met.live_lr.load());
                            momentum.set_momentum(met.live_momentum.load());
                            break;
                        case OptimiserType::RMSprop:
                            rmsprop.set_lr(met.live_lr.load());
                            rmsprop.set_decay(met.live_momentum.load());
                            rmsprop.set_eps(met.live_eps.load());
                            break;
                        case OptimiserType::Adam:
                            adam.set_lr(met.live_lr.load());
                            adam.set_b1(met.live_momentum.load());
                            adam.set_b2(met.live_beta2.load());
                            adam.set_eps(met.live_eps.load());
                            break;
                        default: break;
                    }

                    const std::size_t sample_idx = order[step];
                    const neural::Tensor img = train->image(sample_idx);
                    const std::size_t label = train->label(sample_idx);

                    float loss = 0.0F;
                    bool hit = false;
                    // batch size 1: update every sample; otherwise average gradients over the
                    // batch (a final partial batch is applied at the end of the epoch).
                    auto train_sample = [&](auto& opt) -> float {
                        if (bs == 1) return model.step(img, label, opt, &hit);
                        const float l = model.accumulate(img, label, &hit);
                        ++in_batch;
                        if (in_batch == bs || step + 1 == spe) {
                            model.apply_accumulated(opt, in_batch);
                            in_batch = 0;
                        }
                        return l;
                    };
                    switch (type) {
                        case OptimiserType::SGD:      loss = train_sample(sgd); break;
                        case OptimiserType::Momentum: loss = train_sample(momentum); break;
                        case OptimiserType::RMSprop:  loss = train_sample(rmsprop); break;
                        case OptimiserType::Adam:     loss = train_sample(adam); break;
                        default: break;
                    }

                    if (!std::isfinite(loss)) {
                        met.diverged.store(true);
                        met.elapsed_sec.store(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                        report_error(std::string(optimiser_name(type)) +
                                     " diverged (non-finite loss). Lower its learning rate and restart.");
                        return;
                    }

                    window_loss += static_cast<double>(loss);
                    ++window_hits;
                    ++seen;
                    ++total;
                    if (hit) ++correct_train;
                    met.samples_processed.store(total);

                    if (window_hits == 250) flush();
                }
                flush();

                if (seen > 0) {
                    met.train_acc.store(100.0F * static_cast<float>(correct_train) / static_cast<float>(seen));
                }

                if (test && test->size() > 0 && !st.stop_requested()) {
                    std::size_t correct_val = 0;
                    double val_loss_acc = 0.0;
                    bool aborted = false;
                    for (std::size_t v = 0; v < test->size(); ++v) {
                        if ((v & 0xFFU) == 0U && st.stop_requested()) {
                            aborted = true;
                            break;
                        }
                        const neural::Tensor v_img = test->image(v);
                        const std::size_t v_lbl = test->label(v);
                        const neural::Tensor v_logits = model.forward(v_img);
                        const neural::LossResult lr = neural::softmax_cross_entropy(v_logits, v_lbl);
                        val_loss_acc += static_cast<double>(lr.loss);
                        const auto vd = v_logits.data();
                        const auto best = static_cast<std::size_t>(std::distance(vd.begin(), std::max_element(vd.begin(), vd.end())));
                        if (best == v_lbl) ++correct_val;
                    }
                    if (!aborted) {
                        const auto n = static_cast<double>(test->size());
                        met.val_acc.store(static_cast<float>(100.0 * static_cast<double>(correct_val) / n));
                        met.val_loss.store(static_cast<float>(val_loss_acc / n));
                    }
                }
            }

            met.elapsed_sec.store(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            if (!st.stop_requested()) met.completed.store(true);
        } catch (const std::exception& e) {
            report_error(std::string(optimiser_name(type)) + ": " + e.what());
        } catch (...) {
            report_error(std::string(optimiser_name(type)) + ": unknown error");
        }
    }

    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<std::size_t> epochs_{5};
    std::atomic<std::size_t> max_samples_{0};
    std::atomic<std::size_t> batch_size_{1};
    std::atomic<std::size_t> samples_per_epoch_{0};
    std::array<ArenaMetrics, kOptimiserCount> metrics_;
    mutable std::mutex err_mutex_;
    std::string error_;
    std::jthread coordinator_;  // declared last: destroyed (and joined) first
};

// ============================================================================
// UI System
// ============================================================================

struct Ui {
    Font font{};
    bool custom_font = false;
    Rectangle clip{0, 0, 100000, 100000};
    bool blocked = false;
    int focused = -1;
    std::size_t cursor = 0;
    int cursor_kind = MOUSE_CURSOR_DEFAULT;
    [[nodiscard]] float spacing(float size) const { return custom_font ? 0.4F : size / 10.0F; }
};

void draw_text(const Ui& ui, const std::string& s, float x, float y, float size, Color c) {
    DrawTextEx(ui.font, s.c_str(), vec(x, y), size, ui.spacing(size), c);
}

float measure(const Ui& ui, const std::string& s, float size) {
    return MeasureTextEx(ui.font, s.c_str(), size, ui.spacing(size)).x;
}

std::string elide_middle(const Ui& ui, const std::string& s, float size, float max_width) {
    if (measure(ui, s, size) <= max_width) return s;
    std::size_t keep = s.size();
    while (keep > 4) {
        keep -= 2;
        const std::string t = s.substr(0, keep / 2) + "..." + s.substr(s.size() - keep / 2);
        if (measure(ui, t, size) <= max_width) return t;
    }
    return "...";
}

float draw_wrapped(const Ui& ui, const std::string& text, float x, float y, float w, float size, Color c) {
    const float line_height = size * 1.35F;
    std::istringstream words(text);
    std::string word, line;
    float cy = y;
    auto flush = [&]() {
        if (!line.empty()) {
            draw_text(ui, line, x, cy, size, c);
            cy += line_height;
            line.clear();
        }
    };
    while (words >> word) {
        while (measure(ui, word, size) > w && word.size() > 1) {
            std::size_t n = word.size();
            while (n > 1 && measure(ui, word.substr(0, n), size) > w) --n;
            flush();
            draw_text(ui, word.substr(0, n), x, cy, size, c);
            cy += line_height;
            word = word.substr(n);
        }
        const std::string trial = line.empty() ? word : line + " " + word;
        if (!line.empty() && measure(ui, trial, size) > w) {
            flush();
            line = word;
        } else {
            line = trial;
        }
    }
    flush();
    return cy - y;
}

void fill_rounded(Rectangle r, float radius, Color c) {
    const float roundness = std::min(1.0F, 2.0F * radius / std::max(1.0F, std::min(r.width, r.height)));
    DrawRectangleRounded(r, roundness, 8, c);
}

void fill_rounded_border(Rectangle r, float radius, Color border, Color fill) {
    fill_rounded(r, radius, border);
    fill_rounded(rect(r.x + 1, r.y + 1, r.width - 2, r.height - 2), radius - 1, fill);
}

bool hovered(const Ui& ui, Rectangle r) {
    if (ui.blocked) return false;
    const Vector2 m = GetMousePosition();
    return CheckCollisionPointRec(m, ui.clip) && CheckCollisionPointRec(m, r);
}

enum class ButtonStyle { Normal, Primary, Danger, Selected };

bool button(Ui& ui, Rectangle r, const std::string& label, bool enabled = true, ButtonStyle style = ButtonStyle::Normal) {
    const bool hov = enabled && hovered(ui, r);
    Color fill = theme::panel_alt;
    Color fg = theme::text;
    switch (style) {
        case ButtonStyle::Primary: fill = mix(theme::accent_dim, theme::accent, 0.35F); break;
        case ButtonStyle::Danger: fill = Color{110, 40, 44, 255}; break;
        case ButtonStyle::Selected: fill = theme::accent_dim; break;
        default: break;
    }
    if (!enabled) {
        fill = theme::panel;
        fg = theme::faint;
    } else if (hov) {
        fill = mix(fill, theme::accent, IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? 0.5F : 0.25F);
        ui.cursor_kind = MOUSE_CURSOR_POINTING_HAND;
    }
    fill_rounded_border(r, 6, style == ButtonStyle::Selected ? theme::accent : theme::border, fill);
    const float size = 15;
    const float tw = measure(ui, label, size);
    draw_text(ui, label, r.x + (r.width - tw) / 2, r.y + (r.height - size) / 2, size, fg);
    return hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

bool checkbox(Ui& ui, Rectangle r, const std::string& label, bool& value, bool enabled = true) {
    const bool hov = enabled && hovered(ui, r);
    const Rectangle box = rect(r.x, r.y + (r.height - 20) / 2, 20, 20);
    fill_rounded_border(box, 4, hov ? theme::accent : theme::border, theme::panel_alt);
    if (value) fill_rounded(rect(box.x + 5, box.y + 5, 10, 10), 2, enabled ? theme::accent : theme::faint);
    draw_text(ui, label, r.x + 30, r.y + (r.height - 16) / 2, 16, enabled ? theme::text : theme::faint);
    if (hov) ui.cursor_kind = MOUSE_CURSOR_POINTING_HAND;
    if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        value = !value;
        return true;
    }
    return false;
}

void progress_bar(const Ui& ui, Rectangle r, double fraction, Color color, const std::string& label = "") {
    fraction = std::clamp(fraction, 0.0, 1.0);
    fill_rounded(r, r.height / 2, theme::panel_alt);
    if (fraction > 0.0) {
        const float w = std::max(r.height, static_cast<float>(r.width * fraction));
        fill_rounded(rect(r.x, r.y, w, r.height), r.height / 2, color);
    }
    if (!label.empty()) {
        const float tw = measure(ui, label, 14);
        draw_text(ui, label, r.x + (r.width - tw) / 2, r.y + (r.height - 14) / 2, 14, theme::text);
    }
}

bool key_fire(int key) {
    static double next[512] = {};
    if (key < 0 || key >= 512) return false;
    if (IsKeyPressed(key)) {
        next[key] = GetTime() + 0.4;
        return true;
    }
    if (IsKeyDown(key) && GetTime() >= next[key]) {
        next[key] = GetTime() + 0.035;
        return true;
    }
    return false;
}

bool text_field(Ui& ui, int id, Rectangle r, std::string& value, const char* placeholder, bool enabled = true) {
    const bool hov = enabled && hovered(ui, r);
    bool committed = false;
    if (hov) ui.cursor_kind = MOUSE_CURSOR_IBEAM;
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        if (hov) {
            ui.focused = id;
            ui.cursor = value.size();
        } else if (ui.focused == id) {
            ui.focused = -1;
        }
    }
    if (!enabled && ui.focused == id) ui.focused = -1;
    const bool focus = enabled && ui.focused == id;
    if (focus) {
        ui.cursor = std::min(ui.cursor, value.size());
        for (int ch = GetCharPressed(); ch > 0; ch = GetCharPressed()) {
            if (ch >= 32 && ch <= 126) {
                value.insert(value.begin() + static_cast<std::ptrdiff_t>(ui.cursor), static_cast<char>(ch));
                ++ui.cursor;
            }
        }
        const bool ctrl = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL) ||
                          IsKeyDown(KEY_LEFT_SUPER) || IsKeyDown(KEY_RIGHT_SUPER);
        if (ctrl && IsKeyPressed(KEY_V)) {
            const char* clip = GetClipboardText();
            if (clip != nullptr) {
                std::string paste;
                for (const char* p = clip; *p != '\0'; ++p) {
                    const auto c = static_cast<unsigned char>(*p);
                    if (c >= 32 && c <= 126) paste.push_back(static_cast<char>(c));
                }
                value.insert(ui.cursor, paste);
                ui.cursor += paste.size();
            }
        }
        if (ctrl && IsKeyPressed(KEY_A)) {
            value.clear();
            ui.cursor = 0;
        }
        if (key_fire(KEY_BACKSPACE) && ui.cursor > 0) {
            value.erase(ui.cursor - 1, 1);
            --ui.cursor;
        }
        if (key_fire(KEY_DELETE) && ui.cursor < value.size()) value.erase(ui.cursor, 1);
        if (key_fire(KEY_LEFT) && ui.cursor > 0) --ui.cursor;
        if (key_fire(KEY_RIGHT) && ui.cursor < value.size()) ++ui.cursor;
        if (IsKeyPressed(KEY_HOME)) ui.cursor = 0;
        if (IsKeyPressed(KEY_END)) ui.cursor = value.size();
        if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) {
            committed = true;
            ui.focused = -1;
        }
        if (IsKeyPressed(KEY_ESCAPE)) ui.focused = -1;
    }

    Color border = focus ? theme::accent : (hov ? mix(theme::border, theme::accent, 0.4F) : theme::border);
    fill_rounded_border(r, 6, border, enabled ? theme::bg : theme::panel);

    const float size = 15;
    const float pad = 10;
    const float inner_w = r.width - 2 * pad;
    float scroll = 0;
    if (focus) {
        const float cx = measure(ui, value.substr(0, std::min(ui.cursor, value.size())), size);
        if (cx > inner_w) scroll = cx - inner_w;
    }
    BeginScissorMode(static_cast<int>(r.x + pad), static_cast<int>(r.y),
                     static_cast<int>(std::max(1.0F, inner_w)), static_cast<int>(r.height));
    if (value.empty() && !focus) {
        draw_text(ui, placeholder, r.x + pad, r.y + (r.height - size) / 2, size, theme::faint);
    } else {
        draw_text(ui, value, r.x + pad - scroll, r.y + (r.height - size) / 2, size,
                  enabled ? theme::text : theme::faint);
        if (focus && std::fmod(GetTime(), 1.0) < 0.55) {
            const float cx = measure(ui, value.substr(0, std::min(ui.cursor, value.size())), size);
            DrawRectangleRec(rect(r.x + pad - scroll + cx, r.y + 7, 1.5F, r.height - 14), theme::accent);
        }
    }
    EndScissorMode();
    return committed;
}

float labelled_field(Ui& ui, int id, float x, float y, float w, const char* label,
                     std::string& value, const char* placeholder, bool enabled = true) {
    draw_text(ui, label, x, y, 14, theme::muted);
    text_field(ui, id, rect(x, y + 18, w, 34), value, placeholder, enabled);
    return y + 18 + 34 + 14;
}

void draw_panel(const Ui& ui, Rectangle r, const std::string& title) {
    fill_rounded_border(r, 10, theme::border, theme::panel);
    if (!title.empty()) draw_text(ui, title, r.x + 18, r.y + 14, 18, theme::text);
}

class GrayTexture {
public:
    GrayTexture(int width, int height, const std::vector<std::uint8_t>& pixels, bool smooth)
        : width_(width), height_(height) {
        if (pixels.size() < static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
            throw std::invalid_argument("GrayTexture: buffer too small");
        }
        Image img{};
        img.data = const_cast<std::uint8_t*>(pixels.data());
        img.width = width;
        img.height = height;
        img.mipmaps = 1;
        img.format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE;
        tex_ = LoadTextureFromImage(img);
        SetTextureFilter(tex_, smooth ? TEXTURE_FILTER_BILINEAR : TEXTURE_FILTER_POINT);
    }
    ~GrayTexture() { UnloadTexture(tex_); }
    GrayTexture(const GrayTexture&) = delete;
    GrayTexture& operator=(const GrayTexture&) = delete;

    void update(const std::vector<std::uint8_t>& pixels) {
        if (pixels.size() >= static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_)) {
            UpdateTexture(tex_, pixels.data());
        }
    }
    [[nodiscard]] const Texture2D& texture() const { return tex_; }
    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }

private:
    int width_ = 0, height_ = 0;
    Texture2D tex_{};
};

// ============================================================================
// Charts & Confusion Matrix
// ============================================================================

void draw_loss_chart(const Ui& ui, Rectangle r, const std::vector<LossPoint>& pts, double x_total,
                     std::size_t per_epoch, std::size_t epochs, bool log_scale) {
    fill_rounded_border(r, 8, theme::border, theme::bg);
    if (pts.size() < 2 || !(x_total > 0.0)) {
        draw_text(ui, "Loss curve appears once training has run", r.x + 12.0F, r.y + r.height / 2.0F - 8.0F, 14, theme::faint);
        return;
    }

    const float left = 46.0F, right = 12.0F, top = 12.0F, bottom = 24.0F;
    const Rectangle plot = rect(r.x + left, r.y + top, r.width - left - right, r.height - top - bottom);
    if (plot.width < 40.0F || plot.height < 30.0F) return;

    double lo_v = std::numeric_limits<double>::max();
    double hi_v = 0.0;
    for (const LossPoint& p : pts) {
        const double v = static_cast<double>(p.loss);
        if (!std::isfinite(v)) continue;
        lo_v = std::min(lo_v, v);
        hi_v = std::max(hi_v, v);
    }
    if (lo_v > hi_v) return;

    double y_lo = 0.0;
    double y_hi = hi_v > 0.0 ? hi_v * 1.08 : 1.0;
    if (log_scale) {
        lo_v = std::max(lo_v, 1e-4);
        hi_v = std::max(hi_v, lo_v * 1.5);
        y_lo = lo_v * 0.8;
        y_hi = hi_v * 1.25;
    }

    auto map_y = [&](double v) -> float {
        double t = 0.0;
        if (log_scale) {
            v = std::max(v, y_lo);
            t = (std::log(v) - std::log(y_lo)) / (std::log(y_hi) - std::log(y_lo));
        } else {
            t = (v - y_lo) / (y_hi - y_lo);
        }
        t = std::clamp(t, 0.0, 1.0);
        return plot.y + plot.height * static_cast<float>(1.0 - t);
    };
    auto map_x = [&](double s) -> float {
        return plot.x + plot.width * static_cast<float>(std::clamp(s / x_total, 0.0, 1.0));
    };

    const Color grid = Fade(theme::border, 0.7F);

    std::vector<double> y_ticks;
    if (log_scale) {
        const int k0 = static_cast<int>(std::floor(std::log10(y_lo)));
        const int k1 = static_cast<int>(std::ceil(std::log10(y_hi)));
        for (int k = k0; k <= k1; ++k) {
            for (const double m : {1.0, 2.0, 5.0}) {
                const double v = m * std::pow(10.0, k);
                if (v >= y_lo && v <= y_hi) y_ticks.push_back(v);
            }
        }
    } else {
        const double step = nice_step(y_hi - y_lo, 4);
        for (double v = std::ceil(y_lo / step) * step; v <= y_hi + 1e-12; v += step) y_ticks.push_back(v);
    }
    float last_label_y = -1000.0F;
    for (const double v : y_ticks) {
        const float y = map_y(v);
        DrawLineEx(vec(plot.x, y), vec(plot.x + plot.width, y), 1.0F, grid);
        if (std::fabs(y - last_label_y) >= 14.0F) {
            const std::string lab = tick_label(v);
            draw_text(ui, lab, plot.x - 6.0F - measure(ui, lab, 11), y - 6.0F, 11, theme::faint);
            last_label_y = y;
        }
    }

    if (per_epoch > 0 && epochs > 0) {
        const double step_e = std::max(1.0, nice_step(static_cast<double>(epochs), 5));
        for (double e = 0.0; e <= static_cast<double>(epochs) + 1e-9; e += step_e) {
            const float x = map_x(e * static_cast<double>(per_epoch));
            DrawLineEx(vec(x, plot.y), vec(x, plot.y + plot.height), 1.0F, grid);
            const std::string lab = tick_label(e);
            draw_text(ui, lab, x - measure(ui, lab, 11) / 2.0F, plot.y + plot.height + 5.0F, 11, theme::faint);
        }
    }
    draw_text(ui, "epoch", r.x + 6.0F, r.y + r.height - 16.0F, 11, theme::faint);

    std::vector<double> ema(pts.size());
    ema[0] = static_cast<double>(pts[0].loss);
    constexpr double alpha = 0.15;
    for (std::size_t i = 1; i < pts.size(); ++i) {
        ema[i] = ema[i - 1] + alpha * (static_cast<double>(pts[i].loss) - ema[i - 1]);
    }

    BeginScissorMode(static_cast<int>(plot.x), static_cast<int>(plot.y) - 2,
                     static_cast<int>(plot.width) + 2, static_cast<int>(plot.height) + 4);
    for (std::size_t i = 1; i < pts.size(); ++i) {
        DrawLineEx(vec(map_x(pts[i - 1].samples), map_y(static_cast<double>(pts[i - 1].loss))),
                   vec(map_x(pts[i].samples), map_y(static_cast<double>(pts[i].loss))),
                   1.0F, Fade(theme::accent, 0.30F));
    }
    for (std::size_t i = 1; i < pts.size(); ++i) {
        DrawLineEx(vec(map_x(pts[i - 1].samples), map_y(ema[i - 1])),
                   vec(map_x(pts[i].samples), map_y(ema[i])), 2.5F, theme::accent);
    }
    DrawCircleV(vec(map_x(pts.back().samples), map_y(ema.back())), 3.5F, theme::text);
    EndScissorMode();

    const std::string now = "smoothed " + fixed(ema.back(), 4);
    draw_text(ui, now, plot.x + plot.width - measure(ui, now, 12) - 4.0F, plot.y + 2.0F, 12, theme::muted);
}

void draw_confusion(const Ui& ui, Rectangle area, const EvalSummary& s) {
    const float label_m = 14.0F;
    const float cell = (std::min(area.width, area.height) - label_m) / static_cast<float>(kMnistClasses);
    if (cell < 4.0F) return;
    const float ox = area.x + label_m;
    const float oy = area.y + label_m;

    for (std::size_t i = 0; i < kMnistClasses; ++i) {
        const std::string lab = std::to_string(i);
        draw_text(ui, lab, ox + static_cast<float>(i) * cell + (cell - 7.0F) / 2.0F, area.y, 10, theme::faint);
        draw_text(ui, lab, area.x + 3.0F, oy + static_cast<float>(i) * cell + (cell - 10.0F) / 2.0F, 10, theme::faint);
    }
    for (std::size_t row = 0; row < kMnistClasses; ++row) {
        std::size_t row_total = 0;
        for (std::size_t c = 0; c < kMnistClasses; ++c) row_total += s.confusion[row][c];
        for (std::size_t c = 0; c < kMnistClasses; ++c) {
            const double t = row_total ? static_cast<double>(s.confusion[row][c]) / static_cast<double>(row_total) : 0.0;
            const float intensity = static_cast<float>(std::sqrt(t));
            const Color target = (row == c) ? theme::good : theme::bad;
            DrawRectangleRec(rect(ox + static_cast<float>(c) * cell, oy + static_cast<float>(row) * cell, cell - 1.0F, cell - 1.0F),
                             mix(theme::panel_alt, target, intensity));
        }
    }

    const Rectangle matrix = rect(ox, oy, cell * static_cast<float>(kMnistClasses), cell * static_cast<float>(kMnistClasses));
    if (hovered(ui, matrix)) {
        const Vector2 m = GetMousePosition();
        const auto c = static_cast<std::size_t>(std::clamp((m.x - ox) / cell, 0.0F, 9.0F));
        const auto row = static_cast<std::size_t>(std::clamp((m.y - oy) / cell, 0.0F, 9.0F));
        const std::string tip = "true " + std::to_string(row) + " -> predicted " + std::to_string(c) +
                                ": " + std::to_string(s.confusion[row][c]);
        const float w = measure(ui, tip, 13) + 16.0F;
        const float x = std::min(m.x + 14.0F, static_cast<float>(GetScreenWidth()) - w - 4.0F);
        fill_rounded_border(rect(x, m.y + 14.0F, w, 24.0F), 5, theme::border, theme::bg);
        draw_text(ui, tip, x + 8.0F, m.y + 19.0F, 13, theme::text);
    }
}

void draw_arena_plot(const Ui& ui, Rectangle r, MultiTrainerArena& arena, bool log_scale) {
    fill_rounded_border(r, 8, theme::border, theme::bg);
    draw_text(ui, std::string("LIVE MULTI-OPTIMISER COMPARATIVE LOSS") + (log_scale ? " (LOG)" : ""),
              r.x + 14.0F, r.y + 10.0F, 13, theme::muted);

    const float left = 50.0F, right = 20.0F, top = 36.0F, bottom = 30.0F;
    const Rectangle plot = rect(r.x + left, r.y + top, r.width - left - right, r.height - top - bottom);
    if (plot.width < 40.0F || plot.height < 30.0F) return;

    double max_x = 1000.0;
    double min_v = std::numeric_limits<double>::max();  // smallest positive loss (for log scale)
    double max_v = 0.0;

    for (std::size_t i = 0; i < kOptimiserCount; ++i) {
        auto& met = arena.metrics(static_cast<OptimiserType>(i));
        if (!met.enabled.load()) continue;
        const std::lock_guard<std::mutex> lock(met.curve_mutex);
        if (!met.loss_curve.empty()) {
            max_x = std::max(max_x, met.loss_curve.back().samples);
            for (const auto& pt : met.loss_curve) {
                if (!std::isfinite(pt.loss)) continue;
                const double v = static_cast<double>(pt.loss);
                max_v = std::max(max_v, v);
                if (v > 0.0) min_v = std::min(min_v, v);
            }
        }
    }

    double y_lo = 0.0;
    double y_hi = 1.0;
    if (log_scale) {
        if (min_v > max_v) {  // no data yet
            min_v = 0.01;
            max_v = 2.5;
        }
        min_v = std::max(min_v, 1e-4);
        max_v = std::max(max_v, min_v * 1.5);
        y_lo = min_v * 0.8;
        y_hi = max_v * 1.25;
    } else {
        y_hi = std::min(std::max(2.5, max_v) * 1.05, 10.0);
    }

    auto to_x = [&](double s) {
        return plot.x + plot.width * static_cast<float>(std::clamp(s / max_x, 0.0, 1.0));
    };
    auto to_y = [&](double l) -> float {
        double t = 0.0;
        if (log_scale) {
            l = std::max(l, y_lo);
            t = (std::log(l) - std::log(y_lo)) / (std::log(y_hi) - std::log(y_lo));
        } else {
            t = (l - y_lo) / (y_hi - y_lo);
        }
        t = std::clamp(t, 0.0, 1.0);
        return plot.y + plot.height * static_cast<float>(1.0 - t);
    };

    std::vector<double> y_ticks;
    if (log_scale) {
        const int k0 = static_cast<int>(std::floor(std::log10(y_lo)));
        const int k1 = static_cast<int>(std::ceil(std::log10(y_hi)));
        for (int k = k0; k <= k1; ++k) {
            for (const double m : {1.0, 2.0, 5.0}) {
                const double v = m * std::pow(10.0, k);
                if (v >= y_lo && v <= y_hi) y_ticks.push_back(v);
            }
        }
    } else {
        const double step_y = nice_step(y_hi - y_lo, 4);
        for (double v = 0.0; v <= y_hi + 1e-9; v += step_y) y_ticks.push_back(v);
    }
    float last_label_y = -1000.0F;
    for (const double v : y_ticks) {
        const float y = to_y(v);
        DrawLineEx(vec(plot.x, y), vec(plot.x + plot.width, y), 1.0F, theme::panel_alt);
        if (std::fabs(y - last_label_y) >= 14.0F) {
            const std::string lab = log_scale ? tick_label(v) : fixed(v, 2);
            draw_text(ui, lab, plot.x - 6.0F - measure(ui, lab, 11), y - 6.0F, 11, theme::faint);
            last_label_y = y;
        }
    }

    const double step_x = nice_step(max_x, 5);
    for (double v = 0.0; v <= max_x; v += step_x) {
        const float x = to_x(v);
        DrawLineEx(vec(x, plot.y), vec(x, plot.y + plot.height), 1.0F, theme::panel_alt);
        const std::string s = tick_label(v / 1000.0) + "k";
        draw_text(ui, s, x - 10.0F, plot.y + plot.height + 6.0F, 11, theme::faint);
    }

    BeginScissorMode(static_cast<int>(plot.x), static_cast<int>(plot.y) - 2,
                     static_cast<int>(plot.width) + 4, static_cast<int>(plot.height) + 4);

    for (std::size_t i = 0; i < kOptimiserCount; ++i) {
        auto& met = arena.metrics(static_cast<OptimiserType>(i));
        if (!met.enabled.load()) continue;

        const std::lock_guard<std::mutex> lock(met.curve_mutex);
        const auto& pts = met.loss_curve;
        if (pts.size() < 2) continue;

        const Color col = optimiser_color(static_cast<OptimiserType>(i));
        for (std::size_t k = 1; k < pts.size(); ++k) {
            DrawLineEx(vec(to_x(pts[k - 1].samples), to_y(static_cast<double>(pts[k - 1].loss))),
                       vec(to_x(pts[k].samples), to_y(static_cast<double>(pts[k].loss))),
                       2.0F, col);
        }
        DrawCircleV(vec(to_x(pts.back().samples), to_y(static_cast<double>(pts.back().loss))), 4.0F, col);
    }
    EndScissorMode();

    float leg_x = plot.x + plot.width - 280.0F;
    for (std::size_t i = 0; i < kOptimiserCount; ++i) {
        const auto t = static_cast<OptimiserType>(i);
        auto& met = arena.metrics(t);
        if (!met.enabled.load()) continue;
        DrawCircle(static_cast<int>(leg_x), static_cast<int>(r.y + 16.0F), 4.0F, optimiser_color(t));
        draw_text(ui, optimiser_name(t), leg_x + 10.0F, r.y + 10.0F, 12, theme::text);
        leg_x += 70.0F;
    }
}

// ============================================================================
// App State, Pages & Navigation
// ============================================================================

enum class Page { Models, Predict, Dataset, Train, Arena };
enum FieldId {
    kFieldRoot = 1, kFieldFilter, kFieldContinue, kFieldArch, kFieldEpochs, kFieldLr,
    kFieldMax, kFieldBatch, kFieldMomentum, kFieldDataset, kFieldOutput, kFieldDataset2, kFieldGoto,
    kFieldArenaData, kFieldArenaArch, kFieldArenaEpochs, kFieldArenaMax, kFieldArenaBatch
};

struct App {
    Ui ui;
    Page page = Page::Arena;
    fs::path project_root;

    std::string root_text, filter_text;
    ScanResult scan;
    fs::path selected_path;
    float list_scroll = 0;
    Status models_status;

    std::optional<Network> network;
    fs::path loaded_path;

    std::vector<std::uint8_t> canvas = std::vector<std::uint8_t>(static_cast<std::size_t>(kCanvas * kCanvas), 0);
    std::unique_ptr<GrayTexture> canvas_texture;
    bool canvas_dirty = true;
    bool drawing = false;
    Vector2 last_point{0, 0};
    std::array<float, kMnistInputs> preview{};
    bool has_preview = false;
    std::optional<Prediction> prediction;
    std::string predict_error;

    Trainer trainer;
    bool continue_mode = false;
    std::string continue_text;
    std::string arch_text = "784 512 256 128 64 10";
    neural::WeightInit init = neural::WeightInit::He;
    OptimiserType train_opt = OptimiserType::SGD;
    std::string epochs_text = "5", lr_text = "0.01", max_text = "0", momentum_text = "0.9", batch_text = "1";
    std::string dataset_text, output_text;
    bool evaluate_test = true, save_on_stop = true, log_chart = false;
    Status train_form_status;
    bool was_running = false;

    bool ds_use_test = true, ds_auto_try = true;
    std::shared_ptr<const neural::MNISTDataset> ds_train, ds_test;
    Status ds_status;
    std::size_t grid_start = 0, selected_sample = 0;
    std::string goto_text;
    std::vector<std::uint8_t> grid_pixels;
    std::unique_ptr<GrayTexture> grid_tex;
    bool grid_dirty = true;
    std::vector<std::uint8_t> sample_pixels = std::vector<std::uint8_t>(kMnistInputs, 0);
    std::unique_ptr<GrayTexture> sample_tex;
    bool sample_dirty = true;
    std::size_t sample_label = 0;
    std::optional<Prediction> sample_pred;
    std::string sample_error;
    Evaluator evaluator;
    EvalSummary eval;
    bool eval_valid = false;
    std::string eval_error;

    MultiTrainerArena arena;
    std::string arena_data_text;
    std::string arena_arch_text = "784 512 256 128 64 10";
    std::string arena_epochs_text = "5";
    std::string arena_max_text = "0";
    std::string arena_batch_text = "1";
    bool arena_log = true;  // log-scale loss axis in the arena plot
    std::array<bool, kOptimiserCount> arena_enabled = {true, true, true, true};
    std::array<float, kOptimiserCount> arena_lrs = kDefaultLrs;
    Status arena_status;

    int export_request = 0;  // 0 = none, 1 = arena, 2 = single train
    Status export_status;    // result of the last single-train export

    bool modal_open = false;
    std::string modal_text;
    std::optional<TrainConfig> pending;
};

const ModelEntry* find_entry(const App& app, const fs::path& path) {
    for (const ModelEntry& e : app.scan.entries) {
        if (e.path == path) return &e;
    }
    return nullptr;
}

void run_scan(App& app) {
    app.scan = scan_models(fs::path(trim(app.root_text)));
    if (!app.scan.message.empty()) {
        app.models_status = {app.scan.message, Tone::Bad};
    } else if (app.scan.entries.empty()) {
        app.models_status = {"No NeuralEngine checkpoints found.", Tone::Warn};
    } else {
        std::string s = "Found " + std::to_string(app.scan.entries.size()) + " checkpoint(s)";
        app.models_status = {s, Tone::Good};
    }
    if (!app.selected_path.empty() && find_entry(app, app.selected_path) == nullptr) {
        app.selected_path.clear();
    }
}

void request_prediction(App& app) {
    try {
        const Preprocessed pre = preprocess_digit(app.canvas, kCanvas);
        app.has_preview = !pre.empty;
        app.preview = pre.pixels;
        app.predict_error.clear();
        if (pre.empty || !app.network) {
            app.prediction.reset();
            return;
        }
        app.prediction = predict(*app.network, to_tensor(pre.pixels));
    } catch (const std::exception& e) {
        app.prediction.reset();
        app.predict_error = e.what();
    }
}

void load_model(App& app, const fs::path& path) {
    try {
        Network net = load_checkpoint(path);
        if (!net.is_mnist_compatible()) {
            throw std::runtime_error("Architecture is valid, but digit inference requires 784-in / 10-out.");
        }
        app.network.emplace(std::move(net));
        app.loaded_path = path;
        app.models_status = {"Loaded " + path.filename().string(), Tone::Good};
        app.canvas_dirty = true;
        app.sample_dirty = true;
        app.eval_valid = false;
        app.eval_error.clear();
    } catch (const std::exception& e) {
        app.models_status = {"Failed to load " + path.filename().string() + ": " + e.what(), Tone::Bad};
    }
}

std::optional<TrainConfig> build_config(const App& app, std::string& error) {
    TrainConfig cfg;
    try {
        if (!parse_size(app.epochs_text, cfg.epochs) || cfg.epochs == 0 || cfg.epochs > 10000) {
            throw std::runtime_error("Epochs must be an integer between 1 and 10000");
        }
        if (!parse_float(app.lr_text, cfg.learning_rate) || !std::isfinite(cfg.learning_rate) ||
            cfg.learning_rate <= 0.0F || cfg.learning_rate > 10.0F) {
            throw std::runtime_error("Learning rate must be a positive number no larger than 10");
        }
        if (!parse_size(app.max_text, cfg.max_samples)) {
            throw std::runtime_error("Max samples must be a valid integer (0 = all)");
        }
        if (!parse_size(app.batch_text, cfg.batch_size) || cfg.batch_size == 0 || cfg.batch_size > kMaxBatchSize) {
            throw std::runtime_error("Batch size must be an integer between 1 and " + std::to_string(kMaxBatchSize));
        }
        cfg.optimiser = app.train_opt;
        if (cfg.optimiser != OptimiserType::SGD) {
            if (!parse_float(app.momentum_text, cfg.momentum) || !(cfg.momentum >= 0.0F && cfg.momentum < 1.0F)) {
                throw std::runtime_error("Momentum / decay / beta1 must be in the range [0, 1)");
            }
        }
        cfg.dataset_dir = fs::path(trim(app.dataset_text));
        const DatasetFiles files = dataset_files(cfg.dataset_dir);
        if (!files.has_train()) throw std::runtime_error("Missing training dataset files in directory");
        cfg.evaluate_test = app.evaluate_test && files.has_test();

        if (app.continue_mode) {
            const fs::path src(trim(app.continue_text));
            std::ifstream f(src);
            if (!f) throw std::runtime_error("Cannot open checkpoint to resume");
            cfg.arch = read_checkpoint_header(f);
            cfg.continue_from = src;
        } else {
            cfg.arch = parse_architecture(app.arch_text);
            cfg.init = app.init;
        }
        if (cfg.arch.front() != kMnistInputs || cfg.arch.back() != kMnistClasses) {
            throw std::runtime_error("Architecture must start with 784 inputs and end with 10 outputs");
        }
        std::string out = trim(app.output_text);
        if (out.empty()) throw std::runtime_error("Target output file cannot be empty");
        cfg.output = fs::path(out);
        if (!cfg.output.has_extension()) cfg.output += ".txt";
        cfg.save_on_stop = app.save_on_stop;
    } catch (const std::exception& e) {
        error = e.what();
        return std::nullopt;
    }
    return cfg;
}

void begin_training(App& app) {
    std::string error;
    std::optional<TrainConfig> cfg = build_config(app, error);
    if (!cfg) {
        app.train_form_status = {error, Tone::Bad};
        return;
    }
    std::error_code ec;
    if (fs::exists(cfg->output, ec)) {
        app.modal_text = "Target file already exists. Overwrite when complete?";
        app.pending = std::move(cfg);
        app.modal_open = true;
        app.ui.focused = -1;
        return;
    }
    app.train_form_status = {"Training initiated.", Tone::Good};
    app.trainer.start(std::move(*cfg));
}

std::shared_ptr<const neural::MNISTDataset> active_dataset(const App& app) {
    return app.ds_use_test ? app.ds_test : app.ds_train;
}

void load_dataset_split(App& app, bool test) {
    app.grid_start = 0;
    app.selected_sample = 0;
    app.grid_dirty = true;
    app.sample_dirty = true;
    try {
        const fs::path dir(trim(app.dataset_text));
        const DatasetFiles f = dataset_files(dir);
        const fs::path& images = test ? f.test_images : f.train_images;
        const fs::path& labels = test ? f.test_labels : f.train_labels;
        if (!is_regular_file_checked(images) || !is_regular_file_checked(labels)) {
            throw std::runtime_error("Dataset files missing in directory");
        }
        auto data = std::make_shared<neural::MNISTDataset>(images, labels);
        const std::size_t count = data->size();
        (test ? app.ds_test : app.ds_train) = std::move(data);
        app.ds_status = {std::string("Loaded ") + (test ? "test" : "train") + " (" + std::to_string(count) + " items)", Tone::Good};
    } catch (const std::exception& e) {
        (test ? app.ds_test : app.ds_train).reset();
        app.ds_status = {e.what(), Tone::Bad};
    }
}

void reload_datasets(App& app) {
    app.ds_train.reset();
    app.ds_test.reset();
    app.eval_valid = false;
    load_dataset_split(app, app.ds_use_test);
}

void select_split(App& app, bool test) {
    if (app.ds_use_test == test) return;
    app.ds_use_test = test;
    app.grid_start = 0;
    app.selected_sample = 0;
    app.grid_dirty = true;
    app.sample_dirty = true;
    if (!active_dataset(app)) load_dataset_split(app, test);
}

void fill_grid_pixels(std::vector<std::uint8_t>& buf, const neural::MNISTDataset& data,
                      std::size_t start, std::size_t rows) {
    const std::size_t width = kGridCols * 28;
    buf.assign(width * rows * 28, 0);
    for (std::size_t k = 0; k < kGridCols * rows; ++k) {
        const std::size_t idx = start + k;
        if (idx >= data.size()) break;
        const neural::Tensor img = data.image(idx);
        const auto v = img.data();
        const std::size_t cx = (k % kGridCols) * 28;
        const std::size_t cy = (k / kGridCols) * 28;
        for (std::size_t y = 0; y < 28; ++y) {
            for (std::size_t x = 0; x < 28; ++x) {
                const float value = std::clamp(v[y * 28 + x], 0.0F, 1.0F);
                buf[(cy + y) * width + cx + x] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
            }
        }
    }
}

void update_sample(App& app, const neural::MNISTDataset& data) {
    app.sample_dirty = false;
    app.sample_pred.reset();
    app.sample_error.clear();
    try {
        if (data.size() == 0) return;
        const std::size_t idx = std::min(app.selected_sample, data.size() - 1);
        app.selected_sample = idx;
        app.sample_label = data.label(idx);
        const neural::Tensor img = data.image(idx);
        const auto v = img.data();
        for (std::size_t i = 0; i < kMnistInputs; ++i) {
            app.sample_pixels[i] = static_cast<std::uint8_t>(std::lround(std::clamp(v[i], 0.0F, 1.0F) * 255.0F));
        }
        if (!app.sample_tex) {
            app.sample_tex = std::make_unique<GrayTexture>(28, 28, app.sample_pixels, false);
        } else {
            app.sample_tex->update(app.sample_pixels);
        }
        if (app.network) app.sample_pred = predict(*app.network, img);
    } catch (const std::exception& e) {
        app.sample_error = e.what();
    }
}

void start_eval(App& app) {
    const auto data = active_dataset(app);
    if (!app.network || !data || app.evaluator.running()) return;
    app.eval_error.clear();
    try {
        app.evaluator.start(app.network->clone(), data, app.ds_use_test, app.loaded_path.string());
    } catch (const std::exception& e) {
        app.eval_error = e.what();
    }
}

void page_header(const Ui& ui, Rectangle content, const char* title, const char* subtitle) {
    draw_text(ui, title, content.x, content.y, 30, theme::text);
    draw_text(ui, subtitle, content.x, content.y + 38, 16, theme::muted);
}

void draw_sidebar(App& app, float height) {
    Ui& ui = app.ui;
    const float w = 210;
    DrawRectangleRec(rect(0, 0, w, height), theme::sidebar);
    DrawRectangleRec(rect(w - 1, 0, 1, height), theme::border);

    draw_text(ui, "NeuralEngine", 22, 24, 22, theme::text);
    draw_text(ui, "STUDIO PRO", 22, 50, 14, theme::accent);

    struct Item { Page page; const char* label; };
    const Item items[] = {
        {Page::Arena,   "Arena (Multi-Opt)"},
        {Page::Models,  "Models"},
        {Page::Predict, "Draw & Predict"},
        {Page::Dataset, "MNIST Data"},
        {Page::Train,   "Single Train"}
    };
    float y = 90;
    for (const Item& item : items) {
        const Rectangle r = rect(12, y, w - 24, 38);
        const bool selected = app.page == item.page;
        const bool hov = hovered(ui, r);
        if (selected) {
            fill_rounded(r, 6, theme::accent_dim);
            DrawRectangleRec(rect(12, y + 6, 3, 26), theme::accent);
        } else if (hov) {
            fill_rounded(r, 6, theme::panel_alt);
        }
        if (hov) ui.cursor_kind = MOUSE_CURSOR_POINTING_HAND;
        draw_text(ui, item.label, 28, y + 10, 15, selected ? theme::text : theme::muted);
        if (hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            app.page = item.page;
            ui.focused = -1;
        }
        y += 44;
    }

    const float fy = height - 120;
    draw_text(ui, "ACTIVE INFERENCE", 22, fy, 12, theme::faint);
    if (app.network) {
        draw_text(ui, elide_middle(ui, app.loaded_path.filename().string(), 14, w - 44), 22, fy + 18, 14, theme::text);
        draw_text(ui, elide_middle(ui, arch_to_string(app.network->architecture()), 12, w - 44), 22, fy + 38, 12, theme::muted);
    } else {
        draw_text(ui, "None", 22, fy + 18, 14, theme::muted);
    }
    draw_text(ui, "ARENA ENGINE", 22, fy + 60, 12, theme::faint);
    draw_text(ui, app.arena.is_busy() ? (app.arena.is_paused() ? "PAUSED" : "RUNNING") : "IDLE", 22, fy + 76, 14,
              app.arena.is_busy() ? theme::accent : theme::muted);
}

void page_models(App& app, Rectangle content) {
    Ui& ui = app.ui;
    page_header(ui, content, "Models", "Browse NeuralEngine checkpoints, then load one for inference or continued training.");

    const float top = content.y + 80;
    const float scan_w = 110;
    text_field(ui, kFieldRoot, rect(content.x, top + 18, content.width - scan_w - 12, 34), app.root_text, "Directory to search");
    draw_text(ui, "SEARCH ROOT", content.x, top, 14, theme::muted);
    if (button(ui, rect(content.x + content.width - scan_w, top + 18, scan_w, 34), "Refresh", true, ButtonStyle::Primary)) {
        run_scan(app);
    }

    const float row2 = top + 66;
    draw_text(ui, "FILTER", content.x, row2, 14, theme::muted);
    text_field(ui, kFieldFilter, rect(content.x, row2 + 18, content.width, 34), app.filter_text, "Filter by name or path");

    const std::string needle = lower(trim(app.filter_text));
    std::vector<const ModelEntry*> visible;
    for (const ModelEntry& e : app.scan.entries) {
        if (needle.empty() || lower(e.rel).find(needle) != std::string::npos) visible.push_back(&e);
    }

    const float body_y = row2 + 66;
    const float status_h = 26;
    const float body_h = std::max(120.0F, content.y + content.height - body_y - status_h - 8);
    const float list_w = std::max(320.0F, content.width * 0.56F);
    const Rectangle list = rect(content.x, body_y, list_w, body_h);
    const Rectangle detail = rect(content.x + list_w + 16, body_y, content.width - list_w - 16, body_h);

    if (ui.focused == -1 && !ui.blocked && !visible.empty()) {
        std::size_t idx = visible.size();
        for (std::size_t i = 0; i < visible.size(); ++i) {
            if (visible[i]->path == app.selected_path) idx = i;
        }
        if (IsKeyPressed(KEY_DOWN)) idx = (idx >= visible.size()) ? 0 : std::min(idx + 1, visible.size() - 1);
        if (IsKeyPressed(KEY_UP)) idx = (idx >= visible.size() || idx == 0) ? 0 : idx - 1;
        if (idx < visible.size()) app.selected_path = visible[idx]->path;
        if (IsKeyPressed(KEY_ENTER) && idx < visible.size() && visible[idx]->valid) load_model(app, visible[idx]->path);
    }

    fill_rounded_border(list, 10, theme::border, theme::panel);
    const float row_h = 58;
    const float total_h = static_cast<float>(visible.size()) * row_h;
    const Rectangle inner = rect(list.x + 6, list.y + 6, list.width - 12, list.height - 12);
    if (hovered(ui, list)) app.list_scroll -= GetMouseWheelMove() * 40.0F;
    app.list_scroll = std::clamp(app.list_scroll, 0.0F, std::max(0.0F, total_h - inner.height));

    if (visible.empty()) {
        draw_wrapped(ui, "No checkpoints match the filter.", list.x + 20, list.y + 20, list.width - 40, 16, theme::muted);
    } else {
        BeginScissorMode(static_cast<int>(inner.x), static_cast<int>(inner.y),
                         static_cast<int>(inner.width), static_cast<int>(inner.height));
        const Rectangle saved_clip = ui.clip;
        ui.clip = inner;
        for (std::size_t i = 0; i < visible.size(); ++i) {
            const float y = inner.y + static_cast<float>(i) * row_h - app.list_scroll;
            if (y + row_h < inner.y || y > inner.y + inner.height) continue;
            const ModelEntry& e = *visible[i];
            const Rectangle row = rect(inner.x, y, inner.width - 8, row_h - 4);
            const bool selected = e.path == app.selected_path;
            const bool hov = hovered(ui, row);
            if (selected) {
                fill_rounded(row, 6, theme::accent_dim);
            } else if (hov) {
                fill_rounded(row, 6, theme::panel_alt);
            }
            if (hov) {
                ui.cursor_kind = MOUSE_CURSOR_POINTING_HAND;
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    app.selected_path = e.path;
                    ui.focused = -1;
                }
            }
            const bool loaded = e.path == app.loaded_path && app.network.has_value();
            draw_text(ui, elide_middle(ui, e.name, 17, row.width - 150), row.x + 12, row.y + 7, 17, theme::text);
            draw_text(ui, elide_middle(ui, e.rel, 13, row.width - 24), row.x + 12, row.y + 30, 13, theme::muted);
            const std::string tag = !e.valid ? "INVALID" : (loaded ? "LOADED" : "");
            if (!tag.empty()) {
                const float tw = measure(ui, tag, 12);
                draw_text(ui, tag, row.x + row.width - tw - 12, row.y + 9, 12, e.valid ? theme::good : theme::bad);
            }
        }
        ui.clip = saved_clip;
        EndScissorMode();
    }

    draw_panel(ui, detail, "Selected model");
    const ModelEntry* sel = find_entry(app, app.selected_path);
    float dy = detail.y + 50;
    const float dx = detail.x + 18;
    const float dw = detail.width - 36;
    if (sel == nullptr) {
        draw_wrapped(ui, "Select a checkpoint from the list.", dx, dy, dw, 16, theme::muted);
    } else {
        draw_text(ui, elide_middle(ui, sel->name, 20, dw), dx, dy, 20, theme::text);
        dy += 30;
        draw_text(ui, "PATH", dx, dy, 12, theme::faint);
        dy += 16;
        dy += draw_wrapped(ui, sel->path.string(), dx, dy, dw, 14, theme::muted) + 8;
        draw_text(ui, "SIZE", dx, dy, 12, theme::faint);
        draw_text(ui, human_bytes(sel->bytes), dx + 80, dy - 2, 15, theme::text);
        dy += 24;
        if (sel->valid) {
            draw_text(ui, "ARCHITECTURE", dx, dy, 12, theme::faint);
            dy += 16;
            dy += draw_wrapped(ui, arch_to_string(sel->arch), dx, dy, dw, 16, theme::text) + 4;
            draw_text(ui, "PARAMETERS", dx, dy, 12, theme::faint);
            draw_text(ui, std::to_string(parameter_count(sel->arch)), dx + 100, dy - 2, 15, theme::text);
            dy += 28;
            const bool mnist = sel->arch.front() == kMnistInputs && sel->arch.back() == kMnistClasses;
            if (button(ui, rect(dx, dy, dw, 38), "Load for inference", mnist, ButtonStyle::Primary)) {
                load_model(app, sel->path);
            }
            dy += 46;
            if (button(ui, rect(dx, dy, dw, 38), "Resume training from this", mnist)) {
                app.continue_mode = true;
                app.continue_text = sel->path.string();
                app.page = Page::Train;
                ui.focused = -1;
            }
        } else if (!sel->error.empty()) {
            draw_wrapped(ui, sel->error, dx, dy, dw, 14, theme::bad);
        }
    }
    draw_text(ui, elide_middle(ui, app.models_status.text, 15, content.width),
              content.x, content.y + content.height - status_h, 15, tone_color(app.models_status.tone));
}

void page_predict(App& app, Rectangle content) {
    Ui& ui = app.ui;
    page_header(ui, content, "Draw & Predict", "Draw a digit with mouse. Centered by mass, scaled to 28x28 for inference.");

    const float side = std::clamp(std::min(content.width * 0.46F, content.height - 190.0F), 200.0F, 520.0F);
    const Rectangle area = rect(content.x, content.y + 84, side, side);

    const Vector2 m = GetMousePosition();
    const bool inside = hovered(ui, area);
    if (inside) ui.cursor_kind = MOUSE_CURSOR_CROSSHAIR;

    auto to_canvas = [&](Vector2 p) {
        return vec((p.x - area.x) * static_cast<float>(kCanvas) / area.width,
                   (p.y - area.y) * static_cast<float>(kCanvas) / area.height);
    };
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && inside) {
        app.drawing = true;
        app.last_point = to_canvas(m);
        stamp(app.canvas, app.last_point.x, app.last_point.y, kBrushRadius);
        app.canvas_dirty = true;
        ui.focused = -1;
    } else if (app.drawing && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        const Vector2 p = to_canvas(m);
        if (p.x != app.last_point.x || p.y != app.last_point.y) {
            stroke(app.canvas, app.last_point.x, app.last_point.y, p.x, p.y);
            app.last_point = p;
            app.canvas_dirty = true;
        }
    }
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT)) app.drawing = false;

    const float by = area.y + side + 14;
    if (button(ui, rect(area.x, by, 130, 38), "Clear (C)") || (ui.focused == -1 && !ui.blocked && IsKeyPressed(KEY_C))) {
        std::fill(app.canvas.begin(), app.canvas.end(), std::uint8_t{0});
        app.canvas_dirty = true;
    }
    if (button(ui, rect(area.x + 142, by, 160, 38), "Predict (Enter)", app.network.has_value(), ButtonStyle::Primary) ||
        (ui.focused == -1 && !ui.blocked && app.network.has_value() && IsKeyPressed(KEY_ENTER))) {
        request_prediction(app);
    }
    if (!app.network) {
        draw_text(ui, "No model loaded. Load one from the Models page.", area.x, by + 48, 14, theme::warn);
    } else if (!app.predict_error.empty()) {
        draw_text(ui, elide_middle(ui, app.predict_error, 14, side), area.x, by + 48, 14, theme::bad);
    }

    if (app.canvas_dirty) {
        app.canvas_texture->update(app.canvas);
        request_prediction(app);
        app.canvas_dirty = false;
    }

    fill_rounded_border(rect(area.x - 3, area.y - 3, side + 6, side + 6), 8, theme::border, theme::border);
    DrawTexturePro(app.canvas_texture->texture(),
                   rect(0, 0, static_cast<float>(kCanvas), static_cast<float>(kCanvas)),
                   area, vec(0, 0), 0.0F, WHITE);

    const float rx = area.x + side + 36;
    const float rw = content.x + content.width - rx;
    if (rw < 260) return;
    const Rectangle result_panel = rect(rx, area.y - 3, rw, side + 6 + 52);
    draw_panel(ui, result_panel, "Prediction");

    const float px = rx + 20;
    const auto& pred = app.prediction;
    const std::string digit = pred ? std::to_string(pred->digit) : "-";
    DrawTextEx(ui.font, digit.c_str(), vec(px, area.y + 40), 84, ui.spacing(84), pred ? theme::accent : theme::faint);
    if (pred) {
        draw_text(ui, fixed(100.0 * static_cast<double>(pred->probabilities[pred->digit]), 1) + "% confidence",
                  px + 90, area.y + 70, 18, theme::text);
    } else {
        draw_text(ui, "Draw a digit", px + 90, area.y + 70, 16, theme::muted);
    }

    const float bars_y = area.y + 150;
    const float bar_h = 22, bar_gap = 8, bar_x = px + 26, bar_w = rw - 40 - 26 - 64;
    for (std::size_t i = 0; i < kMnistClasses; ++i) {
        const float y = bars_y + static_cast<float>(i) * (bar_h + bar_gap);
        const bool best = pred && pred->digit == i;
        const double p = pred ? static_cast<double>(pred->probabilities[i]) : 0.0;
        draw_text(ui, std::to_string(i), px, y + 2, 17, best ? theme::accent : theme::muted);
        progress_bar(ui, rect(bar_x, y, bar_w, bar_h), p, best ? theme::accent : mix(theme::accent_dim, theme::faint, 0.5F));
        draw_text(ui, pred ? fixed(100.0 * p, 1) + "%" : "-", bar_x + bar_w + 10, y + 3, 15,
                  best ? theme::text : theme::muted);
    }
}

void page_dataset(App& app, Rectangle content) {
    Ui& ui = app.ui;
    page_header(ui, content, "MNIST Data", "Inspect images, labels, and batch evaluation accuracy on raw IDX data.");

    if (app.ds_auto_try) {
        app.ds_auto_try = false;
        load_dataset_split(app, app.ds_use_test);
    }

    const float top = content.y + 80;
    draw_text(ui, "DATASET DIRECTORY", content.x, top, 14, theme::muted);
    constexpr float split_w = 80.0F, load_w = 90.0F;
    const float field_w = std::max(120.0F, content.width - (2 * split_w + load_w + 30.0F));
    const bool enter = text_field(ui, kFieldDataset2, rect(content.x, top + 18, field_w, 34), app.dataset_text, "Directory with IDX files");
    float bx = content.x + field_w + 10;
    if (button(ui, rect(bx, top + 18, split_w, 34), "Test", true, app.ds_use_test ? ButtonStyle::Selected : ButtonStyle::Normal)) {
        select_split(app, true);
    }
    bx += split_w + 10;
    if (button(ui, rect(bx, top + 18, split_w, 34), "Train", true, !app.ds_use_test ? ButtonStyle::Selected : ButtonStyle::Normal)) {
        select_split(app, false);
    }
    bx += split_w + 10;
    if (button(ui, rect(bx, top + 18, load_w, 34), "Load", true, ButtonStyle::Primary) || enter) {
        reload_datasets(app);
    }

    const auto data = active_dataset(app);
    const float body_y = top + 66, status_h = 28;
    const float body_h = std::max(260.0F, content.y + content.height - body_y - status_h);
    const float gw = std::clamp(content.width * 0.5F, 300.0F, 520.0F);
    const Rectangle gp = rect(content.x, body_y, gw, body_h);
    const Rectangle rp = rect(content.x + gw + 16, body_y, content.width - gw - 16, body_h);

    draw_text(ui, elide_middle(ui, app.ds_status.text, 15, content.width),
              content.x, content.y + content.height - status_h + 6, 15, tone_color(app.ds_status.tone));

    draw_panel(ui, gp, "");
    if (!data) {
        draw_wrapped(ui, "No dataset loaded. Specify path and click Load.", gp.x + 20, gp.y + 20, gp.width - 40, 16, theme::muted);
        draw_panel(ui, rp, "");
        return;
    }

    const std::size_t n = data->size();
    const float cell = std::floor((gw - 24.0F) / static_cast<float>(kGridCols));
    const auto rows = static_cast<std::size_t>(std::clamp(std::floor((body_h - 96.0F) / cell), 1.0F, 10.0F));
    const std::size_t per_page = kGridCols * rows;

    const float pager_y = gp.y + 12;
    if (button(ui, rect(gp.x + 12, pager_y, 64, 32), "Prev", app.grid_start > 0)) {
        app.grid_start = (app.grid_start >= per_page) ? app.grid_start - per_page : 0;
        app.grid_dirty = true;
    }
    if (button(ui, rect(gp.x + 84, pager_y, 64, 32), "Next", app.grid_start + per_page < n)) {
        app.grid_start += per_page;
        app.grid_dirty = true;
    }
    {
        const std::size_t last = std::min(n, app.grid_start + per_page);
        draw_text(ui, std::to_string(app.grid_start) + " - " + std::to_string(last == 0 ? 0 : last - 1) + " of " + std::to_string(n),
                  gp.x + 160, pager_y + 8, 14, theme::muted);
    }

    const int tex_w = static_cast<int>(kGridCols * 28), tex_h = static_cast<int>(rows * 28);
    if (!app.grid_tex || app.grid_tex->width() != tex_w || app.grid_tex->height() != tex_h) {
        app.grid_pixels.assign(static_cast<std::size_t>(tex_w * tex_h), 0);
        app.grid_tex = std::make_unique<GrayTexture>(tex_w, tex_h, app.grid_pixels, true);
        app.grid_dirty = true;
    }
    if (app.grid_dirty) {
        fill_grid_pixels(app.grid_pixels, *data, app.grid_start, rows);
        app.grid_tex->update(app.grid_pixels);
        app.grid_dirty = false;
    }

    const Rectangle ga = rect(gp.x + 12, gp.y + 56, cell * static_cast<float>(kGridCols), cell * static_cast<float>(rows));
    DrawTexturePro(app.grid_tex->texture(), rect(0, 0, static_cast<float>(tex_w), static_cast<float>(tex_h)), ga, vec(0, 0), 0.0F, WHITE);

    if (hovered(ui, ga)) {
        ui.cursor_kind = MOUSE_CURSOR_POINTING_HAND;
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            const Vector2 m = GetMousePosition();
            const auto col = static_cast<std::size_t>(std::clamp((m.x - ga.x) / cell, 0.0F, static_cast<float>(kGridCols - 1)));
            const auto row = static_cast<std::size_t>(std::clamp((m.y - ga.y) / cell, 0.0F, static_cast<float>(rows - 1)));
            const std::size_t idx = app.grid_start + row * kGridCols + col;
            if (idx < n) {
                app.selected_sample = idx;
                app.sample_dirty = true;
            }
            ui.focused = -1;
        }
    }
    if (app.selected_sample >= app.grid_start && app.selected_sample < app.grid_start + per_page) {
        const std::size_t k = app.selected_sample - app.grid_start;
        DrawRectangleLinesEx(rect(ga.x + static_cast<float>(k % kGridCols) * cell,
                                  ga.y + static_cast<float>(k / kGridCols) * cell, cell, cell),
                             2.0F, theme::accent);
    }

    if (app.sample_dirty) update_sample(app, *data);
    const float sample_h = std::clamp(body_h * 0.5F, 224.0F, 310.0F);
    const Rectangle sp = rect(rp.x, rp.y, rp.width, sample_h);
    const Rectangle ep = rect(rp.x, rp.y + sample_h + 12, rp.width, std::max(120.0F, body_h - sample_h - 12));
    draw_panel(ui, sp, "Sample #" + std::to_string(app.selected_sample));

    if (app.sample_tex) {
        DrawTexturePro(app.sample_tex->texture(), rect(0, 0, 28, 28), rect(sp.x + 18, sp.y + 44, 100, 100), vec(0, 0), 0.0F, WHITE);
    }
    {
        const float tx = sp.x + 136;
        draw_text(ui, "Label: " + std::to_string(app.sample_label), tx, sp.y + 48, 20, theme::text);
        if (app.sample_pred) {
            const bool ok = app.sample_pred->digit == app.sample_label;
            draw_text(ui, "Predicted: " + std::to_string(app.sample_pred->digit), tx, sp.y + 78, 16,
                      ok ? theme::good : theme::bad);
            draw_text(ui, fixed(100.0 * static_cast<double>(app.sample_pred->probabilities[app.sample_pred->digit]), 1) + "% confidence",
                      tx, sp.y + 102, 13, theme::muted);
        } else if (!app.network) {
            draw_text(ui, "Load a model to predict", tx, sp.y + 80, 13, theme::muted);
        }
        if (!app.sample_error.empty()) {
            draw_wrapped(ui, app.sample_error, tx, sp.y + 124, sp.width - 154, 12, theme::bad);
        }
    }
    {
        const float gy = sp.y + sp.height - 48;
        const bool go_enter = text_field(ui, kFieldGoto, rect(sp.x + 18, gy, 140, 34), app.goto_text, "Go to index");
        const bool go_click = button(ui, rect(sp.x + 166, gy, 60, 34), "Go");
        if (go_enter || go_click) {
            std::size_t idx = 0;
            if (parse_size(app.goto_text, idx) && idx < n) {
                app.selected_sample = idx;
                app.grid_start = (idx / per_page) * per_page;
                app.grid_dirty = true;
                app.sample_dirty = true;
            } else {
                app.ds_status = {"Index must be an integer between 0 and " + std::to_string(n - 1), Tone::Warn};
            }
        }
    }

    draw_panel(ui, ep, "Evaluation");
    const bool eval_running = app.evaluator.running();
    if (button(ui, rect(ep.x + 18, ep.y + 44, 100, 34), "Evaluate", app.network.has_value() && !eval_running, ButtonStyle::Primary)) {
        start_eval(app);
    }
    if (button(ui, rect(ep.x + 126, ep.y + 44, 60, 34), "Stop", eval_running, ButtonStyle::Danger)) {
        app.evaluator.request_stop();
    }

    const float conf_size = std::clamp(std::min(ep.height - 56.0F, ep.width - 230.0F), 100.0F, 220.0F);
    const float text_w = std::max(100.0F, ep.width - 36.0F - conf_size - 10.0F);
    float ty = ep.y + 90;
    if (!app.network) {
        draw_wrapped(ui, "Load a model from the Models page to evaluate it.", ep.x + 18, ty, text_w, 13, theme::muted);
    }
    if (eval_running) {
        const double frac = app.evaluator.total() ? static_cast<double>(app.evaluator.done()) / static_cast<double>(app.evaluator.total()) : 0.0;
        progress_bar(ui, rect(ep.x + 18, ty, std::min(text_w, 180.0F), 16), frac, theme::accent,
                     fixed(100.0 * frac, 0) + "%");
        draw_text(ui, "Running acc: " + fixed(app.evaluator.running_accuracy(), 2) + "%", ep.x + 18, ty + 24, 13, theme::muted);
    } else if (app.eval_valid) {
        draw_text(ui, "Accuracy: " + fixed(app.eval.accuracy(), 2) + "%", ep.x + 18, ty, 14, theme::text);
        ty += 20;
        draw_text(ui, "Mean loss: " + fixed(app.eval.mean_loss(), 4), ep.x + 18, ty, 13, theme::muted);
        ty += 18;
        draw_text(ui, "Errors: " + std::to_string(app.eval.miss_count) + " / " + std::to_string(app.eval.total), ep.x + 18, ty, 13, theme::muted);
        ty += 18;
        std::size_t weak = 0;
        double weak_acc = 0.0;
        weakest_class(app.eval, weak, weak_acc);
        draw_text(ui, "Weakest: " + std::to_string(weak) + " (" + fixed(weak_acc, 1) + "%)", ep.x + 18, ty, 13, theme::muted);
        ty += 18;
        draw_text(ui, std::string("Split: ") + (app.eval.test_split ? "test" : "train"), ep.x + 18, ty, 13, theme::faint);
    }
    if (app.eval_valid) {
        draw_confusion(ui, rect(ep.x + ep.width - conf_size - 18, ep.y + 44, conf_size, conf_size), app.eval);
    }
    if (!app.eval_error.empty()) {
        draw_wrapped(ui, app.eval_error, ep.x + 18, ep.y + ep.height - 34, ep.width - 36, 12, theme::bad);
    }
}

double train_progress(const TrainSnapshot& s) {
    if (s.phase == TrainPhase::Done) return 1.0;
    if (s.phase == TrainPhase::Evaluating || s.phase == TrainPhase::Saving) return 1.0;
    if (s.per_epoch == 0 || s.epochs == 0) return 0.0;
    const double done = static_cast<double>((s.epoch > 0 ? s.epoch - 1 : 0) * s.per_epoch + s.in_epoch);
    return done / static_cast<double>(s.epochs * s.per_epoch);
}

void page_train(App& app, Rectangle content) {
    Ui& ui = app.ui;
    page_header(ui, content, "Train", "Train a single network with the optimiser of your choice and save a checkpoint.");

    const bool busy = app.trainer.running();
    const float form_w = std::clamp(content.width * 0.48F, 340.0F, 500.0F);
    const Rectangle right_panel = rect(content.x + form_w + 16, content.y + 80, content.width - form_w - 16, content.height - 80);

    float y = content.y + 80;
    const float x = content.x;
    const float w = form_w;
    const float half = w / 2 - 6;

    checkbox(ui, rect(x, y, w, 24), "Resume from checkpoint", app.continue_mode, !busy);
    y += 30;
    if (app.continue_mode) {
        y = labelled_field(ui, kFieldContinue, x, y, w, "CHECKPOINT TO RESUME", app.continue_text, "models/mnist_studio.txt", !busy);
    } else {
        y = labelled_field(ui, kFieldArch, x, y, w, "ARCHITECTURE", app.arch_text, "784 512 256 128 64 10", !busy);
    }

    labelled_field(ui, kFieldEpochs, x, y, half, "EPOCHS", app.epochs_text, "5", !busy);
    y = labelled_field(ui, kFieldLr, x + half + 12, y, half, "LEARNING RATE", app.lr_text, "0.01", !busy);

    const char* mom_label = "MOMENTUM";
    switch (app.train_opt) {
        case OptimiserType::Momentum: mom_label = "MOMENTUM"; break;
        case OptimiserType::RMSprop:  mom_label = "DECAY"; break;
        case OptimiserType::Adam:     mom_label = "BETA1"; break;
        default: break;
    }
    const float third = (w - 24.0F) / 3.0F;
    labelled_field(ui, kFieldMax, x, y, third, "MAX SAMPLES", app.max_text, "0 = all", !busy);
    labelled_field(ui, kFieldBatch, x + third + 12, y, third, "BATCH SIZE", app.batch_text, "1", !busy);
    y = labelled_field(ui, kFieldMomentum, x + 2 * (third + 12), y, third, mom_label, app.momentum_text, "0.9",
                       !busy && app.train_opt != OptimiserType::SGD);

    draw_text(ui, "OPTIMISER", x, y, 14, theme::muted);
    for (std::size_t i = 0; i < kOptimiserCount; ++i) {
        const auto t = static_cast<OptimiserType>(i);
        const Rectangle r = rect(x + static_cast<float>(i) * 84.0F, y + 18, 78, 34);
        if (button(ui, r, optimiser_name(t), !busy, app.train_opt == t ? ButtonStyle::Selected : ButtonStyle::Normal)) {
            app.train_opt = t;
            app.lr_text = kDefaultLrTexts[i];
            app.momentum_text = "0.9";
        }
    }
    y += 66;

    y = labelled_field(ui, kFieldDataset, x, y, w, "DATASET DIRECTORY", app.dataset_text, "data/mnist", !busy);
    y = labelled_field(ui, kFieldOutput, x, y, w, "OUTPUT CHECKPOINT", app.output_text, "models/mnist_single.txt", !busy);

    checkbox(ui, rect(x, y - 6, half + 40, 24), "Eval on test set", app.evaluate_test, !busy);
    checkbox(ui, rect(x + half + 40, y - 6, w - half - 40, 24), "Save on stop", app.save_on_stop, !busy);
    y += 30;

    if (button(ui, rect(x, y, 160, 38), "Start Training", !busy, ButtonStyle::Primary)) {
        begin_training(app);
    }
    if (button(ui, rect(x + 170, y, 120, 38), "Stop", busy, ButtonStyle::Danger)) {
        app.trainer.request_stop();
    }
    if (button(ui, rect(x, y + 46, 290, 34), "Export graph + data")) {
        app.export_request = 2;
    }
    if (!app.export_status.text.empty()) {
        draw_wrapped(ui, app.export_status.text, x, y + 90, w, 13, tone_color(app.export_status.tone));
    }

    draw_panel(ui, right_panel, "Telemetry");
    const TrainSnapshot s = app.trainer.snapshot(true);
    draw_text(ui, std::string("Status: ") + phase_name(s.phase), right_panel.x + 18, right_panel.y + 36, 13, theme::accent);
    draw_text(ui, "Elapsed: " + format_duration(s.elapsed), right_panel.x + 160, right_panel.y + 36, 13, theme::muted);
    checkbox(ui, rect(right_panel.x + right_panel.width - 130, right_panel.y + 32, 112, 22), "Log Y", app.log_chart);

    const Rectangle chart = rect(right_panel.x + 18, right_panel.y + 58, right_panel.width - 36, right_panel.height - 190);
    draw_loss_chart(ui, chart, s.history, static_cast<double>(s.epochs * s.per_epoch), s.per_epoch, s.epochs, app.log_chart);

    float iy = chart.y + chart.height + 12;
    const float ix = right_panel.x + 18;
    const float iw = right_panel.width - 36;
    progress_bar(ui, rect(ix, iy, iw, 14), train_progress(s), theme::accent,
                 fixed(100.0 * train_progress(s), 0) + "%");
    iy += 24;
    std::string line = "Epoch " + std::to_string(s.epoch) + "/" + std::to_string(s.epochs) +
                       "   Loss " + fixed(s.recent_loss, 4);
    if (s.elapsed > 0.5 && s.total_samples > 0) {
        line += "   " + fixed(static_cast<double>(s.total_samples) / s.elapsed, 0) + " samples/s";
    }
    draw_text(ui, elide_middle(ui, line, 13, iw), ix, iy, 13, theme::text);
    iy += 20;
    if (s.accuracy >= 0.0) {
        draw_text(ui, "Test accuracy " + fixed(s.accuracy, 2) + "%   test loss " + fixed(s.test_loss, 4), ix, iy, 13, theme::good);
        iy += 20;
    } else if (s.phase == TrainPhase::Evaluating && s.eval_total > 0) {
        draw_text(ui, "Evaluating " + std::to_string(s.eval_done) + "/" + std::to_string(s.eval_total), ix, iy, 13, theme::muted);
        iy += 20;
    }
    if (!s.error.empty()) {
        iy += draw_wrapped(ui, s.error, ix, iy, iw, 13, theme::bad);
    } else if (!s.message.empty()) {
        iy += draw_wrapped(ui, s.message, ix, iy, iw, 13, theme::muted);
    }
    if (!s.saved_path.empty()) {
        draw_text(ui, elide_middle(ui, "Saved: " + s.saved_path, 12, iw), ix, iy, 12, theme::faint);
    } else if (!app.train_form_status.text.empty() && s.phase == TrainPhase::Idle) {
        draw_wrapped(ui, app.train_form_status.text, ix, iy, iw, 13, tone_color(app.train_form_status.tone));
    }
    if (app.train_form_status.tone == Tone::Bad && !app.train_form_status.text.empty() && s.phase != TrainPhase::Idle &&
        s.error.empty()) {
        draw_wrapped(ui, app.train_form_status.text, ix, iy + 18, iw, 13, theme::bad);
    }
}

void start_arena(App& app) {
    std::size_t epochs = 0;
    std::size_t max_s = 0;
    if (!parse_size(app.arena_epochs_text, epochs) || epochs == 0 || epochs > 10000) {
        app.arena_status = {"Epochs must be an integer between 1 and 10000", Tone::Bad};
        return;
    }
    if (!parse_size(app.arena_max_text, max_s)) {
        app.arena_status = {"Max samples must be a valid integer (0 = all)", Tone::Bad};
        return;
    }
    std::size_t batch = 0;
    if (!parse_size(app.arena_batch_text, batch) || batch == 0 || batch > kMaxBatchSize) {
        app.arena_status = {"Batch size must be an integer between 1 and " + std::to_string(kMaxBatchSize), Tone::Bad};
        return;
    }
    std::vector<std::size_t> arch;
    try {
        arch = parse_architecture(app.arena_arch_text);
    } catch (const std::exception& ex) {
        app.arena_status = {ex.what(), Tone::Bad};
        return;
    }
    if (arch.front() != kMnistInputs || arch.back() != kMnistClasses) {
        app.arena_status = {"Topology must start with 784 inputs and end with 10 outputs", Tone::Bad};
        return;
    }
    const fs::path dir(trim(app.arena_data_text));
    if (!dataset_files(dir).has_train()) {
        app.arena_status = {"Training files not found in the dataset directory", Tone::Bad};
        return;
    }
    if (!app.arena.start(dir, arch, epochs, max_s, batch, app.arena_enabled, app.arena_lrs)) {
        app.arena_status = {"Enable at least one optimiser", Tone::Warn};
        return;
    }
    app.arena_status = {"Benchmark running.", Tone::Good};
}

void page_arena(App& app, Rectangle content) {
    Ui& ui = app.ui;
    page_header(ui, content, "Multi-Optimiser Comparison Arena",
                "Concurrently train models with identical weights and identical sample orders. Adjust learning rates on the fly.");

    checkbox(ui, rect(content.x + content.width - 150.0F, content.y + 4.0F, 150.0F, 24.0F), "Log scale (L)", app.arena_log);
    if (button(ui, rect(content.x + content.width - 160.0F - 90.0F, content.y + 2.0F, 90.0F, 28.0F), "Export")) {
        app.export_request = 1;
    }
    if (ui.focused == -1 && !ui.blocked && IsKeyPressed(KEY_L)) app.arena_log = !app.arena_log;

    const float ctrl_w = 330.0F;
    const Rectangle ctrl_panel = rect(content.x, content.y + 70, ctrl_w, content.height - 70);
    const float right_x = content.x + ctrl_w + 16.0F;
    const float right_w = content.width - ctrl_w - 16.0F;

    const float graph_h = (content.height - 70.0F) * 0.56F;
    const Rectangle graph_panel = rect(right_x, content.y + 70, right_w, graph_h);
    const Rectangle matrix_panel = rect(right_x, content.y + 70 + graph_h + 14.0F, right_w, (content.height - 70.0F) - graph_h - 14.0F);

    draw_panel(ui, ctrl_panel, "Arena Setup");
    float fy = ctrl_panel.y + 44.0F;
    const float fx = ctrl_panel.x + 16.0F;
    const float fw = ctrl_panel.width - 32.0F;

    const bool running = app.arena.is_busy();

    fy = labelled_field(ui, kFieldArenaData, fx, fy, fw, "DATASET DIRECTORY", app.arena_data_text, "data/mnist", !running);
    fy = labelled_field(ui, kFieldArenaArch, fx, fy, fw, "TOPOLOGY", app.arena_arch_text, "784 512 256 128 64 10", !running);

    const float third_w = (fw - 24.0F) / 3.0F;
    labelled_field(ui, kFieldArenaEpochs, fx, fy, third_w, "EPOCHS", app.arena_epochs_text, "5", !running);
    labelled_field(ui, kFieldArenaMax, fx + third_w + 12.0F, fy, third_w, "MAX SAMPLES", app.arena_max_text, "0 = all", !running);
    fy = labelled_field(ui, kFieldArenaBatch, fx + 2.0F * (third_w + 12.0F), fy, third_w, "BATCH", app.arena_batch_text, "1", !running);

    draw_text(ui, "OPTIMISERS & LIVE LEARNING RATE", fx, fy, 13, theme::muted);
    fy += 22.0F;

    for (std::size_t i = 0; i < kOptimiserCount; ++i) {
        const auto t = static_cast<OptimiserType>(i);
        const Rectangle row_box = rect(fx, fy, fw, 46.0F);

        fill_rounded_border(row_box, 6, app.arena_enabled[i] ? optimiser_color(t) : theme::border, theme::panel_alt);
        checkbox(ui, rect(fx + 8, fy + 13, 100, 20), optimiser_name(t), app.arena_enabled[i], !running);

        auto& met = app.arena.metrics(t);
        float current_lr = app.arena_lrs[i];
        draw_text(ui, "LR: " + tick_label(static_cast<double>(current_lr)), fx + 115, fy + 16, 12, theme::muted);

        const Rectangle btn_minus = rect(fx + fw - 48, fy + 13, 20, 20);
        const Rectangle btn_plus  = rect(fx + fw - 24, fy + 13, 20, 20);

        bool changed = false;
        if (button(ui, btn_minus, "-")) {
            current_lr = std::max(kMinLr, current_lr * 0.8F);
            changed = true;
        }
        if (button(ui, btn_plus, "+")) {
            current_lr = std::min(kMaxLr, current_lr * 1.25F);
            changed = true;
        }
        if (changed) {
            app.arena_lrs[i] = current_lr;
            met.live_lr.store(current_lr);  // picked up by the running worker on its next step
        }

        fy += 50.0F;
    }

    {
        const std::string err = app.arena.last_error();
        if (!err.empty()) {
            draw_wrapped(ui, err, fx, fy, fw, 12, theme::bad);
        } else if (!app.arena_status.text.empty()) {
            draw_wrapped(ui, app.arena_status.text, fx, fy, fw, 12, tone_color(app.arena_status.tone));
        }
    }

    const float btn_y = ctrl_panel.y + ctrl_panel.height - 52.0F;
    const Rectangle act_btn = rect(fx, btn_y, fw, 38.0F);

    if (!running) {
        if (button(ui, act_btn, "START BENCHMARK", true, ButtonStyle::Primary)) {
            start_arena(app);
        }
    } else {
        const float split_btn_w = fw / 2.0F - 4.0F;
        const Rectangle pause_btn = rect(fx, btn_y, split_btn_w, 38.0F);
        const Rectangle stop_btn = rect(fx + split_btn_w + 8.0F, btn_y, split_btn_w, 38.0F);

        if (app.arena.is_paused()) {
            if (button(ui, pause_btn, "Resume", true, ButtonStyle::Primary)) app.arena.resume();
        } else {
            if (button(ui, pause_btn, "Pause", true, ButtonStyle::Normal)) app.arena.pause();
        }
        if (button(ui, stop_btn, "Stop", true, ButtonStyle::Danger)) {
            app.arena.stop();
            app.arena_status = {"Benchmark stopped.", Tone::Info};
        }
    }

    draw_arena_plot(ui, graph_panel, app.arena, app.arena_log);

    draw_panel(ui, matrix_panel, "Performance Matrix");
    const float card_w = (matrix_panel.width - 32.0F - static_cast<float>(kOptimiserCount - 1) * 12.0F) / static_cast<float>(kOptimiserCount);
    const float card_h = matrix_panel.height - 48.0F;

    for (std::size_t i = 0; i < kOptimiserCount; ++i) {
        const auto t = static_cast<OptimiserType>(i);
        auto& met = app.arena.metrics(t);
        const bool enabled = met.enabled.load();

        const Rectangle cr = rect(matrix_panel.x + 16.0F + static_cast<float>(i) * (card_w + 12.0F),
                                  matrix_panel.y + 36.0F, card_w, card_h);
        fill_rounded_border(cr, 6, enabled ? optimiser_color(t) : theme::border, theme::panel_alt);

        DrawRectangle(static_cast<int>(cr.x) + 1, static_cast<int>(cr.y) + 1, static_cast<int>(cr.width) - 2, 22,
                      enabled ? mix(theme::panel_alt, optimiser_color(t), 0.25F) : theme::panel);
        draw_text(ui, optimiser_name(t), cr.x + 8, cr.y + 4, 13, enabled ? theme::text : theme::faint);

        if (!enabled) {
            draw_text(ui, "OFF", cr.x + 10, cr.y + 34, 14, theme::faint);
            continue;
        }

        if (met.diverged.load()) {
            const float tw = measure(ui, "DIVERGED", 10);
            draw_text(ui, "DIVERGED", cr.x + cr.width - tw - 6, cr.y + 6, 10, theme::bad);
        } else if (met.completed.load()) {
            const float tw = measure(ui, "DONE", 10);
            draw_text(ui, "DONE", cr.x + cr.width - tw - 6, cr.y + 6, 10, theme::good);
        }

        float py = cr.y + 30.0F;
        draw_text(ui, "Loss: " + fixed(static_cast<double>(met.recent_loss.load()), 4), cr.x + 8, py, 12, theme::text);
        py += 18.0F;
        draw_text(ui, "Train: " + fixed(static_cast<double>(met.train_acc.load()), 1) + "%", cr.x + 8, py, 12, theme::good);
        py += 18.0F;
        draw_text(ui, "Val: " + fixed(static_cast<double>(met.val_acc.load()), 1) + "%", cr.x + 8, py, 12, theme::accent);
        py += 18.0F;

        const double el = met.elapsed_sec.load();
        const double rate = el > 0.5 ? static_cast<double>(met.samples_processed.load()) / el : 0.0;
        draw_text(ui, "Speed: " + fixed(rate, 0) + "/s", cr.x + 8, py, 11, theme::muted);
        py += 18.0F;

        const double denom = static_cast<double>(app.arena.target_epochs()) * static_cast<double>(app.arena.samples_per_epoch());
        const double progress = denom > 0.0 ? static_cast<double>(met.samples_processed.load()) / denom : 0.0;
        progress_bar(ui, rect(cr.x + 8, py + 4, cr.width - 16, 6), progress, optimiser_color(t));
    }
}

void draw_modal(App& app, float width, float height) {
    Ui& ui = app.ui;
    DrawRectangleRec(rect(0, 0, width, height), Fade(BLACK, 0.6F));
    const float w = std::min(520.0F, width - 60);
    const Rectangle box = rect((width - w) / 2, height / 2 - 120, w, 230);
    fill_rounded_border(box, 12, theme::border, theme::panel);
    draw_text(ui, "Overwrite existing file?", box.x + 22, box.y + 18, 20, theme::text);
    std::string flat = app.modal_text;
    for (char& c : flat) if (c == '\n') c = ' ';
    draw_wrapped(ui, flat, box.x + 22, box.y + 56, box.width - 44, 15, theme::muted);
    const bool overwrite = button(ui, rect(box.x + 22, box.y + box.height - 60, 150, 40), "Overwrite", true, ButtonStyle::Danger);
    const bool cancel = button(ui, rect(box.x + 184, box.y + box.height - 60, 120, 40), "Cancel") || IsKeyPressed(KEY_ESCAPE);
    if (overwrite && app.pending) {
        app.train_form_status = {"Training initiated.", Tone::Good};
        app.trainer.start(std::move(*app.pending));
        app.pending.reset();
        app.modal_open = false;
    } else if (cancel) {
        app.pending.reset();
        app.modal_open = false;
    }
}

// ============================================================================
// Export (CSV data + PNG graphs)
// ============================================================================

std::string timestamp_string() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

std::string csv_quote(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

void write_text_file(const fs::path& path, const std::string& body) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Could not open for writing: " + path.string());
    f << body;
    f.flush();
    if (!f) throw std::runtime_error("Failed while writing " + path.string());
}

// Renders `draw` into an offscreen texture the size of the window and saves it as a PNG.
// The texture matches the window size on purpose: raylib's scissor mode (used by the plots)
// computes its Y flip from the screen height.
void render_png(const fs::path& path, const std::function<void(Rectangle)>& draw) {
    const int w = GetScreenWidth();
    const int h = GetScreenHeight();
    RenderTexture2D rt = LoadRenderTexture(w, h);
    if (rt.id == 0) throw std::runtime_error("Could not create render texture for export");
    BeginTextureMode(rt);
    ClearBackground(theme::bg);
    draw(rect(0.0F, 0.0F, static_cast<float>(w), static_cast<float>(h)));
    EndTextureMode();
    Image img = LoadImageFromTexture(rt.texture);
    ImageFlipVertical(&img);
    const bool ok = ExportImage(img, path.string().c_str());
    UnloadImage(img);
    UnloadRenderTexture(rt);
    if (!ok) throw std::runtime_error("Could not write PNG: " + path.string());
}

void export_arena(App& app, const fs::path& dir) {
    std::ostringstream curves, summary, config;
    curves << std::setprecision(9) << "optimiser,samples,loss\n";
    summary << std::setprecision(9)
            << "optimiser,enabled,completed,diverged,learning_rate,epoch,samples_processed,"
               "recent_loss,train_acc_pct,val_acc_pct,val_loss,elapsed_sec,samples_per_sec\n";
    bool any_points = false;
    for (std::size_t i = 0; i < kOptimiserCount; ++i) {
        const auto t = static_cast<OptimiserType>(i);
        auto& met = app.arena.metrics(t);
        const bool enabled = met.enabled.load();
        if (enabled) {
            const std::lock_guard<std::mutex> lock(met.curve_mutex);
            for (const LossPoint& pt : met.loss_curve) {
                if (!std::isfinite(pt.loss)) continue;
                curves << optimiser_name(t) << ',' << pt.samples << ',' << pt.loss << '\n';
                any_points = true;
            }
        }
        const double el = met.elapsed_sec.load();
        const double rate = el > 0.5 ? static_cast<double>(met.samples_processed.load()) / el : 0.0;
        summary << optimiser_name(t) << ',' << (enabled ? 1 : 0) << ',' << (met.completed.load() ? 1 : 0) << ','
                << (met.diverged.load() ? 1 : 0) << ',' << met.live_lr.load() << ',' << met.current_epoch.load() << ','
                << met.samples_processed.load() << ',' << met.recent_loss.load() << ',' << met.train_acc.load() << ','
                << met.val_acc.load() << ',' << met.val_loss.load() << ',' << el << ',' << rate << '\n';
    }
    if (!any_points) throw std::runtime_error("Nothing to export yet. Run the benchmark first.");

    config << "key,value\n"
           << "exported_at," << timestamp_string() << '\n'
           << "dataset," << csv_quote(trim(app.arena_data_text)) << '\n'
           << "topology," << csv_quote(trim(app.arena_arch_text)) << '\n'
           << "epochs," << csv_quote(trim(app.arena_epochs_text)) << '\n'
           << "max_samples," << csv_quote(trim(app.arena_max_text)) << '\n'
           << "batch_size," << csv_quote(trim(app.arena_batch_text)) << '\n'
           << "log_scale," << (app.arena_log ? "true" : "false") << '\n';

    write_text_file(dir / "loss_curves.csv", curves.str());
    write_text_file(dir / "summary.csv", summary.str());
    write_text_file(dir / "config.csv", config.str());
    render_png(dir / "arena_loss.png", [&](Rectangle r) { draw_arena_plot(app.ui, r, app.arena, app.arena_log); });
}

void export_train(App& app, const fs::path& dir) {
    const TrainSnapshot s = app.trainer.snapshot(true);
    if (s.history.size() < 2) throw std::runtime_error("Nothing to export yet. Run a training first.");

    std::ostringstream hist, summary;
    hist << std::setprecision(9) << "samples,loss\n";
    for (const LossPoint& pt : s.history) {
        if (std::isfinite(pt.loss)) hist << pt.samples << ',' << pt.loss << '\n';
    }
    summary << std::setprecision(9) << "key,value\n"
            << "exported_at," << timestamp_string() << '\n'
            << "phase," << phase_name(s.phase) << '\n'
            << "epoch," << s.epoch << '\n'
            << "epochs," << s.epochs << '\n'
            << "samples_per_epoch," << s.per_epoch << '\n'
            << "total_samples," << s.total_samples << '\n'
            << "recent_loss," << s.recent_loss << '\n'
            << "test_accuracy_pct," << s.accuracy << '\n'
            << "test_loss," << s.test_loss << '\n'
            << "elapsed_sec," << s.elapsed << '\n'
            << "saved_checkpoint," << csv_quote(s.saved_path) << '\n'
            << "optimiser," << optimiser_name(app.train_opt) << '\n'
            << "learning_rate_field," << csv_quote(trim(app.lr_text)) << '\n'
            << "batch_size_field," << csv_quote(trim(app.batch_text)) << '\n'
            << "momentum_field," << csv_quote(trim(app.momentum_text)) << '\n'
            << "architecture_field," << csv_quote(trim(app.arch_text)) << '\n'
            << "dataset," << csv_quote(trim(app.dataset_text)) << '\n';

    write_text_file(dir / "train_history.csv", hist.str());
    write_text_file(dir / "summary.csv", summary.str());
    render_png(dir / "train_loss.png", [&](Rectangle r) {
        draw_loss_chart(app.ui, r, s.history, static_cast<double>(s.epochs * s.per_epoch), s.per_epoch, s.epochs,
                        app.log_chart);
    });
}

// Runs outside BeginDrawing/EndDrawing (called from frame() after EndDrawing).
void run_exports(App& app) {
    if (app.export_request == 0) return;
    const int kind = app.export_request;
    app.export_request = 0;
    Status& status = (kind == 1) ? app.arena_status : app.export_status;
    try {
        const fs::path dir = app.project_root / "exports" / ((kind == 1 ? "arena_" : "train_") + timestamp_string());
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) throw std::runtime_error("Cannot create " + dir.string() + ": " + ec.message());
        if (kind == 1) {
            export_arena(app, dir);
        } else {
            export_train(app, dir);
        }
        status = {"Exported to " + dir.string(), Tone::Good};
    } catch (const std::exception& e) {
        status = {std::string("Export failed: ") + e.what(), Tone::Bad};
    }
}

void frame(App& app) {
    Ui& ui = app.ui;
    ui.cursor_kind = MOUSE_CURSOR_DEFAULT;
    ui.clip = rect(0, 0, 100000, 100000);
    ui.blocked = app.modal_open;

    const float width = static_cast<float>(GetScreenWidth());
    const float height = static_cast<float>(GetScreenHeight());

    BeginDrawing();
    ClearBackground(theme::bg);
    draw_sidebar(app, height);

    const float pad = 28;
    const Rectangle content = rect(210 + pad, pad, width - 210 - 2 * pad, height - 2 * pad);
    switch (app.page) {
        case Page::Arena:   page_arena(app, content); break;
        case Page::Models:  page_models(app, content); break;
        case Page::Predict: page_predict(app, content); break;
        case Page::Dataset: page_dataset(app, content); break;
        case Page::Train:   page_train(app, content); break;
    }

    if (app.modal_open) {
        ui.blocked = false;
        draw_modal(app, width, height);
    }
    EndDrawing();

    run_exports(app);

    SetMouseCursor(ui.cursor_kind);
    if (ui.focused == -1) {
        while (GetCharPressed() != 0) {}
    }

    const bool running = app.trainer.running();
    if (app.was_running && !running) {
        const TrainSnapshot s = app.trainer.snapshot(false);
        if (!s.saved_path.empty()) {
            run_scan(app);
            app.selected_path = fs::path(s.saved_path);
        }
    }
    app.was_running = running;

    EvalSummary result;
    std::string eval_error;
    if (app.evaluator.take_result(result, eval_error)) {
        if (!eval_error.empty()) {
            app.eval_error = eval_error;
            app.eval_valid = false;
        } else if (result.cancelled) {
            app.eval_valid = false;
        } else if (app.network && result.model_name == app.loaded_path.string()) {
            app.eval = std::move(result);
            app.eval_valid = true;
        } else {
            app.eval_valid = false;
        }
    }
}

// ============================================================================
// Headless Self-Test Runner
// ============================================================================

void write_test_idx(const fs::path& images, const fs::path& labels, std::uint32_t count) {
    auto be32 = [](std::ofstream& f, std::uint32_t v) {
        const std::array<char, 4> b = {static_cast<char>((v >> 24U) & 0xFFU), static_cast<char>((v >> 16U) & 0xFFU),
                                       static_cast<char>((v >> 8U) & 0xFFU), static_cast<char>(v & 0xFFU)};
        f.write(b.data(), 4);
    };
    std::ofstream fi(images, std::ios::binary);
    std::ofstream fl(labels, std::ios::binary);
    be32(fi, 2051U);
    be32(fi, count);
    be32(fi, 28U);
    be32(fi, 28U);
    be32(fl, 2049U);
    be32(fl, count);
    for (std::uint32_t i = 0; i < count; ++i) {
        for (std::uint32_t p = 0; p < 784U; ++p) {
            fi.put(((p + i * 13U) % 9U == 0U) ? static_cast<char>(255) : static_cast<char>(0));
        }
        fl.put(static_cast<char>(i % 10U));
    }
}

int run_self_tests() {
    int failures = 0;
    auto check = [&](bool ok, const std::string& name) {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << '\n';
        if (!ok) ++failures;
    };
    auto throws = [](const std::function<void()>& fn) {
        try { fn(); } catch (const std::exception&) { return true; }
        return false;
    };

    const fs::path tmp = fs::temp_directory_path() /
        ("neural_studio_selftest_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(tmp);

    try {
        {
            std::vector<std::uint8_t> canvas(static_cast<std::size_t>(kCanvas * kCanvas), 0);
            check(preprocess_digit(canvas, kCanvas).empty, "empty canvas yields no digit");
            stroke(canvas, 100, 40, 160, 240);
            const Preprocessed p = preprocess_digit(canvas, kCanvas);
            check(!p.empty && p.pixels.size() == 784, "preprocessed image has 784 values");
        }

        check(throws([] { (void)parse_architecture("784 abc 10"); }), "architecture parser rejects non-numeric input");
        check(throws([] { (void)parse_architecture("784"); }), "architecture parser rejects single size");
        check(parse_architecture("784, 32, 10").size() == 3, "architecture parser accepts commas");

        const std::vector<std::size_t> arch = {784, 24, 12, 10};
        Network a(arch, neural::WeightInit::He, kBaseSeed);
        const fs::path good = tmp / "models" / "good.txt";
        a.save(good);
        check(is_regular_file_checked(good), "checkpoint written to disk");

        Network b = load_checkpoint(good);
        check(a.layer_count() == b.layer_count(), "layer count matches across reload");

        {
            neural::Tensor probe({kMnistInputs}, 0.0F);
            auto d = probe.data();
            for (std::size_t i = 0; i < kMnistInputs; ++i) d[i] = (i % 7 == 0) ? 1.0F : 0.0F;
            const neural::Tensor la = a.forward(probe);
            const neural::Tensor lb = b.forward(probe);
            bool same = la.size() == lb.size();
            const auto da = la.data();
            const auto db = lb.data();
            for (std::size_t i = 0; same && i < la.size(); ++i) {
                same = std::fabs(da[i] - db[i]) < 1e-5F;
            }
            check(same, "reloaded checkpoint reproduces identical outputs");
        }

        auto write = [&](const char* name, const std::string& body) {
            const fs::path p = tmp / "bad" / name;
            fs::create_directories(p.parent_path());
            std::ofstream(p) << body;
            return p;
        };
        const fs::path junk = write("notes.txt", "just plain text\n");
        check(throws([&] { (void)load_checkpoint(junk); }), "invalid checkpoint correctly rejected");

        {
            Network net({784, 32, 10}, neural::WeightInit::He, kBaseSeed);
            neural::Tensor input({kMnistInputs}, 0.0F);
            auto d = input.data();
            for (std::size_t i = 0; i < 784; ++i) d[i] = (i % 5 == 0) ? 1.0F : 0.0F;
            const neural::SGD sgd(0.05F);
            const float first = net.train_step(input, 3, sgd);
            float last = first;
            for (int i = 0; i < 30; ++i) last = net.train_step(input, 3, sgd);
            check(last < first, "single sample train_step reduces loss");
        }

        {
            auto loss_drops = [&](auto& opt) {
                Network net({784, 32, 10}, neural::WeightInit::He, kBaseSeed);
                neural::Tensor input({kMnistInputs}, 0.0F);
                auto d = input.data();
                for (std::size_t i = 0; i < 784; ++i) d[i] = (i % 5 == 0) ? 1.0F : 0.0F;
                const float first = net.step(input, 3, opt);
                float last = first;
                for (int i = 0; i < 60; ++i) last = net.step(input, 3, opt);
                return std::isfinite(last) && last < first;
            };
            ArenaSGD sgd(0.05F);
            ArenaMomentum momentum(0.005F, 0.9F);
            ArenaRMSprop rmsprop(1e-4F, 0.9F);
            ArenaAdam adam(5e-4F);
            check(loss_drops(sgd), "ArenaSGD reduces loss");
            check(loss_drops(momentum), "ArenaMomentum reduces loss");
            check(loss_drops(rmsprop), "ArenaRMSprop reduces loss (bias uses its own accumulator)");
            check(loss_drops(adam), "ArenaAdam reduces loss");
        }

        {
            // A batch made of identical samples must give exactly the same update as one single step.
            auto make = [] { return Network({784, 16, 10}, neural::WeightInit::He, kBaseSeed); };
            Network single = make();
            Network batched = make();
            neural::Tensor input({kMnistInputs}, 0.0F);
            auto d = input.data();
            for (std::size_t i = 0; i < 784; ++i) d[i] = (i % 5 == 0) ? 1.0F : 0.0F;
            ArenaSGD o1(0.05F);
            ArenaSGD o2(0.05F);
            (void)single.step(input, 3, o1);
            for (int i = 0; i < 4; ++i) (void)batched.accumulate(input, 3);
            batched.apply_accumulated(o2, 4);
            const neural::Tensor ls = single.forward(input);
            const neural::Tensor lb = batched.forward(input);
            bool same = ls.size() == lb.size();
            const auto ds = ls.data();
            const auto db = lb.data();
            for (std::size_t i = 0; same && i < ls.size(); ++i) same = std::fabs(ds[i] - db[i]) < 1e-5F;
            check(same, "mini-batch of identical samples equals one single step");
        }

        {
            const fs::path images = tmp / "synthetic-images-idx3-ubyte";
            const fs::path labels = tmp / "synthetic-labels-idx1-ubyte";
            write_test_idx(images, labels, 30);
            const auto ds = std::make_shared<neural::MNISTDataset>(images, labels);
            check(ds->size() == 30 && ds->label(13) == 3, "synthetic IDX dataset parses accurately");
        }

        {
            const fs::path data_dir = tmp / "arena_data";
            fs::create_directories(data_dir);
            write_test_idx(data_dir / kTrainImages, data_dir / kTrainLabels, 40);

            MultiTrainerArena arena;
            const MultiTrainerArena::OptFlags enabled = {true, true, true, true};
            const MultiTrainerArena::OptLrs lrs = kDefaultLrs;
            const bool started = arena.start(data_dir, {784, 16, 10}, 2, 0, 7, enabled, lrs);
            check(started, "arena starts");

            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
            while (arena.is_busy() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            arena.stop();

            bool all_done = true;
            for (std::size_t i = 0; i < kOptimiserCount; ++i) {
                auto& m = arena.metrics(static_cast<OptimiserType>(i));
                all_done = all_done && m.completed.load() && m.samples_processed.load() == 80;
            }
            check(all_done && arena.last_error().empty(), "arena trains all four optimisers to completion without errors");
            check(!arena.is_busy(), "arena is idle after completion");
        }
    } catch (const std::exception& e) {
        std::cout << "[FAIL] unexpected exception: " << e.what() << '\n';
        ++failures;
    }

    std::error_code ec;
    fs::remove_all(tmp, ec);
    std::cout << (failures == 0 ? "All self-tests passed.\n" : "Some tests failed.\n");
    return failures == 0 ? 0 : 1;
}

}  // namespace

// ============================================================================
// Main Application Entrypoint
// ============================================================================

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--self-test") {
        return run_self_tests();
    }

    try {
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "NeuralEngine Studio - High-Performance Experiment Lab");
        SetWindowMinSize(1000, 680);
        SetExitKey(KEY_NULL);
        SetTargetFPS(60);

        Font font = GetFontDefault();
        bool custom_font = false;
        const char* candidates[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        };
        for (const char* path : candidates) {
            if (FileExists(path)) {
                const Font f = LoadFontEx(path, 40, nullptr, 0);
                if (f.texture.id != 0) {
                    SetTextureFilter(f.texture, TEXTURE_FILTER_BILINEAR);
                    font = f;
                    custom_font = true;
                    break;
                }
            }
        }

        {
            App app;
            app.ui.font = font;
            app.ui.custom_font = custom_font;
            app.project_root = find_project_root();
            app.root_text = app.project_root.string();
            app.dataset_text = default_dataset_dir(app.project_root).string();
            app.arena_data_text = app.dataset_text;
            app.output_text = (app.project_root / "models" / "mnist_studio.txt").string();
            app.canvas_texture = std::make_unique<GrayTexture>(kCanvas, kCanvas, app.canvas, true);

            run_scan(app);

            while (!WindowShouldClose()) {
                frame(app);
            }
            app.trainer.request_stop();
            app.evaluator.request_stop();
            app.arena.stop();
        }

        if (custom_font) UnloadFont(font);
        CloseWindow();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal engine termination: " << e.what() << '\n';
        return 1;
    }
}