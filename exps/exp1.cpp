//use this to reoroduce the experiment_results.csv 

#include "neural/data/dataset.hpp"
#include "neural/nn/layer.hpp"
#include "neural/ops/activations.hpp"
#include "neural/ops/loss.hpp"
#include "neural/optim/optimizer.hpp"

#include <raylib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

using Optimizer = std::variant<
    neural::SGD,
    neural::MomentumSGD,
    neural::RMSprop,
    neural::Adam
>;

constexpr std::size_t TRAIN_LIMIT = 30000;
constexpr std::size_t VALIDATION_START = 30000;
constexpr std::size_t VALIDATION_LIMIT = 5000;
constexpr std::size_t EPOCHS = 10;

constexpr std::array<std::size_t, 10> BATCH_SIZES = {
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
};

constexpr std::array<float, 3> LEARNING_RATES = {
    0.001F, 0.01F, 0.1F
};

constexpr std::array<const char*, 4> OPTIMIZER_NAMES = {
    "SGD", "Momentum", "RMSprop", "Adam"
};

constexpr std::array<Color, 4> OPTIMIZER_COLORS = {
    RED, BLUE, GREEN, GOLD
};

enum class OptimizerType {
    SGD,
    Momentum,
    RMSprop,
    Adam
};

enum class GraphMetric {
    Accuracy,
    TrainingLoss,
    ValidationLoss,
    Time
};

struct Architecture {
    std::string name;
    std::vector<std::size_t> hidden;
};

struct Metrics {
    float loss = 0.0F;
    float accuracy = 0.0F;
};

struct EpochResult {
    std::string optimizer;
    std::string architecture;
    float learning_rate = 0.0F;
    std::size_t batch_size = 0;
    std::size_t epoch = 0;
    float training_loss = 0.0F;
    float validation_loss = 0.0F;
    float validation_accuracy = 0.0F;
    double elapsed_seconds = 0.0;
    std::size_t updates = 0;
};

std::vector<Architecture> architectures() {
    return {
        {"784-4-10", {4}},
        {"784-8-10", {8}},
        {"784-4-4-10", {4, 4}},
        {"784-4-8-10", {4, 8}},
        {"784-8-4-10", {8, 4}},
        {"784-8-8-10", {8, 8}}
    };
}

Optimizer make_optimizer(OptimizerType type, float learning_rate) {
    switch (type) {
        case OptimizerType::SGD:
            return neural::SGD(learning_rate);
        case OptimizerType::Momentum:
            return neural::MomentumSGD(learning_rate);
        case OptimizerType::RMSprop:
            return neural::RMSprop(learning_rate);
        case OptimizerType::Adam:
            return neural::Adam(learning_rate);
    }

    return neural::SGD(learning_rate);
}

void optimizer_step(Optimizer& optimizer, neural::Dense& layer) {
    std::visit([&](auto& opt) {
        opt.step(layer);
    }, optimizer);
}

void relu(neural::Tensor& tensor) {
    for (float& value : tensor.data())
        value = std::max(0.0F, value);
}

void relu_backward(
    neural::Tensor& gradient,
    const neural::Tensor& preactivation
) {
    auto gradients = gradient.data();
    const auto values = preactivation.data();

    for (std::size_t i = 0; i < gradients.size(); ++i) {
        if (values[i] <= 0.0F)
            gradients[i] = 0.0F;
    }
}

std::pair<float, neural::Tensor> batch_loss_and_gradient(
    const neural::Tensor& logits,
    const std::vector<std::size_t>& labels
) {
    const std::size_t batch_size = logits.shape()[0];
    const std::size_t classes = logits.shape()[1];

    neural::Tensor gradient({batch_size, classes});

    const auto predictions = logits.data();
    auto gradients = gradient.data();

    float total_loss = 0.0F;

    for (std::size_t row = 0; row < batch_size; ++row) {
        const std::size_t offset = row * classes;

        float maximum = -std::numeric_limits<float>::infinity();

        for (std::size_t col = 0; col < classes; ++col)
            maximum = std::max(maximum, predictions[offset + col]);

        float exponential_sum = 0.0F;

        for (std::size_t col = 0; col < classes; ++col)
            exponential_sum += std::exp(
                predictions[offset + col] - maximum
            );

        total_loss += maximum + std::log(exponential_sum)
                    - predictions[offset + labels[row]];

        for (std::size_t col = 0; col < classes; ++col) {
            const float probability =
                std::exp(predictions[offset + col] - maximum)
                / exponential_sum;

            gradients[offset + col] =
                (probability - (col == labels[row] ? 1.0F : 0.0F))
                / static_cast<float>(batch_size);
        }
    }

    total_loss /= static_cast<float>(batch_size);

    return {total_loss, std::move(gradient)};
}

std::vector<neural::Dense> make_layers(
    const Architecture& architecture
) {
    std::vector<neural::Dense> layers;
    layers.reserve(architecture.hidden.size() + 1);

    std::size_t input_features = 784;
    std::uint32_t seed = 101;

    for (std::size_t neurons : architecture.hidden) {
        layers.emplace_back(
            input_features,
            neurons,
            neural::WeightInit::He,
            seed++
        );

        input_features = neurons;
    }

    layers.emplace_back(
        input_features,
        10,
        neural::WeightInit::Xavier,
        seed
    );

    return layers;
}

Metrics evaluate(
    const neural::MNISTDataset& dataset,
    std::vector<neural::Dense>& layers,
    const Architecture& architecture
) {
    Metrics metrics;
    neural::ReLU activation;

    for (std::size_t index = VALIDATION_START;
         index < VALIDATION_START + VALIDATION_LIMIT; ++index) {

        neural::Tensor output = dataset.image(index);

        for (std::size_t i = 0; i < architecture.hidden.size(); ++i) {
            output = layers[i].forward(output, false);
            output = activation.forward(output);
        }

        const neural::Tensor logits =
            layers.back().forward(output, false);

        const auto values = logits.data();
        const std::size_t target = dataset.label(index);

        const float maximum =
            *std::max_element(values.begin(), values.end());

        float exponential_sum = 0.0F;

        for (float value : values)
            exponential_sum += std::exp(value - maximum);

        metrics.loss += maximum + std::log(exponential_sum)
                      - values[target];

        const auto prediction = static_cast<std::size_t>(
            std::distance(
                values.begin(),
                std::max_element(values.begin(), values.end())
            )
        );

        if (prediction == target)
            metrics.accuracy += 1.0F;
    }

    metrics.loss /= static_cast<float>(VALIDATION_LIMIT);
    metrics.accuracy =
        metrics.accuracy / static_cast<float>(VALIDATION_LIMIT) * 100.0F;

    return metrics;
}

void write_csv_header(std::ofstream& csv) {
    csv << "optimizer,architecture,learning_rate,batch_size,"
           "epoch,training_loss,validation_loss,validation_accuracy,"
           "elapsed_seconds,updates\n";
}

void write_csv(std::ofstream& csv, const EpochResult& result) {
    csv << result.optimizer << ','
        << result.architecture << ','
        << result.learning_rate << ','
        << result.batch_size << ','
        << result.epoch << ','
        << result.training_loss << ','
        << result.validation_loss << ','
        << result.validation_accuracy << ','
        << result.elapsed_seconds << ','
        << result.updates << '\n';

    csv.flush();
}

std::vector<EpochResult> train_configuration(
    const neural::MNISTDataset& dataset,
    const Architecture& architecture,
    OptimizerType optimizer_type,
    float learning_rate,
    std::size_t batch_size,
    std::ofstream& csv,
    std::size_t configuration_number,
    std::size_t total_configurations
) {
    auto layers = make_layers(architecture);
    auto optimizer = make_optimizer(optimizer_type, learning_rate);

    std::vector<std::size_t> indices(TRAIN_LIMIT);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(42);

    std::vector<EpochResult> history;
    history.reserve(EPOCHS);

    for (std::size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        const auto start_time = Clock::now();

        std::shuffle(indices.begin(), indices.end(), rng);

        float total_loss = 0.0F;
        std::size_t processed = 0;
        std::size_t updates = 0;

        for (std::size_t start = 0; start < TRAIN_LIMIT;
             start += batch_size) {

            const std::size_t current_batch =
                std::min(batch_size, TRAIN_LIMIT - start);

            neural::Tensor input({current_batch, 784});
            std::vector<std::size_t> labels(current_batch);

            auto input_values = input.data();

            for (std::size_t i = 0; i < current_batch; ++i) {
                const std::size_t index = indices[start + i];
                const neural::Tensor image = dataset.image(index);

                std::copy(
                    image.data().begin(),
                    image.data().end(),
                    input_values.begin()
                        + static_cast<std::ptrdiff_t>(i * 784)
                );

                labels[i] = dataset.label(index);
            }

            std::vector<neural::Tensor> preactivations;
            preactivations.reserve(architecture.hidden.size());

            neural::Tensor activation = std::move(input);

            for (std::size_t i = 0;
                 i < architecture.hidden.size(); ++i) {

                neural::Tensor z = layers[i].forward(activation);
                preactivations.push_back(z);

                activation = z;
                relu(activation);
            }

            neural::Tensor logits =
                layers.back().forward(activation);

            auto [loss, gradient] =
                batch_loss_and_gradient(logits, labels);

            total_loss += loss * static_cast<float>(current_batch);
            processed += current_batch;

            for (std::size_t i = layers.size(); i-- > 0;) {
                if (i < architecture.hidden.size())
                    relu_backward(gradient, preactivations[i]);

                gradient = layers[i].backward(gradient);
            }

            for (auto& layer : layers)
                optimizer_step(optimizer, layer);

            ++updates;
        }

        const Metrics validation = evaluate(
            dataset,
            layers,
            architecture
        );

        const auto end_time = Clock::now();

        EpochResult result;
        result.optimizer = OPTIMIZER_NAMES[
            static_cast<std::size_t>(optimizer_type)
        ];
        result.architecture = architecture.name;
        result.learning_rate = learning_rate;
        result.batch_size = batch_size;
        result.epoch = epoch + 1;
        result.training_loss =
            total_loss / static_cast<float>(processed);
        result.validation_loss = validation.loss;
        result.validation_accuracy = validation.accuracy;
        result.elapsed_seconds =
            std::chrono::duration<double>(
                end_time - start_time
            ).count();
        result.updates = updates;

        write_csv(csv, result);
        history.push_back(result);

        std::cout
            << "\n[" << configuration_number << '/'
            << total_configurations << "] "
            << result.optimizer << " | "
            << architecture.name << " | LR "
            << learning_rate << " | Batch "
            << batch_size << '\n'
            << "Epoch " << result.epoch << '/' << EPOCHS
            << " | Train loss: " << result.training_loss
            << " | Validation loss: " << result.validation_loss
            << " | Accuracy: " << result.validation_accuracy << "%"
            << " | Time: " << result.elapsed_seconds << "s"
            << " | Updates: " << result.updates << '\n';
    }

    return history;
}

// -----------------------------------------------------------------------------
// Graph viewer
// -----------------------------------------------------------------------------

float metric_value(const EpochResult& result, GraphMetric metric) {
    switch (metric) {
        case GraphMetric::Accuracy:
            return result.validation_accuracy;
        case GraphMetric::TrainingLoss:
            return result.training_loss;
        case GraphMetric::ValidationLoss:
            return result.validation_loss;
        case GraphMetric::Time:
            return static_cast<float>(result.elapsed_seconds);
    }

    return 0.0F;
}

const char* metric_name(GraphMetric metric) {
    switch (metric) {
        case GraphMetric::Accuracy: return "Validation accuracy (%)";
        case GraphMetric::TrainingLoss: return "Training loss";
        case GraphMetric::ValidationLoss: return "Validation loss";
        case GraphMetric::Time: return "Training time (seconds)";
    }

    return "";
}

void draw_graph(
    const std::vector<EpochResult>& results,
    const std::vector<Architecture>& archs
) {
    constexpr int width = 1200;
    constexpr int height = 800;
    constexpr int left = 100;
    constexpr int right = 70;
    constexpr int top = 120;
    constexpr int bottom = 110;

    constexpr int graph_width = width - left - right;
    constexpr int graph_height = height - top - bottom;

    InitWindow(width, height, "NeuralEngine - Experiment Results");
    SetTargetFPS(60);

    std::size_t architecture_index = 0;
    std::size_t learning_rate_index = 0;
    GraphMetric metric = GraphMetric::Accuracy;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_RIGHT))
            architecture_index =
                (architecture_index + 1) % archs.size();

        if (IsKeyPressed(KEY_LEFT))
            architecture_index =
                (architecture_index + archs.size() - 1) % archs.size();

        if (IsKeyPressed(KEY_UP))
            learning_rate_index =
                (learning_rate_index + 1) % LEARNING_RATES.size();

        if (IsKeyPressed(KEY_DOWN))
            learning_rate_index =
                (learning_rate_index + LEARNING_RATES.size() - 1)
                % LEARNING_RATES.size();

        if (IsKeyPressed(KEY_TAB))
            metric = static_cast<GraphMetric>(
                (static_cast<int>(metric) + 1) % 4
            );

        float minimum = 0.0F;
        float maximum = metric == GraphMetric::Accuracy ? 100.0F : 1.0F;

        if (metric != GraphMetric::Accuracy) {
            minimum = std::numeric_limits<float>::max();
            maximum = std::numeric_limits<float>::lowest();

            for (const auto& result : results) {
                if (result.architecture != archs[architecture_index].name ||
                    result.learning_rate !=
                        LEARNING_RATES[learning_rate_index] ||
                    result.epoch != EPOCHS)
                    continue;

                const float value = metric_value(result, metric);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }

            if (minimum == std::numeric_limits<float>::max()) {
                minimum = 0.0F;
                maximum = 1.0F;
            }

            if (maximum - minimum < 1e-6F) {
                minimum -= 0.5F;
                maximum += 0.5F;
            } else {
                const float padding = (maximum - minimum) * 0.1F;
                minimum = std::max(0.0F, minimum - padding);
                maximum += padding;
            }
        }

        BeginDrawing();
        ClearBackground({22, 24, 32, 255});

        DrawText("NeuralEngine - Experiment Results",
                 left, 20, 28, RAYWHITE);

        DrawText(
            TextFormat("Architecture: %s    Learning rate: %.3f",
                       archs[architecture_index].name.c_str(),
                       LEARNING_RATES[learning_rate_index]),
            left, 60, 19, LIGHTGRAY
        );

        DrawText(TextFormat("Metric: %s", metric_name(metric)),
                 left, 88, 17, SKYBLUE);

        DrawLine(left, top, left, top + graph_height, LIGHTGRAY);
        DrawLine(left, top + graph_height,
                 left + graph_width, top + graph_height, LIGHTGRAY);

        for (int i = 0; i <= 10; ++i) {
            const float ratio = static_cast<float>(i) / 10.0F;
            const int y = top + graph_height
                - static_cast<int>(ratio * graph_height);

            DrawLine(left, y, left + graph_width, y,
                     {55, 58, 68, 255});

            const float value =
                minimum + ratio * (maximum - minimum);

            DrawText(TextFormat("%.2f", value),
                     35, y - 8, 15, LIGHTGRAY);
        }

        for (std::size_t i = 0; i < BATCH_SIZES.size(); ++i) {
            const float ratio =
                std::log2(static_cast<float>(BATCH_SIZES[i]) / 8.0F)
                / std::log2(4096.0F / 8.0F);

            const int x = left + static_cast<int>(ratio * graph_width);

            DrawLine(x, top, x, top + graph_height,
                     {45, 48, 58, 255});

            DrawText(TextFormat("%zu", BATCH_SIZES[i]),
                     x - 17, top + graph_height + 12, 15, LIGHTGRAY);
        }

        DrawText("Batch size (logarithmic scale)",
                 left + graph_width / 2 - 115,
                 height - 35, 18, RAYWHITE);

        for (std::size_t opt = 0; opt < OPTIMIZER_NAMES.size(); ++opt) {
            std::vector<Vector2> points;

            for (std::size_t batch : BATCH_SIZES) {
                const EpochResult* selected = nullptr;

                for (const auto& result : results) {
                    if (result.optimizer != OPTIMIZER_NAMES[opt] ||
                        result.architecture !=
                            archs[architecture_index].name ||
                        result.learning_rate !=
                            LEARNING_RATES[learning_rate_index] ||
                        result.batch_size != batch ||
                        result.epoch != EPOCHS)
                        continue;

                    selected = &result;
                    break;
                }

                if (selected == nullptr)
                    continue;

                const float x_ratio =
                    std::log2(static_cast<float>(batch) / 8.0F)
                    / std::log2(4096.0F / 8.0F);

                const float value = metric_value(*selected, metric);
                const float y_ratio =
                    (value - minimum) / (maximum - minimum);

                points.push_back({
                    static_cast<float>(left) + x_ratio * graph_width,
                    static_cast<float>(top + graph_height)
                        - y_ratio * graph_height
                });
            }

            for (std::size_t i = 0; i < points.size(); ++i) {
                DrawCircleV(points[i], 5.0F, OPTIMIZER_COLORS[opt]);

                if (i > 0)
                    DrawLineEx(points[i - 1], points[i],
                               2.5F, OPTIMIZER_COLORS[opt]);
            }

            const int legend_x = left + static_cast<int>(opt) * 180;

            DrawRectangle(legend_x, top + graph_height + 55,
                          18, 5, OPTIMIZER_COLORS[opt]);

            DrawText(OPTIMIZER_NAMES[opt],
                     legend_x + 25, top + graph_height + 48,
                     16, RAYWHITE);
        }

        DrawText("LEFT/RIGHT: architecture | UP/DOWN: learning rate",
                 left, height - 78, 16, LIGHTGRAY);
        DrawText("TAB: change metric | ESC: close",
                 left, height - 56, 16, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();
}

} // namespace

int main() {
    std::cout << "Loading MNIST...\n";

    neural::MNISTDataset training(
        "../data/mnist/train-images-idx3-ubyte",
        "../data/mnist/train-labels-idx1-ubyte"
    );

    const auto archs = architectures();

    constexpr std::size_t total_configurations =
        6 * 4 * LEARNING_RATES.size() * BATCH_SIZES.size();

    std::ofstream csv("experiment_results.csv");

    if (!csv) {
        std::cerr << "Could not open experiment_results.csv\n";
        return 1;
    }

    csv << std::fixed << std::setprecision(6);
    write_csv_header(csv);

    std::vector<EpochResult> results;
    results.reserve(total_configurations * EPOCHS);

    std::size_t configuration_number = 0;

    std::cout << "\nStarting experiment\n"
              << "Training samples: " << TRAIN_LIMIT << '\n'
              << "Validation samples: " << VALIDATION_LIMIT << '\n'
              << "Epochs: " << EPOCHS << '\n'
              << "Batch sizes: 8 to 4096\n"
              << "Configurations: " << total_configurations << '\n'
              << "Total epochs: " << total_configurations * EPOCHS
              << "\n\n";

    const auto experiment_start = Clock::now();

    for (const auto& architecture : archs) {
        for (std::size_t opt = 0; opt < OPTIMIZER_NAMES.size(); ++opt) {
            const auto optimizer_type = static_cast<OptimizerType>(opt);

            for (float learning_rate : LEARNING_RATES) {
                for (std::size_t batch_size : BATCH_SIZES) {
                    ++configuration_number;

                    std::cout
                        << "\n========================================\n"
                        << "Configuration " << configuration_number
                        << '/' << total_configurations
                        << "\n========================================\n";

                    auto history = train_configuration(
                        training,
                        architecture,
                        optimizer_type,
                        learning_rate,
                        batch_size,
                        csv,
                        configuration_number,
                        total_configurations
                    );

                    results.insert(
                        results.end(),
                        history.begin(),
                        history.end()
                    );
                }
            }
        }
    }

    const auto experiment_end = Clock::now();

    std::cout << "\nExperiment completed in "
              << std::chrono::duration<double>(
                     experiment_end - experiment_start
                 ).count()
              << " seconds.\n";

    csv.close();

    std::cout << "Results saved to experiment_results.csv\n";
    std::cout << "Opening graph viewer...\n";

    draw_graph(results, archs);

    return 0;
}