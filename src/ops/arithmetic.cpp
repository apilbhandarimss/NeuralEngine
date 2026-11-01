// compile with: -O3 -march=native -fopenmp -fno-math-errno
#include "neural/ops/arithmetic.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace neural {
namespace {

constexpr int kThreads = 14;
// Element-wise ops are memory-bound; tiny tensors stay single-threaded.
constexpr std::size_t kParallelMin = 32768;
// matmul threshold counts multiply-adds.
constexpr std::size_t kParallelMatmulWork = 32768;
// Transpose tile edge (32 x 32 floats = 4 KB per tile).
constexpr std::size_t kTile = 32;
// matmul processes this many output rows per pass over the rhs matrix.
constexpr std::size_t kRowBlock = 4;

std::size_t element_count(
    const std::vector<std::size_t>& shape
) {
    std::size_t count = 1;

    for (const std::size_t dimension : shape) {
        count *= dimension;
    }

    return count;
}

std::vector<std::size_t> strides(
    const std::vector<std::size_t>& shape
) {
    std::vector<std::size_t> result(shape.size(), 1);

    for (std::size_t i = shape.size(); i > 1; --i) {
        result[i - 2] = result[i - 1] * shape[i - 1];
    }

    return result;
}

std::vector<std::size_t> broadcast_shape(
    const std::vector<std::size_t>& lhs,
    const std::vector<std::size_t>& rhs
) {
    const std::size_t rank =
        lhs.size() > rhs.size() ? lhs.size() : rhs.size();

    std::vector<std::size_t> result(rank, 1);

    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t lhs_offset = rank - lhs.size();
        const std::size_t rhs_offset = rank - rhs.size();

        const std::size_t lhs_dim =
            i < lhs_offset ? 1 : lhs[i - lhs_offset];

        const std::size_t rhs_dim =
            i < rhs_offset ? 1 : rhs[i - rhs_offset];

        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw std::invalid_argument(
                "Incompatible tensor shapes for broadcasting"
            );
        }

        result[i] = lhs_dim > rhs_dim ? lhs_dim : rhs_dim;
    }

    return result;
}

template <typename Function>
Tensor elementwise_binary(
    const Tensor& lhs,
    const Tensor& rhs,
    Function function
) {
    const auto output_shape =
        broadcast_shape(lhs.shape(), rhs.shape());

    Tensor result(output_shape, 0.0F);

    const std::size_t total = result.size();

    if (total == 0) {
        return result;
    }

    const float* left = lhs.data().data();
    const float* right = rhs.data().data();
    float* __restrict output = result.data().data();

    // Fast path: identical shapes -> one flat vectorized loop.
    if (lhs.shape() == rhs.shape()) {
        #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (total > kParallelMin)
        for (std::size_t i = 0; i < total; ++i) {
            output[i] = function(left[i], right[i]);
        }

        return result;
    }

    const std::size_t rank = output_shape.size();

    if (rank == 0) {
        output[0] = function(left[0], right[0]);
        return result;
    }

    // General broadcasting. The original recomputed a division and modulo
    // per axis, per element, per operand. Instead each input gets an
    // "effective stride" per output axis (0 where it is broadcast), the
    // outer coordinates are decoded once per row, and the last axis is a
    // tight inner loop.
    const auto lhs_strides = strides(lhs.shape());
    const auto rhs_strides = strides(rhs.shape());

    std::vector<std::size_t> lhs_eff(rank, 0);
    std::vector<std::size_t> rhs_eff(rank, 0);

    const std::size_t lhs_align = rank - lhs.shape().size();
    const std::size_t rhs_align = rank - rhs.shape().size();

    for (std::size_t a = lhs_align; a < rank; ++a) {
        const std::size_t ia = a - lhs_align;
        lhs_eff[a] = lhs.shape()[ia] == 1 ? 0 : lhs_strides[ia];
    }

    for (std::size_t a = rhs_align; a < rank; ++a) {
        const std::size_t ia = a - rhs_align;
        rhs_eff[a] = rhs.shape()[ia] == 1 ? 0 : rhs_strides[ia];
    }

    const std::size_t inner = output_shape[rank - 1];
    const std::size_t outer = total / inner;
    const std::size_t lhs_inner_stride = lhs_eff[rank - 1];
    const std::size_t rhs_inner_stride = rhs_eff[rank - 1];

    const std::size_t* out_dims = output_shape.data();
    const std::size_t* le = lhs_eff.data();
    const std::size_t* re = rhs_eff.data();

    // Each outer row o writes output[o * inner, (o + 1) * inner) only,
    // so there are no races.
    #pragma omp parallel for schedule(static) num_threads(kThreads) if (total > kParallelMin)
    for (std::size_t o = 0; o < outer; ++o) {
        std::size_t remaining = o;
        std::size_t lhs_base = 0;
        std::size_t rhs_base = 0;

        for (std::size_t a = rank - 1; a-- > 0;) {
            const std::size_t coordinate = remaining % out_dims[a];
            remaining /= out_dims[a];

            lhs_base += coordinate * le[a];
            rhs_base += coordinate * re[a];
        }

        const float* l = left + lhs_base;
        const float* r = right + rhs_base;
        float* __restrict out = output + o * inner;

        if (lhs_inner_stride != 0 && rhs_inner_stride != 0) {
            #pragma omp simd
            for (std::size_t j = 0; j < inner; ++j) {
                out[j] = function(l[j], r[j]);
            }
        } else if (lhs_inner_stride != 0) {
            const float rv = r[0];

            #pragma omp simd
            for (std::size_t j = 0; j < inner; ++j) {
                out[j] = function(l[j], rv);
            }
        } else if (rhs_inner_stride != 0) {
            const float lv = l[0];

            #pragma omp simd
            for (std::size_t j = 0; j < inner; ++j) {
                out[j] = function(lv, r[j]);
            }
        } else {
            const float value = function(l[0], r[0]);

            #pragma omp simd
            for (std::size_t j = 0; j < inner; ++j) {
                out[j] = value;
            }
        }
    }

    return result;
}

std::vector<std::size_t> batch_shape(
    const Tensor& tensor
) {
    const auto& shape = tensor.shape();

    if (tensor.rank() == 1) {
        return {};
    }

    return std::vector<std::size_t>(
        shape.begin(),
        shape.end() - 2
    );
}

std::size_t batch_offset(
    const std::vector<std::size_t>& input_batch_shape,
    const std::vector<std::size_t>& output_batch_shape,
    std::size_t output_batch_index
) {
    if (input_batch_shape.empty()) {
        return 0;
    }

    const std::size_t alignment =
        output_batch_shape.size() - input_batch_shape.size();

    std::size_t input_batch_index = 0;

    for (std::size_t i = 0; i < input_batch_shape.size(); ++i) {
        const std::size_t output_axis = alignment + i;

        std::size_t stride = 1;

        for (
            std::size_t j = output_axis + 1;
            j < output_batch_shape.size();
            ++j
        ) {
            stride *= output_batch_shape[j];
        }

        const std::size_t coordinate =
            (output_batch_index / stride)
            % output_batch_shape[output_axis];

        const std::size_t input_coordinate =
            input_batch_shape[i] == 1 ? 0 : coordinate;

        input_batch_index =
            input_batch_index * input_batch_shape[i]
            + input_coordinate;
    }

    return input_batch_index;
}

} // namespace

Tensor operator+(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a + b; }
    );
}

Tensor operator-(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a - b; }
    );
}

Tensor operator*(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a * b; }
    );
}

Tensor operator/(
    const Tensor& lhs,
    const Tensor& rhs
) {
    return elementwise_binary(
        lhs, rhs,
        [](float a, float b) { return a / b; }
    );
}

Tensor operator-(const Tensor& input) {
    Tensor result(input.shape(), 0.0F);

    const std::size_t n = input.size();
    const float* __restrict values = input.data().data();
    float* __restrict output = result.data().data();

    #pragma omp parallel for simd num_threads(kThreads) schedule(static) if (n > kParallelMin)
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = -values[i];
    }

    return result;
}

Tensor transpose(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument(
            "transpose: input must be a rank-2 tensor"
        );
    }

    const std::size_t rows = input.shape()[0];
    const std::size_t cols = input.shape()[1];

    Tensor result({cols, rows}, 0.0F);

    const float* __restrict values = input.data().data();
    float* __restrict output = result.data().data();

    const std::size_t row_tiles = (rows + kTile - 1) / kTile;
    const std::size_t col_tiles = (cols + kTile - 1) / kTile;

    // Tiled so both the reads and the strided writes stay inside cache.
    // Each output element belongs to exactly one tile -> no races.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(kThreads) if (rows * cols > kParallelMin)
    for (std::size_t ti = 0; ti < row_tiles; ++ti) {
        for (std::size_t tj = 0; tj < col_tiles; ++tj) {
            const std::size_t i0 = ti * kTile;
            const std::size_t j0 = tj * kTile;
            const std::size_t i1 = i0 + kTile < rows ? i0 + kTile : rows;
            const std::size_t j1 = j0 + kTile < cols ? j0 + kTile : cols;

            for (std::size_t i = i0; i < i1; ++i) {
                for (std::size_t j = j0; j < j1; ++j) {
                    output[j * rows + i] = values[i * cols + j];
                }
            }
        }
    }

    return result;
}

Tensor matmul(
    const Tensor& lhs,
    const Tensor& rhs
) {
    const std::size_t lhs_rank = lhs.rank();
    const std::size_t rhs_rank = rhs.rank();

    if (
        lhs_rank < 1 || lhs_rank > 5
        || rhs_rank < 1 || rhs_rank > 5
    ) {
        throw std::invalid_argument(
            "matmul: input ranks must be between 1 and 5"
        );
    }

    const auto& lhs_shape = lhs.shape();
    const auto& rhs_shape = rhs.shape();

    const std::size_t m =
        lhs_rank == 1 ? 1 : lhs_shape[lhs_rank - 2];

    const std::size_t lhs_k = lhs_shape[lhs_rank - 1];

    const std::size_t rhs_k =
        rhs_rank == 1 ? rhs_shape[0] : rhs_shape[rhs_rank - 2];

    const std::size_t n =
        rhs_rank == 1 ? 1 : rhs_shape[rhs_rank - 1];

    if (lhs_k != rhs_k) {
        throw std::invalid_argument(
            "matmul: inner dimensions do not match"
        );
    }

    const auto lhs_batch_shape = batch_shape(lhs);
    const auto rhs_batch_shape = batch_shape(rhs);

    const auto output_batch_shape =
        broadcast_shape(lhs_batch_shape, rhs_batch_shape);

    std::vector<std::size_t> output_shape = output_batch_shape;

    if (lhs_rank > 1) {
        output_shape.push_back(m);
    }

    if (rhs_rank > 1) {
        output_shape.push_back(n);
    }

    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    Tensor result(output_shape, 0.0F);

    const float* left = lhs.data().data();
    const float* right = rhs.data().data();
    float* output = result.data().data();

    const std::size_t batch_count =
        element_count(output_batch_shape);

    const std::size_t lhs_matrix_size = m * lhs_k;
    const std::size_t rhs_matrix_size = rhs_k * n;
    const std::size_t output_matrix_size = m * n;

    // Batch offsets computed once up front instead of inside the hot loop.
    std::vector<std::size_t> lhs_bases(batch_count);
    std::vector<std::size_t> rhs_bases(batch_count);

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        lhs_bases[batch] =
            batch_offset(lhs_batch_shape, output_batch_shape, batch)
            * lhs_matrix_size;

        rhs_bases[batch] =
            batch_offset(rhs_batch_shape, output_batch_shape, batch)
            * rhs_matrix_size;
    }

    const std::size_t K = lhs_k;
    const std::size_t row_blocks = (m + kRowBlock - 1) / kRowBlock;
    const bool par = batch_count * m * n * K > kParallelMatmulWork;

    // Parallel over (batch, block of 4 output rows). Every output row is
    // owned by exactly one thread -> no races, no atomics.
    // Inside a block the loop order is still i-k-j (j contiguous), but each
    // rhs row is loaded once and used for 4 output rows.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(kThreads) if (par)
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        for (std::size_t rb = 0; rb < row_blocks; ++rb) {
            const float* L = left + lhs_bases[batch];
            const float* R = right + rhs_bases[batch];
            float* O = output + batch * output_matrix_size;

            const std::size_t i0 = rb * kRowBlock;
            const std::size_t i1 =
                i0 + kRowBlock < m ? i0 + kRowBlock : m;

            if (n == 1) {
                // Matrix-vector: each output is a dot product.
                for (std::size_t i = i0; i < i1; ++i) {
                    const float* __restrict lr = L + i * K;
                    float sum = 0.0F;

                    #pragma omp simd reduction(+ : sum)
                    for (std::size_t k = 0; k < K; ++k) {
                        sum += lr[k] * R[k];
                    }

                    O[i] = sum;
                }
            } else if (i1 - i0 == kRowBlock) {
                const float* __restrict l0 = L + (i0 + 0) * K;
                const float* __restrict l1 = L + (i0 + 1) * K;
                const float* __restrict l2 = L + (i0 + 2) * K;
                const float* __restrict l3 = L + (i0 + 3) * K;

                float* __restrict o0 = O + (i0 + 0) * n;
                float* __restrict o1 = O + (i0 + 1) * n;
                float* __restrict o2 = O + (i0 + 2) * n;
                float* __restrict o3 = O + (i0 + 3) * n;

                for (std::size_t k = 0; k < K; ++k) {
                    const float a0 = l0[k];
                    const float a1 = l1[k];
                    const float a2 = l2[k];
                    const float a3 = l3[k];

                    const float* __restrict r = R + k * n;

                    #pragma omp simd
                    for (std::size_t j = 0; j < n; ++j) {
                        const float rv = r[j];

                        o0[j] += a0 * rv;
                        o1[j] += a1 * rv;
                        o2[j] += a2 * rv;
                        o3[j] += a3 * rv;
                    }
                }
            } else {
                // Leftover rows (m not a multiple of 4): original loop.
                for (std::size_t i = i0; i < i1; ++i) {
                    float* __restrict o = O + i * n;

                    for (std::size_t k = 0; k < K; ++k) {
                        const float a = L[i * K + k];
                        const float* __restrict r = R + k * n;

                        #pragma omp simd
                        for (std::size_t j = 0; j < n; ++j) {
                            o[j] += a * r[j];
                        }
                    }
                }
            }
        }
    }

    return result;
}

} // namespace neural