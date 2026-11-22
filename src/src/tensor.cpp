#include "tensor.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

static void check(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
}

void Tensor::print() const {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            std::printf("%8.4f%s", at(r, c), c + 1 < cols ? " " : "\n");
        }
    }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    check(a.cols == b.rows, "matmul: inner dimensions do not match");
    Tensor out(a.rows, b.cols);
    // ikj order keeps the inner loop walking contiguous memory in both b and out.
    for (int i = 0; i < a.rows; i++) {
        for (int k = 0; k < a.cols; k++) {
            const float aik = a.at(i, k);
            if (aik == 0.0f) continue;
            const float* brow = &b.data[static_cast<size_t>(k) * b.cols];
            float* orow = &out.data[static_cast<size_t>(i) * out.cols];
            for (int j = 0; j < b.cols; j++) orow[j] += aik * brow[j];
        }
    }
    return out;
}

Tensor transpose(const Tensor& t) {
    Tensor out(t.cols, t.rows);
    for (int i = 0; i < t.rows; i++)
        for (int j = 0; j < t.cols; j++) out.at(j, i) = t.at(i, j);
    return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
    check(a.rows == b.rows && a.cols == b.cols, "add: shapes do not match");
    Tensor out(a.rows, a.cols);
    for (size_t i = 0; i < a.data.size(); i++) out.data[i] = a.data[i] + b.data[i];
    return out;
}

void add_inplace(Tensor& a, const Tensor& b) {
    check(a.rows == b.rows && a.cols == b.cols, "add_inplace: shapes do not match");
    for (size_t i = 0; i < a.data.size(); i++) a.data[i] += b.data[i];
}

Tensor scalar_mul(const Tensor& t, float s) {
    Tensor out(t.rows, t.cols);
    for (size_t i = 0; i < t.data.size(); i++) out.data[i] = t.data[i] * s;
    return out;
}

Tensor softmax_rows(const Tensor& t) {
    Tensor out(t.rows, t.cols);
    for (int i = 0; i < t.rows; i++) {
        const float* row = &t.data[static_cast<size_t>(i) * t.cols];
        float* orow = &out.data[static_cast<size_t>(i) * out.cols];
        float mx = *std::max_element(row, row + t.cols);
        float sum = 0.0f;
        for (int j = 0; j < t.cols; j++) { orow[j] = std::exp(row[j] - mx); sum += orow[j]; }
        for (int j = 0; j < t.cols; j++) orow[j] /= sum;
    }
    return out;
}

Tensor layernorm_rows(const Tensor& t, const Tensor& gamma, const Tensor& beta, float eps) {
    check(gamma.rows == 1 && beta.rows == 1 && gamma.cols == t.cols && beta.cols == t.cols,
          "layernorm: gamma/beta must be 1 x cols");
    Tensor out(t.rows, t.cols);
    for (int i = 0; i < t.rows; i++) {
        const float* row = &t.data[static_cast<size_t>(i) * t.cols];
        float mean = 0.0f;
        for (int j = 0; j < t.cols; j++) mean += row[j];
        mean /= t.cols;
        float var = 0.0f;
        for (int j = 0; j < t.cols; j++) { float d = row[j] - mean; var += d * d; }
        var /= t.cols;
        float inv = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < t.cols; j++)
            out.at(i, j) = (row[j] - mean) * inv * gamma.at(0, j) + beta.at(0, j);
    }
    return out;
}

static float gelu_one(float x) {
    // tanh approximation, same as GPT-2.
    float x3 = x * x * x;
    float inner = 0.79788456080286536f * (x + 0.044715f * x3);  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(inner));
}

Tensor gelu(const Tensor& t) {
    Tensor out(t.rows, t.cols);
    for (size_t i = 0; i < t.data.size(); i++) out.data[i] = gelu_one(t.data[i]);
    return out;
}

Tensor get_cols(const Tensor& t, int c0, int c1) {
    check(c0 >= 0 && c1 <= t.cols && c0 < c1, "get_cols: bad range");
    Tensor out(t.rows, c1 - c0);
    for (int i = 0; i < t.rows; i++)
        for (int j = c0; j < c1; j++) out.at(i, j - c0) = t.at(i, j);
    return out;
}

void set_cols(Tensor& dst, int c0, const Tensor& src) {
    check(c0 >= 0 && c0 + src.cols <= dst.cols && dst.rows == src.rows, "set_cols: bad range");
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) dst.at(i, c0 + j) = src.at(i, j);
}

void acc_cols(Tensor& dst, int c0, const Tensor& src) {
    check(c0 >= 0 && c0 + src.cols <= dst.cols && dst.rows == src.rows, "acc_cols: bad range");
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) dst.at(i, c0 + j) += src.at(i, j);
}

float cross_entropy(const Tensor& logits, const std::vector<int>& targets) {
    check(static_cast<int>(targets.size()) == logits.rows, "cross_entropy: one target per row");
    Tensor probs = softmax_rows(logits);
    float loss = 0.0f;
    for (int i = 0; i < logits.rows; i++) {
        int t = targets[static_cast<size_t>(i)];
        check(t >= 0 && t < logits.cols, "cross_entropy: target out of range");
        loss -= std::log(std::max(probs.at(i, t), 1e-12f));
    }
    return loss / logits.rows;
}

// ---------------- backwards ----------------

void matmul_backward(const Tensor& a, const Tensor& b, const Tensor& d_out,
                     Tensor& d_a, Tensor& d_b) {
    // d_a += d_out @ b^T ; d_b += a^T @ d_out
    add_inplace(d_a, matmul(d_out, transpose(b)));
    add_inplace(d_b, matmul(transpose(a), d_out));
}

void scalar_mul_backward(const Tensor& d_out, float s, Tensor& d_t) {
    for (size_t i = 0; i < d_t.data.size(); i++) d_t.data[i] += d_out.data[i] * s;
}

void softmax_rows_backward(const Tensor& probs, const Tensor& d_out, Tensor& d_t) {
    for (int i = 0; i < probs.rows; i++) {
        float dot = 0.0f;
        for (int j = 0; j < probs.cols; j++) dot += d_out.at(i, j) * probs.at(i, j);
        for (int j = 0; j < probs.cols; j++)
            d_t.at(i, j) += probs.at(i, j) * (d_out.at(i, j) - dot);
    }
}

void layernorm_rows_backward(const Tensor& t, const Tensor& gamma, const Tensor& normed_unused,
                             const Tensor& d_out, Tensor& d_t, Tensor& d_gamma,
                             Tensor& d_beta, float eps) {
    (void)normed_unused;
    const int C = t.cols;
    for (int i = 0; i < t.rows; i++) {
        const float* row = &t.data[static_cast<size_t>(i) * C];
        float mean = 0.0f;
        for (int j = 0; j < C; j++) mean += row[j];
        mean /= C;
        float var = 0.0f;
        for (int j = 0; j < C; j++) { float d = row[j] - mean; var += d * d; }
        var /= C;
        float inv = 1.0f / std::sqrt(var + eps);

        // normed_j = (x_j - mean) * inv ; dy_j given.
        // per-row reductions for d(x_j):
        float s1 = 0.0f, s2 = 0.0f;  // sum(dy*g), sum(dy*g*normed)
        for (int j = 0; j < C; j++) {
            float nrm = (row[j] - mean) * inv;
            float dyg = d_out.at(i, j) * gamma.at(0, j);
            s1 += dyg;
            s2 += dyg * nrm;
        }
        for (int j = 0; j < C; j++) {
            float nrm = (row[j] - mean) * inv;
            d_t.at(i, j) += inv / C * (C * d_out.at(i, j) * gamma.at(0, j) - s1 - nrm * s2);
            d_gamma.at(0, j) += d_out.at(i, j) * nrm;
            d_beta.at(0, j) += d_out.at(i, j);
        }
    }
}

void gelu_backward(const Tensor& t, const Tensor& d_out, Tensor& d_t) {
    for (size_t i = 0; i < t.data.size(); i++) {
        float x = t.data[i];
        float x3 = x * x * x;
        float inner = 0.79788456080286536f * (x + 0.044715f * x3);
        float th = std::tanh(inner);
        float sech2 = 1.0f - th * th;
        float g = 0.5f * (1.0f + th) +
                  0.5f * x * sech2 * 0.79788456080286536f * (1.0f + 3.0f * 0.044715f * x * x);
        d_t.data[i] += d_out.data[i] * g;
    }
}

Tensor cross_entropy_backward(const Tensor& logits, const std::vector<int>& targets) {
    Tensor d = softmax_rows(logits);
    for (int i = 0; i < logits.rows; i++) d.at(i, targets[static_cast<size_t>(i)]) -= 1.0f;
    float scale = 1.0f / logits.rows;
    for (auto& v : d.data) v *= scale;
    return d;
}
