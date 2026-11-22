#pragma once

#include <cmath>
#include <vector>

// Dense 2-D tensor over a flat row-major buffer.
//
// This class grew out of the repo's original tensor.cpp (kept in git history):
// same name and same core operations (add, multiply, softmax, transpose,
// scalar multiply), with the debug printing removed and the operations a
// small GPT needs added on top.
class Tensor {
public:
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    Tensor() = default;
    Tensor(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0f) {}

    float& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    float at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }

    void fill(float v) { std::fill(data.begin(), data.end(), v); }
    void print() const;
};

// ---------- forward ops (out of place unless marked inplace) ----------

Tensor matmul(const Tensor& a, const Tensor& b);           // (m,k) @ (k,n) -> (m,n)
Tensor transpose(const Tensor& t);
Tensor add(const Tensor& a, const Tensor& b);              // same shape
void add_inplace(Tensor& a, const Tensor& b);              // a += b
Tensor scalar_mul(const Tensor& t, float s);
Tensor softmax_rows(const Tensor& t);                      // softmax along each row
Tensor layernorm_rows(const Tensor& t, const Tensor& gamma, const Tensor& beta,
                      float eps = 1e-5f);                  // per row; gamma/beta are 1 x cols
Tensor gelu(const Tensor& t);                              // tanh approximation, elementwise

// Column-block views used by multi-head attention and the fused QKV projection.
Tensor get_cols(const Tensor& t, int c0, int c1);          // copy out columns [c0, c1)
void set_cols(Tensor& dst, int c0, const Tensor& src);     // overwrite cols starting at c0
void acc_cols(Tensor& dst, int c0, const Tensor& src);     // accumulate into cols starting at c0

// Row-wise softmax + cross-entropy against integer targets, averaged over rows.
// logits: (T, vocab); targets: T token ids. Returns mean loss.
float cross_entropy(const Tensor& logits, const std::vector<int>& targets);

// ---------- backward ops: accumulate parameter/input gradients ----------
// Every backward accumulates into its gradient outputs (+=), so callers can
// share gradient buffers across uses of the same weight.

void matmul_backward(const Tensor& a, const Tensor& b, const Tensor& d_out,
                     Tensor& d_a, Tensor& d_b);
void scalar_mul_backward(const Tensor& d_out, float s, Tensor& d_t);
void softmax_rows_backward(const Tensor& probs, const Tensor& d_out, Tensor& d_t);
void layernorm_rows_backward(const Tensor& t, const Tensor& gamma, const Tensor& normed,
                             const Tensor& d_out, Tensor& d_t, Tensor& d_gamma,
                             Tensor& d_beta, float eps = 1e-5f);
void gelu_backward(const Tensor& t, const Tensor& d_out, Tensor& d_t);

// Gradient of mean cross-entropy w.r.t. logits: (softmax(logits) - onehot(targets)) / T.
Tensor cross_entropy_backward(const Tensor& logits, const std::vector<int>& targets);
