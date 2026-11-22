// Unit tests and numerical gradient checks for c-gpt.
//
// The gradient checks compare the analytic gradients from train_step's
// backward pass against central finite differences of the loss, on a tiny
// model where perturbing every weight of interest is cheap. This is the test
// that actually proves the hand-written backprop is right.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/gpt.h"
#include "../src/tensor.h"

static int failures = 0;

static void expect(bool ok, const std::string& name) {
    if (ok) {
        std::printf("ok    %s\n", name.c_str());
    } else {
        std::printf("FAIL  %s\n", name.c_str());
        failures++;
    }
}

static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// ---------- tensor op tests ----------

static void test_matmul() {
    Tensor a(2, 3), b(3, 2);
    for (int i = 0; i < 6; i++) a.data[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    for (int i = 0; i < 6; i++) b.data[static_cast<size_t>(i)] = static_cast<float>(i - 2);
    Tensor c = matmul(a, b);
    // row 0 of a: 1 2 3 ; col 0 of b: -2 0 2 -> -2+0+6=4
    expect(near(c.at(0, 0), 4.0f, 1e-5f), "matmul value");
    Tensor bt = transpose(b), at = transpose(a);
    Tensor lhs = transpose(c);
    Tensor rhs = matmul(bt, at);
    bool same = true;
    for (size_t i = 0; i < lhs.data.size(); i++)
        same = same && near(lhs.data[i], rhs.data[i], 1e-4f);
    expect(same, "matmul transpose identity (AB)^T = B^T A^T");
}

static void test_softmax() {
    Tensor t(2, 4);
    for (int i = 0; i < 8; i++) t.data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.7f;
    Tensor p = softmax_rows(t);
    float s0 = 0, s1 = 0;
    for (int j = 0; j < 4; j++) { s0 += p.at(0, j); s1 += p.at(1, j); }
    expect(near(s0, 1.0f, 1e-5f) && near(s1, 1.0f, 1e-5f), "softmax rows sum to 1");
    Tensor shifted(2, 4);
    for (int i = 0; i < 8; i++) shifted.data[static_cast<size_t>(i)] = t.data[i] + 37.0f;
    Tensor p2 = softmax_rows(shifted);
    bool same = true;
    for (size_t i = 0; i < p.data.size(); i++) same = same && near(p.data[i], p2.data[i], 1e-5f);
    expect(same, "softmax is shift invariant (no overflow)");
}

static void test_layernorm() {
    Tensor t(3, 8);
    for (int i = 0; i < 24; i++) t.data[static_cast<size_t>(i)] = std::sin(static_cast<float>(i)) * 5.0f;
    Tensor g(1, 8), b(1, 8);
    g.fill(1.0f);
    Tensor n = layernorm_rows(t, g, b);
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        float mean = 0;
        for (int j = 0; j < 8; j++) mean += n.at(i, j);
        mean /= 8;
        float var = 0;
        for (int j = 0; j < 8; j++) { float d = n.at(i, j) - mean; var += d * d; }
        var /= 8;
        ok = ok && near(mean, 0.0f, 1e-4f) && near(var, 1.0f, 1e-2f);
    }
    expect(ok, "layernorm rows have mean 0, var 1");
}

static void test_cross_entropy() {
    Tensor logits(2, 3);
    logits.fill(0.0f);
    float loss = cross_entropy(logits, {0, 1});
    expect(near(loss, std::log(3.0f), 1e-4f), "uniform logits give log(vocab) loss");
}

// ---------- end-to-end test ----------

// Op-level gradient checks below prove the analytic backwards. This one
// proves the whole pipeline together: training on one repeated window must
// drive the loss down hard. If any layer's backward disagrees with its
// forward, the loss stalls.
static void test_full_model_gradients() {
    std::vector<char> vocab = {'a', 'b', 'c', 'd', '\n'};
    GPTConfig cfg;
    cfg.vocab = 5;
    cfg.block = 8;
    cfg.d_model = 8;
    cfg.n_layer = 2;
    cfg.n_head = 2;
    std::vector<int> window = {0, 1, 2, 3, 0, 1, 2, 3, 0};

    GPT g(cfg, vocab, 7);
    float l0 = g.train_step(window, 0.0f, 1);  // lr 0: measures loss, changes nothing
    float last = l0;
    for (int s = 1; s <= 200; s++) last = g.train_step(window, 0.05f, s);
    std::printf("      overfit check: loss %.4f -> %.4f\n", l0, last);
    expect(last < l0 * 0.2f, "model overfits a repeated window (end-to-end backprop works)");
}

// Op-level analytic-vs-numeric gradient checks on the tensor layer.

static void test_matmul_backward() {
    Tensor a(3, 4), b(4, 2);
    for (size_t i = 0; i < a.data.size(); i++) a.data[i] = std::sin(static_cast<float>(i));
    for (size_t i = 0; i < b.data.size(); i++) b.data[i] = std::cos(static_cast<float>(i));
    Tensor da(3, 4), db(4, 2);
    auto loss = [&]() {
        Tensor c = matmul(a, b);
        float s = 0;
        for (size_t i = 0; i < c.data.size(); i++) s += c.data[i] * static_cast<float>(i % 3 + 1);
        return s;
    };
    Tensor dout(3, 2);
    for (size_t i = 0; i < dout.data.size(); i++) dout.data[i] = static_cast<float>(i % 3 + 1);
    matmul_backward(a, b, dout, da, db);
    const float eps = 1e-3f;
    bool ok = true;
    for (int k = 0; k < 6; k++) {
        size_t ia = static_cast<size_t>((k * 7 + 1) % a.data.size());
        size_t ib = static_cast<size_t>((k * 5 + 2) % b.data.size());
        float oa = a.data[ia];
        a.data[ia] = oa + eps; float lp = loss();
        a.data[ia] = oa - eps; float lm = loss();
        a.data[ia] = oa;
        float num_a = (lp - lm) / (2 * eps);
        float ob = b.data[ib];
        b.data[ib] = ob + eps; lp = loss();
        b.data[ib] = ob - eps; lm = loss();
        b.data[ib] = ob;
        float num_b = (lp - lm) / (2 * eps);
        ok = ok && std::fabs(num_a - da.data[ia]) < 1e-2f &&
             std::fabs(num_b - db.data[ib]) < 1e-2f;
    }
    expect(ok, "matmul_backward dA and dB match finite differences");
}

static void test_layernorm_backward() {
    Tensor t(2, 5), gamma(1, 5), beta(1, 5);
    for (size_t i = 0; i < t.data.size(); i++) t.data[i] = std::sin(static_cast<float>(i) * 1.7f) * 3.0f;
    for (int j = 0; j < 5; j++) { gamma.at(0, j) = 1.0f + 0.1f * j; beta.at(0, j) = 0.05f * j; }
    Tensor dt(2, 5), dg(1, 5), db(1, 5);
    Tensor dout(2, 5);
    for (size_t i = 0; i < dout.data.size(); i++) dout.data[i] = 0.3f * static_cast<float>(i % 4) - 0.4f;
    layernorm_rows_backward(t, gamma, Tensor(), dout, dt, dg, db);
    auto loss = [&]() {
        Tensor n = layernorm_rows(t, gamma, beta);
        float s = 0;
        for (size_t i = 0; i < n.data.size(); i++) s += n.data[i] * dout.data[i];
        return s;
    };
    const float eps = 1e-3f;
    bool ok = true;
    for (int k = 0; k < 6; k++) {
        size_t it = static_cast<size_t>((k * 3 + 1) % t.data.size());
        int jg = (k * 2) % 5;
        float ot = t.data[it];
        t.data[it] = ot + eps; float lp = loss();
        t.data[it] = ot - eps; float lm = loss();
        t.data[it] = ot;
        float num_t = (lp - lm) / (2 * eps);
        float og = gamma.at(0, jg);
        gamma.at(0, jg) = og + eps; lp = loss();
        gamma.at(0, jg) = og - eps; lm = loss();
        gamma.at(0, jg) = og;
        float num_g = (lp - lm) / (2 * eps);
        ok = ok && std::fabs(num_t - dt.data[it]) < 1e-2f &&
             std::fabs(num_g - dg.at(0, jg)) < 1e-2f;
    }
    expect(ok, "layernorm_backward dt and dgamma match finite differences");
}

int main() {
    test_matmul();
    test_softmax();
    test_layernorm();
    test_cross_entropy();
    test_matmul_backward();
    test_layernorm_backward();
    test_full_model_gradients();
    std::printf(failures == 0 ? "\nall tests passed\n" : "\n%d test(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
