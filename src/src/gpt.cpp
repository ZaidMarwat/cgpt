#include "gpt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

void add_bias_inplace(Tensor& m, const Tensor& b) {
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++) m.at(i, j) += b.at(0, j);
}

Tensor colsum(const Tensor& m) {
    Tensor out(1, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++) out.at(0, j) += m.at(i, j);
    return out;
}

void randn(Tensor& t, std::mt19937& rng, float std) {
    std::normal_distribution<float> d(0.0f, std);
    for (auto& v : t.data) v = d(rng);
}

struct LayerCache {
    Tensor x;        // block input, (T, d)
    Tensor xn1;      // after ln1
    Tensor qkv;      // (T, 3d)
    Tensor attn;     // concatenated head outputs, (T, d)
    Tensor x2;       // x + attention projection
    Tensor xn2;      // after ln2
    Tensor h1;       // MLP pre-activation, (T, 4d)
    Tensor a1;       // post-GELU
    std::vector<Tensor> probs;  // per head, (T, T) post-mask softmax
};

struct LayerForward {
    Tensor out;
    LayerCache cache;
};

LayerForward layer_forward(const LayerParams& p, const Tensor& x, int n_head) {
    const int T = x.rows, d = x.cols, dh = d / n_head;
    LayerCache c;
    c.x = x;
    c.xn1 = layernorm_rows(x, p.ln1_g.w, p.ln1_b.w);
    c.qkv = matmul(c.xn1, p.wqkv.w);
    add_bias_inplace(c.qkv, p.bqkv.w);

    c.attn = Tensor(T, d);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    for (int h = 0; h < n_head; h++) {
        Tensor q = get_cols(c.qkv, h * dh, (h + 1) * dh);
        Tensor k = get_cols(c.qkv, d + h * dh, d + (h + 1) * dh);
        Tensor v = get_cols(c.qkv, 2 * d + h * dh, 2 * d + (h + 1) * dh);
        Tensor scores = scalar_mul(matmul(q, transpose(k)), scale);
        for (int i = 0; i < T; i++)
            for (int j = i + 1; j < T; j++) scores.at(i, j) = -1e30f;  // causal mask
        Tensor probs = softmax_rows(scores);
        Tensor head_out = matmul(probs, v);
        set_cols(c.attn, h * dh, head_out);
        c.probs.push_back(std::move(probs));
    }

    Tensor proj = matmul(c.attn, p.wo.w);
    add_bias_inplace(proj, p.bo.w);
    c.x2 = add(x, proj);

    c.xn2 = layernorm_rows(c.x2, p.ln2_g.w, p.ln2_b.w);
    c.h1 = matmul(c.xn2, p.w1.w);
    add_bias_inplace(c.h1, p.b1.w);
    c.a1 = gelu(c.h1);
    Tensor h2 = matmul(c.a1, p.w2.w);
    add_bias_inplace(h2, p.b2.w);

    LayerForward f;
    f.out = add(c.x2, h2);
    f.cache = std::move(c);
    return f;
}

// Accumulates gradients into p.*.g and returns d(input).
Tensor layer_backward(LayerParams& p, const LayerCache& c, const Tensor& d_out, int n_head) {
    const int T = c.x.rows, d = c.x.cols, dh = d / n_head;

    // out = x2 + h2
    Tensor d_x2 = d_out;
    Tensor d_h2 = d_out;

    // h2 = a1 @ w2 + b2
    Tensor d_a1(T, 4 * d), d_h1, d_xn2;
    matmul_backward(c.a1, p.w2.w, d_h2, d_a1, p.w2.g);
    add_inplace(p.b2.g, colsum(d_h2));

    // a1 = gelu(h1)
    d_h1 = Tensor(T, 4 * d);
    gelu_backward(c.h1, d_a1, d_h1);

    // h1 = xn2 @ w1 + b1
    d_xn2 = Tensor(T, d);
    matmul_backward(c.xn2, p.w1.w, d_h1, d_xn2, p.w1.g);
    add_inplace(p.b1.g, colsum(d_h1));

    // xn2 = ln(x2); x2 also feeds the residual
    Tensor d_x(T, d);
    layernorm_rows_backward(c.x2, p.ln2_g.w, Tensor(), d_xn2, d_x2, p.ln2_g.g, p.ln2_b.g);
    // d_x2 now holds d(ln input) += flow into proj path; residual flow added below.

    // proj = attn @ wo + bo ; d_proj = d_x2 (residual branch of out's second input)
    Tensor d_proj = d_x2;
    Tensor d_attn(T, d);
    matmul_backward(c.attn, p.wo.w, d_proj, d_attn, p.wo.g);
    add_inplace(p.bo.g, colsum(d_proj));

    // per-head attention backward
    Tensor d_qkv(T, 3 * d);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    for (int h = 0; h < n_head; h++) {
        Tensor d_head = get_cols(d_attn, h * dh, (h + 1) * dh);
        Tensor q = get_cols(c.qkv, h * dh, (h + 1) * dh);
        Tensor k = get_cols(c.qkv, d + h * dh, d + (h + 1) * dh);
        Tensor v = get_cols(c.qkv, 2 * d + h * dh, 2 * d + (h + 1) * dh);
        const Tensor& probs = c.probs[static_cast<size_t>(h)];

        Tensor d_v = matmul(transpose(probs), d_head);
        acc_cols(d_qkv, 2 * d + h * dh, d_v);

        Tensor d_probs = matmul(d_head, transpose(v));
        Tensor d_scores(T, T);
        softmax_rows_backward(probs, d_probs, d_scores);  // masked entries keep p=0, grad 0
        d_scores = scalar_mul(d_scores, scale);

        Tensor d_q = matmul(d_scores, k);
        acc_cols(d_qkv, h * dh, d_q);
        Tensor d_k = matmul(transpose(d_scores), q);
        acc_cols(d_qkv, d + h * dh, d_k);
    }

    // qkv = xn1 @ wqkv + bqkv
    Tensor d_xn1(T, d);
    matmul_backward(c.xn1, p.wqkv.w, d_qkv, d_xn1, p.wqkv.g);
    add_inplace(p.bqkv.g, colsum(d_qkv));

    // xn1 = ln(x); residual from x2 = x + proj adds d_x2
    layernorm_rows_backward(c.x, p.ln1_g.w, Tensor(), d_xn1, d_x, p.ln1_g.g, p.ln1_b.g);
    add_inplace(d_x, d_x2);
    return d_x;
}

}  // namespace

struct GPT::Cache {
    std::vector<int> inputs;
    Tensor x;                      // (T, d) after embeddings
    std::vector<LayerCache> lc;
    Tensor xf;                     // after final block
    Tensor xnf;                    // after final layernorm
    Tensor logits;                 // (T, vocab)
};

GPT::GPT(const GPTConfig& c, const std::vector<char>& vocab_chars, uint32_t seed)
    : cfg(c), id_to_ch(vocab_chars), rng(seed) {
    std::fill(ch_to_id, ch_to_id + 256, -1);
    for (size_t i = 0; i < id_to_ch.size(); i++)
        ch_to_id[static_cast<unsigned char>(id_to_ch[i])] = static_cast<int>(i);

    const int d = cfg.d_model;
    wte = Param(cfg.vocab, d);
    wpe = Param(cfg.block, d);
    randn(wte.w, rng, 0.02f);
    randn(wpe.w, rng, 0.02f);
    layers.resize(static_cast<size_t>(cfg.n_layer));
    for (auto& l : layers) {
        l.ln1_g = Param(1, d); l.ln1_g.w.fill(1.0f);
        l.ln1_b = Param(1, d);
        l.wqkv = Param(d, 3 * d); randn(l.wqkv.w, rng, 0.02f);
        l.bqkv = Param(1, 3 * d);
        l.wo = Param(d, d); randn(l.wo.w, rng, 0.02f);
        l.bo = Param(1, d);
        l.ln2_g = Param(1, d); l.ln2_g.w.fill(1.0f);
        l.ln2_b = Param(1, d);
        l.w1 = Param(d, 4 * d); randn(l.w1.w, rng, 0.02f);
        l.b1 = Param(1, 4 * d);
        l.w2 = Param(4 * d, d); randn(l.w2.w, rng, 0.02f);
        l.b2 = Param(1, d);
    }
    lnf_g = Param(1, d); lnf_g.w.fill(1.0f);
    lnf_b = Param(1, d);
}

float GPT::forward(const std::vector<int>& tokens, Cache& cache) {
    const int T = static_cast<int>(tokens.size());
    const int d = cfg.d_model;
    cache.inputs = tokens;
    cache.x = Tensor(T, d);
    for (int i = 0; i < T; i++)
        for (int j = 0; j < d; j++)
            cache.x.at(i, j) = wte.w.at(tokens[static_cast<size_t>(i)], j) + wpe.w.at(i, j);

    cache.lc.clear();
    Tensor x = cache.x;
    for (auto& l : layers) {
        LayerForward f = layer_forward(l, x, cfg.n_head);
        x = f.out;
        cache.lc.push_back(std::move(f.cache));
    }
    cache.xf = x;
    cache.xnf = layernorm_rows(x, lnf_g.w, lnf_b.w);
    cache.logits = matmul(cache.xnf, transpose(wte.w));  // tied head
    return 0.0f;  // loss computed by caller
}

void GPT::backward(const Cache& cache, const std::vector<int>& targets) {
    Tensor d_logits = cross_entropy_backward(cache.logits, targets);

    // logits = xnf @ wte^T, so d_wte = d_logits^T @ xnf (vocab x d)
    Tensor d_xnf = matmul(d_logits, wte.w);
    add_inplace(wte.g, matmul(transpose(d_logits), cache.xnf));  // head contribution

    Tensor d_x(cache.x.rows, cfg.d_model);
    layernorm_rows_backward(cache.xf, lnf_g.w, Tensor(), d_xnf, d_x, lnf_g.g, lnf_b.g);

    for (int li = cfg.n_layer - 1; li >= 0; li--) {
        Tensor d_in = layer_backward(layers[static_cast<size_t>(li)],
                                     cache.lc[static_cast<size_t>(li)], d_x, cfg.n_head);
        d_x = std::move(d_in);
    }

    // embedding backward: scatter-add rows
    for (int i = 0; i < cache.x.rows; i++) {
        int tok = cache.inputs[static_cast<size_t>(i)];
        for (int j = 0; j < cfg.d_model; j++) {
            wte.g.at(tok, j) += d_x.at(i, j);
            wpe.g.at(i, j) += d_x.at(i, j);
        }
    }
}

static void adam_one(Param& p, float lr, float b1, float b2, float eps, int t) {
    const float bc1 = 1.0f - std::pow(b1, t);
    const float bc2 = 1.0f - std::pow(b2, t);
    for (size_t i = 0; i < p.w.data.size(); i++) {
        float g = p.g.data[i];
        p.m.data[i] = b1 * p.m.data[i] + (1 - b1) * g;
        p.v.data[i] = b2 * p.v.data[i] + (1 - b2) * g * g;
        float mhat = p.m.data[i] / bc1;
        float vhat = p.v.data[i] / bc2;
        p.w.data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
    std::fill(p.g.data.begin(), p.g.data.end(), 0.0f);
}

float GPT::train_step(const std::vector<int>& tokens, float lr, int adam_t) {
    std::vector<int> inputs(tokens.begin(), tokens.end() - 1);
    std::vector<int> targets(tokens.begin() + 1, tokens.end());

    Cache cache;
    forward(inputs, cache);
    float loss = cross_entropy(cache.logits, targets);

    backward(cache, targets);

    // global-norm gradient clipping at 1.0
    std::vector<Param*> ps;
    ps.push_back(&wte); ps.push_back(&wpe); ps.push_back(&lnf_g); ps.push_back(&lnf_b);
    for (auto& l : layers) {
        ps.insert(ps.end(), {&l.ln1_g, &l.ln1_b, &l.wqkv, &l.bqkv, &l.wo, &l.bo,
                             &l.ln2_g, &l.ln2_b, &l.w1, &l.b1, &l.w2, &l.b2});
    }
    double norm2 = 0.0;
    for (auto* p : ps)
        for (float g : p->g.data) norm2 += static_cast<double>(g) * g;
    double norm = std::sqrt(norm2);
    if (norm > 1.0) {
        float s = static_cast<float>(1.0 / norm);
        for (auto* p : ps)
            for (auto& g : p->g.data) g *= s;
    }

    for (auto* p : ps) adam_one(*p, lr, 0.9f, 0.999f, 1e-8f, adam_t);
    return loss;
}

std::string GPT::generate(const std::string& prompt, int count, float temperature,
                          uint32_t seed) {
    std::mt19937 gen(seed);
    std::vector<int> ctx = encode(prompt);
    if (ctx.empty()) throw std::runtime_error("generate: empty prompt");
    if (static_cast<int>(ctx.size()) > cfg.block)
        ctx.erase(ctx.begin(), ctx.end() - cfg.block);

    std::vector<int> generated;
    generated.reserve(static_cast<size_t>(count));
    for (int n = 0; n < count; n++) {
        Cache cache;
        forward(ctx, cache);
        int last = static_cast<int>(ctx.size()) - 1;
        // temperature-scaled softmax over the last position's logits
        std::vector<float> p(static_cast<size_t>(cfg.vocab));
        float mx = -1e30f;
        for (int j = 0; j < cfg.vocab; j++)
            mx = std::max(mx, cache.logits.at(last, j) / temperature);
        float sum = 0.0f;
        for (int j = 0; j < cfg.vocab; j++) {
            p[static_cast<size_t>(j)] = std::exp(cache.logits.at(last, j) / temperature - mx);
            sum += p[static_cast<size_t>(j)];
        }
        std::uniform_real_distribution<float> u(0.0f, sum);
        float r = u(gen);
        int next = cfg.vocab - 1;
        float acc = 0.0f;
        for (int j = 0; j < cfg.vocab; j++) {
            acc += p[static_cast<size_t>(j)];
            if (r <= acc) { next = j; break; }
        }
        ctx.push_back(next);
        generated.push_back(next);
        if (static_cast<int>(ctx.size()) > cfg.block) ctx.erase(ctx.begin());
    }
    return decode(generated);
}

std::vector<int> GPT::encode(const std::string& s) const {
    std::vector<int> out;
    for (char ch : s) {
        int id = ch_to_id[static_cast<unsigned char>(ch)];
        if (id < 0) throw std::runtime_error(std::string("encode: character not in vocab: ") + ch);
        out.push_back(id);
    }
    return out;
}

std::string GPT::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += id_to_ch[static_cast<size_t>(id)];
    return out;
}

size_t GPT::param_count() const {
    size_t n = wte.w.data.size() + wpe.w.data.size() + lnf_g.w.data.size() + lnf_b.w.data.size();
    for (const auto& l : layers)
        n += l.ln1_g.w.data.size() + l.ln1_b.w.data.size() + l.wqkv.w.data.size() +
             l.bqkv.w.data.size() + l.wo.w.data.size() + l.bo.w.data.size() +
             l.ln2_g.w.data.size() + l.ln2_b.w.data.size() + l.w1.w.data.size() +
             l.b1.w.data.size() + l.w2.w.data.size() + l.b2.w.data.size();
    return n;
}

// ---------- serialization ----------

static void write_tensor(std::ofstream& f, const Tensor& t) {
    f.write(reinterpret_cast<const char*>(&t.rows), sizeof(int));
    f.write(reinterpret_cast<const char*>(&t.cols), sizeof(int));
    f.write(reinterpret_cast<const char*>(t.data.data()),
            static_cast<std::streamsize>(t.data.size() * sizeof(float)));
}

static Tensor read_tensor(std::ifstream& f) {
    Tensor t;
    f.read(reinterpret_cast<char*>(&t.rows), sizeof(int));
    f.read(reinterpret_cast<char*>(&t.cols), sizeof(int));
    t.data.resize(static_cast<size_t>(t.rows) * t.cols);
    f.read(reinterpret_cast<char*>(t.data.data()),
           static_cast<std::streamsize>(t.data.size() * sizeof(float)));
    return t;
}

void GPT::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save: cannot open " + path);
    f.write("CGPT1", 5);
    f.write(reinterpret_cast<const char*>(&cfg), sizeof(GPTConfig));
    int vs = static_cast<int>(id_to_ch.size());
    f.write(reinterpret_cast<const char*>(&vs), sizeof(int));
    f.write(id_to_ch.data(), vs);
    auto write_param = [&](const Param& p) { write_tensor(f, p.w); };
    write_param(wte); write_param(wpe);
    for (const auto& l : layers) {
        write_param(l.ln1_g); write_param(l.ln1_b); write_param(l.wqkv); write_param(l.bqkv);
        write_param(l.wo); write_param(l.bo); write_param(l.ln2_g); write_param(l.ln2_b);
        write_param(l.w1); write_param(l.b1); write_param(l.w2); write_param(l.b2);
    }
    write_param(lnf_g); write_param(lnf_b);
}

GPT GPT::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load: cannot open " + path);
    char magic[5];
    f.read(magic, 5);
    if (std::memcmp(magic, "CGPT1", 5) != 0) throw std::runtime_error("load: bad magic");
    GPTConfig c;
    f.read(reinterpret_cast<char*>(&c), sizeof(GPTConfig));
    int vs = 0;
    f.read(reinterpret_cast<char*>(&vs), sizeof(int));
    std::vector<char> vocab(static_cast<size_t>(vs));
    f.read(vocab.data(), vs);

    GPT g(c, vocab);
    auto read_param = [&](Param& p) {
        Tensor w = read_tensor(f);
        p = Param(1, 1);  // reset shapes
        p.w = w;
        p.g = Tensor(w.rows, w.cols);
        p.m = Tensor(w.rows, w.cols);
        p.v = Tensor(w.rows, w.cols);
    };
    read_param(g.wte); read_param(g.wpe);
    for (auto& l : g.layers) {
        read_param(l.ln1_g); read_param(l.ln1_b); read_param(l.wqkv); read_param(l.bqkv);
        read_param(l.wo); read_param(l.bo); read_param(l.ln2_g); read_param(l.ln2_b);
        read_param(l.w1); read_param(l.b1); read_param(l.w2); read_param(l.b2);
    }
    read_param(g.lnf_g); read_param(g.lnf_b);
    return g;
}
