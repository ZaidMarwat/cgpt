#pragma once

#include <random>
#include <string>
#include <vector>

#include "tensor.h"

// A small character-level GPT: token + positional embeddings, N pre-norm
// transformer blocks (causal multi-head self-attention + MLP), a final layer
// norm, and an output projection tied to the token embedding. Trained with
// plain manual backprop and Adam. No libraries beyond the C++ standard.

struct GPTConfig {
    int vocab = 0;     // number of distinct characters
    int block = 64;    // context length
    int d_model = 64;  // embedding width
    int n_layer = 2;
    int n_head = 4;
};

struct Param {
    Tensor w;  // weights
    Tensor g;  // gradients, accumulated during backward, zeroed each step
    Tensor m, v;  // Adam moments
    Param() = default;
    Param(int r, int c) : w(r, c), g(r, c), m(r, c), v(r, c) {}
};

struct LayerParams {
    Param ln1_g, ln1_b;
    Param wqkv, bqkv;
    Param wo, bo;
    Param ln2_g, ln2_b;
    Param w1, b1;
    Param w2, b2;
};

class GPT {
public:
    GPTConfig cfg;

    // Character vocabulary: id <-> char. Built from the training corpus.
    std::vector<char> id_to_ch;
    int ch_to_id[256];

    Param wte;  // (vocab, d) token embedding, also the tied output head
    Param wpe;  // (block, d) positional embedding
    std::vector<LayerParams> layers;
    Param lnf_g, lnf_b;

    GPT() = default;
    GPT(const GPTConfig& c, const std::vector<char>& vocab_chars, uint32_t seed = 42);

    // One training step on a single window: tokens has cfg.block + 1 ids,
    // predicts tokens[i+1] from tokens[..i]. Returns mean cross-entropy.
    float train_step(const std::vector<int>& tokens, float lr, int adam_t);

    // Sample count characters starting from prompt at the given temperature.
    std::string generate(const std::string& prompt, int count, float temperature,
                         uint32_t seed = 123);

    void save(const std::string& path) const;
    static GPT load(const std::string& path);

    std::vector<int> encode(const std::string& s) const;
    std::string decode(const std::vector<int>& ids) const;

    size_t param_count() const;

private:
    // Forward pass over a window, caching everything backward needs.
    struct Cache;
    float forward(const std::vector<int>& tokens, Cache& cache);
    void backward(const Cache& cache, const std::vector<int>& targets);
    void adam_update(float lr, int t);

    std::mt19937 rng;
};

// Exposed for tests: numerical gradient checking drives the public train_step,
// so tests live in tests/test_main.cpp against this same interface.
