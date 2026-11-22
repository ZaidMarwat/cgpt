#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gpt.h"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<char> build_vocab(const std::string& text) {
    bool seen[256] = {false};
    for (char ch : text) seen[static_cast<unsigned char>(ch)] = true;
    std::vector<char> vocab;
    for (int i = 0; i < 256; i++)
        if (seen[i]) vocab.push_back(static_cast<char>(i));
    return vocab;
}

const char* kUsage =
    "c-gpt: a small character-level GPT in plain C++\n"
    "\n"
    "  gpt train  --data FILE [options]     train on a text file\n"
    "  gpt sample --model FILE [options]    generate text from a trained model\n"
    "\n"
    "train options:\n"
    "  --steps N        training steps          (default 2000)\n"
    "  --out FILE       where to save weights   (default model.bin)\n"
    "  --block N        context length          (default 64)\n"
    "  --d N            embedding width         (default 64)\n"
    "  --layers N       transformer blocks      (default 2)\n"
    "  --heads N        attention heads         (default 4)\n"
    "  --lr F           learning rate           (default 3e-3)\n"
    "  --seed N         RNG seed                (default 42)\n"
    "\n"
    "sample options:\n"
    "  --prompt STR     starting text           (default \"\\n\")\n"
    "  --tokens N       characters to generate  (default 300)\n"
    "  --temperature F  sampling temperature    (default 0.8)\n"
    "  --seed N         RNG seed                (default 123)\n";

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "%s", kUsage);
        return 2;
    }
    std::string mode = argv[1];

    // tiny flag parser: --key value pairs
    auto arg = [&](const std::string& key, const std::string& fallback) {
        for (int i = 2; i + 1 < argc; i++)
            if (argv[i] == key) return std::string(argv[i + 1]);
        return fallback;
    };

    try {
        if (mode == "train") {
            std::string data_path = arg("--data", "");
            if (data_path.empty()) {
                std::fprintf(stderr, "train needs --data FILE\n%s", kUsage);
                return 2;
            }
            int steps = std::stoi(arg("--steps", "2000"));
            std::string out = arg("--out", "model.bin");
            GPTConfig cfg;
            cfg.block = std::stoi(arg("--block", "64"));
            cfg.d_model = std::stoi(arg("--d", "64"));
            cfg.n_layer = std::stoi(arg("--layers", "2"));
            cfg.n_head = std::stoi(arg("--heads", "4"));
            float lr = std::stof(arg("--lr", "3e-3"));
            uint32_t seed = static_cast<uint32_t>(std::stoul(arg("--seed", "42")));

            std::string text = read_file(data_path);
            std::vector<char> vocab = build_vocab(text);
            cfg.vocab = static_cast<int>(vocab.size());

            GPT g(cfg, vocab, seed);
            std::vector<int> ids = g.encode(text);
            std::fprintf(stderr,
                         "corpus: %zu chars, vocab: %d, params: %zu, config: block=%d d=%d "
                         "layers=%d heads=%d\n",
                         text.size(), cfg.vocab, g.param_count(), cfg.block, cfg.d_model,
                         cfg.n_layer, cfg.n_head);
            if (static_cast<int>(ids.size()) < cfg.block + 1) {
                std::fprintf(stderr, "corpus smaller than one training window\n");
                return 1;
            }

            std::mt19937 pick(seed + 1);
            std::uniform_int_distribution<int> start(0, static_cast<int>(ids.size()) - cfg.block - 1);
            for (int s = 1; s <= steps; s++) {
                int off = start(pick);
                std::vector<int> window(ids.begin() + off, ids.begin() + off + cfg.block + 1);
                float loss = g.train_step(window, lr, s);
                if (s == 1 || s % (steps / 10) == 0)
                    std::fprintf(stderr, "step %5d/%d  loss %.4f\n", s, steps, loss);
            }
            g.save(out);
            std::fprintf(stderr, "saved %s\n", out.c_str());
            return 0;
        }

        if (mode == "sample") {
            std::string model_path = arg("--model", "model.bin");
            std::string prompt = arg("--prompt", "\n");
            int tokens = std::stoi(arg("--tokens", "300"));
            float temp = std::stof(arg("--temperature", "0.8"));
            uint32_t seed = static_cast<uint32_t>(std::stoul(arg("--seed", "123")));
            GPT g = GPT::load(model_path);
            std::string out = g.generate(prompt, tokens, temp, seed);
            std::printf("%s%s\n", prompt.c_str(), out.c_str());
            return 0;
        }

        std::fprintf(stderr, "unknown mode %s\n%s", mode.c_str(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
