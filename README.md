# C-GPT

A small character-level GPT written from scratch in C++. No libraries beyond
the C++ standard library: tensors, matrix math, the transformer forward pass,
manual backpropagation, and Adam are all implemented in this repo.

## Provenance

This project grew out of an earlier C-GPT repository started by Zaid
Marwat with a collaborator; that repo is unchanged and stays where it is.
Everything in THIS repository — the tensor ops, the transformer, the
training loop, checkpointing, sampling, the tests, and this README — was
written fresh by Zaid with the help of his AI assistant. The training data
is a slice of the public-domain Tiny Shakespeare corpus.

## Architecture

- Character-level tokens (vocab = distinct characters in the corpus)
- Token + learned positional embeddings
- N pre-norm transformer blocks: causal multi-head self-attention, then a
  GELU MLP (4x width), residual connections around both
- Final layer norm, output projection tied to the token embedding
- Cross-entropy loss, manual backprop through every op, Adam with
  global-norm gradient clipping at 1.0

## Layout

- `src/tensor.h/.cpp` — dense tensor and ops (matmul, softmax, layer norm,
  GELU, cross-entropy) plus their backward passes
- `src/gpt.h/.cpp` — the model: forward with cached activations, backward,
  Adam update, save/load, sampling
- `src/main.cpp` — the `gpt` command line tool
- `tests/test_main.cpp` — op tests, finite-difference gradient checks, and
  an end-to-end overfit test
- `data/tiny.txt` — a small slice of the public-domain Tiny Shakespeare
  corpus for a quick train

## Build and test

```sh
make            # builds ./gpt and ./build/tests
make check      # builds and runs the test suite
```

The tests verify the math ops against known values, check the hand-written
gradients against finite differences, and confirm the full model can overfit
a repeated window (loss 1.63 -> 0.0000), which exercises end-to-end backprop.

## Train

```sh
./gpt train --data data/tiny.txt --steps 4000 --block 64 --d 64 \
    --layers 2 --heads 4 --lr 0.003 --out build/tiny.bin
```

On the included 60 KB corpus this 2-layer, 64-wide model (~108K parameters)
trains in under a minute on one CPU core, with the loss dropping from 4.08
(random for 59 characters is ln(59) ~= 4.08) to about 2.2.

## Sample

```sh
./gpt sample --model build/tiny.bin --prompt "KING:" \
    --tokens 400 --temperature 0.8 --seed 7
```

(Train first; the checkpoint is a build artifact, not committed.)

Output after 4000 steps — the model has learned Shakespeare's character
structure (speaker prefixes, line breaks, word shapes) even though the
individual words are still mostly invented:

```
KING:
He myort onod ld nousind ar sthandss Ioulus derst meas s m.
BRUSUS:
Burlt blal lle omat lle you is llis s gha s merellns, wald pldoves thelou om m, the s mandilun ngous o, thes ssomy anon,
...
```

Bigger data and more steps improve fluency; any plain-text file works as
`--data`.

## CLI

```
gpt train --data FILE [--steps N] [--out FILE] [--block N] [--d N]
          [--layers N] [--heads N] [--lr F] [--seed N]
gpt sample --model FILE [--prompt STR] [--tokens N] [--temperature F] [--seed N]
```
