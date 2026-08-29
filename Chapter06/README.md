# Chapter 6: The DeepSeek Training Pipeline

This chapter brings together all the architectural innovations from previous chapters into a complete, trainable "MiniDeepSeek V3" model. Unlike other chapters that use Jupyter notebooks, this chapter consists of **four standalone Python scripts** that form a complete training pipeline.

## Code Structure

The code is organized as a pipeline — run the scripts in order:

```
ch06/
├── requirements.txt             # Project dependencies (Listing 6.1)
├── Stage_01_prepare.py          # Data preparation pipeline (Listings 6.2–6.5)
├── Stage_02_model.py            # Complete MiniDeepSeek architecture (Listings 6.6–6.18)
├── Stage_03_train.py            # Training loop with evaluation (Listings 6.19–6.24)
└── Stage_04_sample.py           # Text generation from trained model
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and tokenize the TinyStories dataset
python Stage_01_prepare.py

# 3. Train the model
python Stage_03_train.py

# 4. Generate text from the trained model
python Stage_04_sample.py
```

## What Each File Contains

- **Stage_01_prepare.py** — Downloads TinyStories from Hugging Face, tokenizes with the `gpt2` BPE tokenizer (tiktoken), saves as memory-mapped binary files for efficient training
- **Stage_02_model.py** — The complete MiniDeepSeek architecture integrating:
  - `RotaryPositionalEncoding` — RoPE with position offset for cached inference
  - `DeepSeekAttention` — Decoupled MLA + RoPE with KV cache support
  - `ExpertFFN` + `DeepSeekMoE` — Mixture-of-Experts with auxiliary-loss-free load balancing
  - `MTPModule` — Causal Multi-Token Prediction
  - `TransformerBlock` — Pre-norm Transformer block combining attention + MoE
  - `MiniDeepSeek` — The top-level model with separate training and inference paths
- **Stage_03_train.py** — Training script with cosine LR scheduling, mixed-precision training, and periodic checkpointing (~130M parameter "flagship" configuration)
- **Stage_04_sample.py** — Interactive text generation using KV-cached inference

## Pipeline Visual Overview

```
Raw Text Data (TinyStories)
        ↓
[Stage 01: Prepare]
  Tokenize → Memory-map → Save .bin files
        ↓
[Stage 02: Model Definition]
  Define architecture with all DeepSeek innovations
        ↓
[Stage 03: Training]
  Train for N epochs with LR scheduling + mixed precision
        ↓
[Stage 04: Sampling]
  Load checkpoint → Generate text interactively
```

This pipeline approach ensures you can swap out any component (e.g., dataset, model size, tokenizer) without breaking the entire workflow — exactly how real AI research is conducted.