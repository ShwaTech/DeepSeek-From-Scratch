# 🚀 Building DeepSeek From Scratch

> **A hands-on journey into the architecture, Attention Mechanisms, Mixture-of-Experts, Multi-Token Prediction, Knowledge Distillation, and Reinforcement Learning techniques behind modern DeepSeek-style Large Language Models.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge\&logo=pytorch)
![LLM](https://img.shields.io/badge/Focus-LLM%20Architecture-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🌟 Overview

**Building DeepSeek From Scratch** is an educational project focused on understanding and implementing the fundamental building blocks behind modern Large Language Models and DeepSeek-inspired architectures.

Rather than treating an LLM as a black box, this repository breaks complex ideas into small, understandable implementations — starting from **tokenization and embeddings**, moving through **attention mechanisms**, and progressing toward advanced architectures such as:

* ⚡ KV Cache
* 🔥 Multi-Head Attention (MHA)
* 🧩 Multi-Query Attention (MQA)
* 🧠 Grouped-Query Attention (GQA)
* 🚀 Multi-Head Latent Attention (MLA)
* 🏗️ Mixture of Experts (MoE)
* 🎯 Multi-Token Prediction (MTP)
* 📦 Quantization
* 🧬 Knowledge Distillation
* 🤖 GRPO + RLVR

The goal is simple:

> **Understand the ideas behind modern reasoning models by building their core components from scratch.**

---

# 🗺️ Learning Journey

```text
      Tokenization
           │
           ▼
      Pair Encoding
           │
           ▼
      Data Loading
           │
           ▼
      Embeddings
           │
           ▼
      Self-Attention
           │
           ▼
      Causal Attention
           │
           ▼
      Multi-Head Attention
           │
           ▼
      KV Cache
           │
           ├──────────────► MQA
           │
           └──────────────► GQA
                               │
                               ▼
                              MLA
                               │
                               ▼
                     Mixture of Experts
                               │
                               ▼
                    Multi-Token Prediction
                               │
                               ▼
                        Quantization
                               │
                               ▼
                   Knowledge Distillation
                               │
                               ▼
                        GRPO + RLVR
```

---

# 📚 Project Structure

```text
Building-DeepSeek-From-Scratch/
│
├── 📁 Chapter01/
│   ├── 01_Tokenization.ipynb
│   ├── 02_Byte_Pair_Encoding.ipynb
│   ├── 03_Data_Loader.ipynb
│   ├── 04_Embeddings.ipynb
│   ├── 05_Self_Attention.ipynb
│   ├── 06_Causal_Attention.ipynb
│   ├── 07_Multi-Head_Attention(MHA).ipynb
│   └── 08_Multi-Head_Attention_Visualized.ipynb
│
├── 📁 Chapter02/
│   ├── 01_KV_Cache.ipynb
│   ├── 02_Multi-Query_Attention(MQA).ipynb
│   └── 03_Group-Query_Attention(GQA).ipynb
│
├── 📁 Chapter03/
│   ├── 📁 Bonus/
│   │   └── MHA_vs_MQA_vs_GQA_vs_MLA.ipynb
│   │
│   ├── 01_Multi-Head_Latent_Attention(MLA).ipynb
│   └── 02_MLA_with_Decoupled_RoPE(DSA).ipynb
│
├── 📁 Chapter04/
│   ├── 📁 Bonus/
│   │   └── DeepSeek_MoE_Comparison.ipynb
│   │
│   ├── 01_Mixture_of_Experts(MoE)_from_Scratch.ipynb
│   └── 02_DeepSeek_Mixture_of_Experts(DeepSeekMoE).ipynb
│
├── 📁 Chapter05/
│   ├── 01_Multi_Token_Prediction(MTP)_From_Scratch.ipynb
│   ├── 02_DeepSeek_Multi_Token_Prediction(DeepSeekMTP).ipynb
│   └── 03_DeepSeek_Quantization.ipynb
│
├── 📁 Chapter06/
│   ├── README.md
│   ├── requirements.txt
│   ├── Stage_01_prepare.py
│   ├── Stage_02_model.py
│   ├── Stage_03_train.py
│   └── Stage_04_sample.py
│
├── 📁 Chapter07/
│   ├── Minimal_GRPO_RLVR.py
│   └── README.md
│
├── 📁 Chapter08/
│   ├── Knowledge_Distillation.ipynb
│   └── README.md
│
├── 📁 data/
│
├── .gitignore
├── .python-version
├── LICENSE
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── template.py
└── uv.lock
```

---

# 📖 Chapters

## 🧩 Chapter 1 — Foundations of Transformers

Build the fundamental components required to understand modern Transformer-based models.

| Topic                | Description                       |
| -------------------- | --------------------------------- |
| Tokenization         | Converting text into tokens       |
| Byte Pair Encoding   | Learning subword vocabularies     |
| Data Loader          | Preparing training data           |
| Embeddings           | Representing tokens as vectors    |
| Self-Attention       | Understanding token relationships |
| Causal Attention     | Autoregressive attention masking  |
| Multi-Head Attention | Parallel attention heads          |

---

## ⚡ Chapter 2 — Efficient Attention

Explore techniques designed to reduce the computational and memory cost of autoregressive generation.

* KV Cache
* Multi-Query Attention (MQA)
* Grouped-Query Attention (GQA)

---

## 🧠 Chapter 3 — Multi-Head Latent Attention

A deep dive into **Multi-Head Latent Attention (MLA)** and DeepSeek's efficient attention design.

Topics include:

* Multi-Head Latent Attention
* KV Compression
* Latent Representations
* Decoupled RoPE
* Comparison between MHA, MQA, GQA, and MLA

---

## 🏗️ Chapter 4 — Mixture of Experts

Explore sparse architectures where only a subset of experts is activated for each token.

Topics include:

* Mixture of Experts from scratch
* Expert routing
* Sparse activation
* DeepSeekMoE
* DeepSeek MoE architecture comparisons

---

## 🎯 Chapter 5 — Multi-Token Prediction & Quantization

Study two important techniques for improving LLM efficiency.

### Multi-Token Prediction

Instead of predicting only the next token:

```text
Input → Predict Token t+1
```

MTP explores predicting multiple future tokens:

```text
Input → Predict Token t+1, t+2, ..., t+n
```

### Quantization

Learn how DeepSeek-style models reduce memory and computation requirements using lower-precision numerical representations.

---

## 🧪 Chapter 6 — Building a Small Language Model

A structured implementation of a small language model training pipeline.

```text
      Prepare Data
           │
           ▼
      Build Model
           │
           ▼
      Train Model
           │
           ▼
    Generate Samples
```

The chapter is organized into independent stages:

* `Stage_01_prepare.py`
* `Stage_02_model.py`
* `Stage_03_train.py`
* `Stage_04_sample.py`

---

## 🎮 Chapter 7 — GRPO + RLVR

Explore reinforcement learning techniques used for training reasoning models.

### RLVR

**Reinforcement Learning with Verifiable Rewards** uses deterministic verifiers to evaluate model outputs.

```text
Generate → Verify → Reward → Learn
```

### GRPO

**Group Relative Policy Optimization** compares multiple sampled responses for the same prompt and optimizes the policy using relative advantages.

```text
              Prompt
                 │
                 ▼
     Generate Multiple Responses
                 │
                 ▼
           Verify Rewards
                 │
                 ▼
      Compute Group Advantages
                 │
                 ▼
           Update Policy
```

---

## 🧬 Chapter 8 — Knowledge Distillation

Learn how knowledge from a large **teacher model** can be transferred to a smaller **student model**.

Topics include:

* Teacher–Student learning
* Soft targets
* Temperature scaling
* KL Divergence
* Distillation loss
* Reasoning distillation

---

# 🛠️ Tech Stack

This project primarily uses:

* 🐍 Python
* 🔥 PyTorch
* 📓 Jupyter Notebooks
* 🧮 NumPy
* 🤗 Hugging Face ecosystem

---

# 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/ShwaTech/DeepSeek-From-Scratch.git
cd DeepSeek-From-Scratch
```

### Sync uv environment

```bash
uv sync
```

### Install dependencies

```bash
uv add -r requirements.txt
```

---

# 🎯 Project Philosophy

This repository follows one simple philosophy:

> **Don't just use modern AI architectures — understand how they work internally.**

Every major concept is broken into smaller implementations so that complex systems can be understood step by step.

The journey moves from:

```text
              Tokens
                 ↓
             Embeddings
                 ↓
             Attention
                 ↓
        Efficient Attention
                 ↓
                MLA
                 ↓
                MoE
                 ↓
                MTP
                 ↓
            Quantization
                 ↓
       Knowledge Distillation
                 ↓
       Reinforcement Learning
```

---

# ⚠️ Disclaimer

This repository is primarily an **educational and research project**.

The implementations focus on:

* Understanding
* Experimentation
* Mathematical intuition
* Core architectural concepts

They are not intended to reproduce the complete production-scale DeepSeek training infrastructure.

---

# 🤝 Contributions

Contributions, improvements, suggestions, and discussions are welcome!

If you find this project useful, consider giving it a ⭐.

---

# 📜 License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Mohamed Fathy**

Built with ❤️ and curiosity for understanding how modern Large Language Models work from the inside out.

---

<div align="center">

### ⭐ If you're learning how modern LLMs work internally, I hope this project helps you on the journey!

**From Tokens → Attention → DeepSeek-Style Architectures → Reasoning Models 🚀**

</div>

---

