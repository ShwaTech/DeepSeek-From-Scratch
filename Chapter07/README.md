# 🚀 GRPO + RLVR: Minimal Reinforcement Learning Building Blocks

A compact educational implementation of the core ideas behind **Reinforcement Learning with Verifiable Rewards (RLVR)** and **Group Relative Policy Optimization (GRPO)**.

The goal of this project is to provide a minimal and readable reference for understanding how modern reasoning models can be optimized using automatically verifiable rewards and group-relative policy optimization.

> **Note:** This is an educational implementation designed to explain the core mechanics. It is not a production-ready GRPO training framework.

---

## 📌 Overview

Modern reasoning models can be improved through reinforcement learning after pretraining and supervised fine-tuning.

This project focuses on two key ideas:

- **RLVR (Reinforcement Learning with Verifiable Rewards):** Uses deterministic verifiers to automatically evaluate whether a generated answer is correct.
- **GRPO (Group Relative Policy Optimization):** Generates multiple responses for the same prompt and optimizes the policy according to their relative performance.

The overall training pipeline is:

```text
            Prompt
               │
               ▼
    Sample Multiple Completions
               │
               ▼
     Verify Each Completion
               │
               ▼
        Compute Rewards
               │
               ▼
   Calculate Group Advantages
               │
               ▼
     Compute GRPO Objective
               │
               ▼
         Update Policy
```

---

## 🧠 RLVR — Verifiable Rewards

RLVR replaces subjective reward signals with deterministic verification whenever correctness can be automatically checked.

For example:

```text
Prompt:
What is 17 × 6?

Completion A → "102" → Reward = 1
Completion B → "104" → Reward = 0
```

This approach is particularly useful for tasks with objective correctness, such as:

* Mathematics
* Programming
* Code generation
* Logical reasoning
* Symbolic tasks

In this project, a simple mathematical verifier extracts the final number from a completion and compares it against the expected answer.

---

## 🧠 GRPO — Group Relative Policy Optimization

Instead of generating a single completion per prompt, GRPO samples a group of candidate responses:

```text
Prompt
 ├── Completion 1
 ├── Completion 2
 ├── Completion 3
 └── Completion G
```

Each completion receives a reward. GRPO then evaluates how well each response performed relative to the other responses in the same group.

### Group-Relative Advantage

For a group of completions with rewards \(r_1, r_2, \dots, r_G\), GRPO
normalizes each reward relative to the other samples in the same group:

$$
A_i = \frac{r_i - \mu}{\sigma + \epsilon}
$$

where:

- \(r_i\) is the reward assigned to completion \(i\)
- \(\mu\) is the mean reward within the group
- \(\sigma\) is the standard deviation of rewards within the group
- \(\epsilon\) is a small constant for numerical stability

| Advantage | Interpretation |
|---|---|
| \(A_i > 0\) | Better than the group average |
| \(A_i < 0\) | Worse than the group average |
| \(A_i \approx 0\) | Approximately average |

This relative normalization allows GRPO to use the sampled group as a baseline,
avoiding the need for a separate value or critic model.

---

## 🔄 Training Pipeline

The `train_step()` function combines the main components:

```text
1. Sample multiple completions
2. Verify each completion
3. Compute rewards
4. Calculate group-relative advantages
5. Score completions under the current policy
6. Score completions under the reference policy
7. Compute GRPO loss
8. Backpropagate and update the policy
```

---

## 📂 Project Structure

```text
Chapter07
├── Minimal_GRPO_RLVR.py
└── README.md
```

### Core Components

| Function                 | Purpose                                 |
| ------------------------ | --------------------------------------- |
| `extract_final_number()` | Extracts the final numerical answer     |
| `verify_math_answer()`   | Provides deterministic rewards          |
| `group_advantages()`     | Computes group-relative advantages      |
| `grpo_loss()`            | Computes the clipped GRPO objective     |
| `sample_group()`         | Placeholder for policy rollouts         |
| `sequence_logprobs()`    | Placeholder for completion scoring      |
| `train_step()`           | Demonstrates one complete training step |

---

## ▶️ Running the Demo

```bash
python Minimal_GRPO_RLVR.py
```

Example output:

```text
rewards
tensor([
    [1., 0., 1., 0.],
    [1., 0., 0., 1.]
])

advantages
tensor([...])

demo loss: ...
```

---

## ⚙️ Key Hyperparameters

| Parameter    | Default | Description                              |
| ------------ | ------: | ---------------------------------------- |
| `group_size` |     `4` | Number of completions sampled per prompt |
| `eps`        |   `0.2` | Clipping coefficient                     |
| `beta`       |  `0.04` | Reference-policy regularization strength |

---

## 🎯 Key Takeaway

The complete idea can be summarized as:

> **Generate multiple solutions → Verify them automatically → Compare them within each group → Increase the probability of successful behaviors → Regularize the policy for stable learning.**

> **RLVR** provides reliable rewards through deterministic verification.

> **GRPO** uses group-relative performance to optimize the policy without requiring a separate critic model.
