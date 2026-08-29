# GRPO + RLVR: Minimal Reinforcement Learning Building Blocks

A compact educational implementation of the core building blocks behind **Reinforcement Learning with Verifiable Rewards (RLVR)** and **Group Relative Policy Optimization (GRPO)**.

This repository is designed as a **minimal reference implementation** for understanding the fundamental mechanics behind modern reasoning-model reinforcement learning. It intentionally focuses on clarity rather than production-scale training infrastructure.

> **Note:** This is an educational implementation, not a production-ready GRPO trainer.

---

## 📖 Overview

Modern reasoning models can be improved using reinforcement learning after pretraining and supervised fine-tuning.

Two important ideas behind this process are:

* **RLVR (Reinforcement Learning with Verifiable Rewards)** — uses deterministic verifiers to evaluate whether a model's output is correct.
* **GRPO (Group Relative Policy Optimization)** — optimizes a policy by comparing multiple sampled responses generated for the same prompt.

The high-level pipeline is:

```text
                    Prompt
                       │
                       ▼
           Generate Multiple Completions
                       │
                       ▼
              Deterministic Verifier
                       │
                       ▼
                    Rewards
                       │
                       ▼
            Group-Relative Advantages
                       │
                       ▼
            Clipped GRPO Objective
                       │
                       ▼
          Reference Policy Regularization
                       │
                       ▼
                Update Policy
```

---

# 🧠 RLVR — Reinforcement Learning with Verifiable Rewards

Traditional RLHF-style training often relies on human preferences or learned reward models.

RLVR takes a different approach.

Instead of asking:

> "Does this response look good?"

RLVR asks:

> "Can we automatically verify whether this response is correct?"

For example:

```text
Prompt:
    What is 17 × 6?

Completion A:
    The answer is 102.

Completion B:
    The answer is 104.
```

A deterministic verifier can assign:

```text
Completion A → Reward = 1
Completion B → Reward = 0
```

---

# 🚀 GRPO — Group Relative Policy Optimization

GRPO generates multiple candidate responses for the same prompt.

Instead of:

```text
Prompt → One Completion
```

GRPO performs:

```text
Prompt
   │
   ├── Completion 1
   ├── Completion 2
   ├── Completion 3
   └── Completion G
```

Each completion receives a reward from the verifier.

For example:

```text
Rewards = [1, 0, 1, 0]
```

GRPO then compares responses **relative to the other responses in the same group**.

This removes the need for a separate value model or critic.

---

## 📊 Group-Relative Advantages

Raw rewards are normalized within each prompt group.

The advantage is computed as:

$$
A_i =
\frac{r_i - \mu}
{\sigma + \epsilon}
$$

Where:

* \(r_i\) = reward for completion \(i\)
* \(\mu\) = mean reward within the group
* \(\sigma\) = standard deviation of rewards
* \(\epsilon\) = numerical stability constant

Interpretation:

| Advantage         | Meaning                       |
| ----------------- | ----------------------------- |
| \(A_i > 0\)       | Better than the group average |
| \(A_i < 0\)       | Worse than the group average  |
| \(A_i \approx 0\) | Approximately average         |

For example:

```text
Rewards:

[1, 0, 1, 0]

        │
        ▼

Group Normalization

        │
        ▼

Advantages:

[+, -, +, -]
```

The training objective will:

* Increase the probability of high-advantage responses.
* Decrease the probability of low-advantage responses.

---

# 📈 PPO-Style Clipped Objective

GRPO uses a clipped policy optimization objective similar to PPO.

The probability ratio between the current policy and the policy that generated the samples is:

$$
r(\theta)
=
\frac{\pi_\theta(y|x)}
{\pi_{\text{old}}(y|x)}
$$

Using log probabilities:

$$
r(\theta)
=
\exp(
\log \pi_\theta
-
\log \pi_{\text{old}}
)
$$

To prevent excessively large updates, the ratio is clipped:

$$
\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)
$$

The clipped surrogate objective is:

$$
\min
\left(
r(\theta)A,
\text{clip}(r(\theta),1-\epsilon,1+\epsilon)A
\right)
$$

This helps keep reinforcement learning updates stable.

---

# 🔒 Reference Policy Regularization

Reinforcement learning can cause the policy to drift too far from its original behavior.

To prevent this, GRPO typically keeps a frozen **reference policy**.

Conceptually:

```text
Reference Policy
       │
       │  Regularization
       ▼
Current Policy
       │
       ▼
Higher Reward
```

The optimization objective balances:

```text
Reward Improvement
        -
Policy Divergence
```

The simplified objective used in this implementation is:

$$
L
=
-
\left(
\text{Surrogate Objective}
-
\beta \cdot KL
\right)
$$

Where:

* `β` controls the strength of reference-policy regularization.
* `KL` measures divergence between the current and reference policies.

---

# 🔄 Complete Training Pipeline

The `train_step()` function demonstrates the complete RLVR + GRPO workflow.

### Step 1 — Sample Multiple Completions

For each prompt:

```text
Prompt
   │
   ▼
Generate G responses
```

```python
completions, old_logp = sample_group(
    policy,
    prompts,
    group_size
)
```

---

### Step 2 — Verify Responses

Each completion receives a deterministic reward.

```python
rewards = [
    verify_math_answer(completion, gold)
]
```

Example:

```text
Completion 1 → Correct   → 1
Completion 2 → Incorrect → 0
Completion 3 → Correct   → 1
Completion 4 → Incorrect → 0
```

---

### Step 3 — Compute Group Advantages

Rewards are normalized relative to the other completions:

```python
advantages = group_advantages(rewards)
```

---

### Step 4 — Score Current Policy

The sampled completions are evaluated under the trainable policy:

```python
logp = sequence_logprobs(
    policy,
    prompts,
    completions
)
```

---

### Step 5 — Score Reference Policy

The same completions are evaluated using a frozen reference model:

```python
with torch.no_grad():
    ref_logp = sequence_logprobs(
        reference,
        prompts,
        completions
    )
```

---

### Step 6 — Compute GRPO Loss

The loss combines:

* Policy probability ratio
* Group-relative advantages
* PPO-style clipping
* Reference-policy regularization

```python
loss = grpo_loss(
    logp,
    old_logp,
    ref_logp,
    advantages
)
```

---

### Step 7 — Update the Policy

Finally, standard gradient descent is performed:

```python
optimizer.zero_grad(set_to_none=True)

loss.backward()

optimizer.step()
```

---

# 📂 Project Structure

```text
.
├── GRPO_RLVR.py
└── README.md
```

### `GRPO_RLVR.py`

Contains the main building blocks:

```text
extract_final_number()
        │
        ▼
verify_math_answer()
        │
        ▼
group_advantages()
        │
        ▼
grpo_loss()
        │
        ▼
sample_group()
        │
        ▼
sequence_logprobs()
        │
        ▼
train_step()
```

---

# 🧩 Core Functions

## `extract_final_number`

Extracts the final numerical value from a generated completion.

```python
extract_final_number(
    "After calculation, the answer is 42."
)

# "42"
```

---

## `verify_math_answer`

A simple deterministic verifier for answer-only mathematical tasks.

```python
verify_math_answer(
    completion="The answer is 42.",
    gold="42"
)

# 1.0
```

Returns:

```text
1.0 → Correct
0.0 → Incorrect
```

---

## `group_advantages`

Normalizes rewards within each prompt group.

Input:

```text
[
    [1, 0, 1, 0],
    [1, 0, 0, 1]
]
```

Output:

```text
Normalized relative advantages
```

Tensor shape:

```text
[batch_size, group_size]
```

---

## `grpo_loss`

Computes the simplified clipped GRPO objective.

Inputs:

```text
logp       → Current policy log probabilities
old_logp   → Rollout policy log probabilities
ref_logp   → Reference policy log probabilities
advantages → Group-relative advantages
```

Tensor shapes:

```text
logp       [batch_size, group_size, tokens]
old_logp   [batch_size, group_size, tokens]
ref_logp   [batch_size, group_size, tokens]

advantages [batch_size, group_size]
```

---

# ▶️ Running the Demo

Run:

```bash
python GRPO_RLVR.py
```

Example output:

```text
rewards

tensor([
    [1., 0., 1., 0.],
    [1., 0., 0., 1.]
])

advantages

tensor([
    [...],
    [...]
])

demo loss: ...
```

---

# 🎯 Key Hyperparameters

| Parameter             | Default | Description                                |
| --------------------- | ------: | ------------------------------------------ |
| `group_size`          |     `4` | Number of completions generated per prompt |
| `eps`                 |   `0.2` | PPO/GRPO clipping coefficient              |
| `beta`                |  `0.04` | Reference-policy regularization strength   |
| `group_advantage_eps` |  `1e-6` | Numerical stability constant               |

---

# 🧠 Main Takeaway

The entire RLVR + GRPO pipeline can be summarized as:

```text
Generate Multiple Solutions
            │
            ▼
Automatically Verify Them
            │
            ▼
Compare Solutions Within Each Group
            │
            ▼
Compute Relative Advantages
            │
            ▼
Increase Probability of Better Responses
            │
            ▼
Decrease Probability of Worse Responses
            │
            ▼
Prevent Excessive Policy Drift
```

In short:

> **RLVR provides the reward signal through deterministic verification.**

> **GRPO uses relative performance among multiple sampled responses to optimize the policy without requiring a separate critic model.**

Together, they form an elegant and efficient framework for reinforcement learning on tasks where correctness can be automatically verified.
