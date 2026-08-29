# Knowledge Distillation — Making Powerful Models Practical

A concise implementation and educational guide to **Knowledge Distillation (KD)**, demonstrating how knowledge from a large, capable **teacher model** can be transferred to a smaller **student model**.

This project covers both **classical knowledge distillation** using soft probability targets and the **reasoning distillation approach** used by models such as DeepSeek-R1.

---

## 📚 Topics Covered

* Teacher–Student learning paradigm
* Hard labels vs. soft targets
* Temperature-scaled softmax
* Dark knowledge
* KL Divergence
* Combined distillation loss
* Temperature squared (`T²`) scaling
* Classical Knowledge Distillation
* DeepSeek-R1 reasoning distillation

---

## 🧠 The Core Idea

Instead of training a small model only on ground-truth labels:

```text
Input → Student Model → Prediction
```

Knowledge Distillation introduces a powerful teacher:

```text
                 ┌──➔ Teacher Model ──➔ Soft Knowledge
Input ───────────┤
                 └──➔ Student Model ──➔ Learning From Teacher
```

> The student learns not only from the correct answer but also from the teacher's probability distribution, which contains additional information about relationships between classes.

---

## 🌡️ Temperature Scaling

Softmax temperature controls how smooth the probability distribution becomes:

$$
P_i = \frac{\exp(z_i / T)}
{\sum_j \exp(z_j / T)}
$$

Where:

* `T = 1` → Standard softmax
* `T > 1` → Softer probability distribution
* Higher `T` → More visible teacher knowledge

---

## 📉 Distillation Loss

The complete distillation objective combines:

1. **Hard Loss** — Learning from ground-truth labels
2. **Soft Loss** — Matching the teacher's predictions

$$
\mathcal{L}
=
\alpha \mathcal{L}_{hard}
+
(1-\alpha)T^2\mathcal{L}_{soft}
$$

The soft loss is typically based on **KL Divergence** between the teacher and student probability distributions.

---

## 🔬 DeepSeek-R1 Distillation

Classical knowledge distillation transfers the teacher's probability distributions.

DeepSeek-R1-style distillation takes a different approach:

```text
                            Large Reasoning Model
                                        │
                                        ▼
                            Generate Reasoning Traces
                                        │
                                        ▼
                    Train Smaller Model on Generated Data
                                        │
                                        ▼
                        Student Learns Reasoning Patterns
```

Instead of requiring access to teacher logits, the student learns from the teacher's generated reasoning trajectories using standard supervised fine-tuning.

---

## 📂 Project Structure

```text
Chapter08/
├── Knowledge_Distillation.ipynb
└── README.md
```

The notebook includes:

* Temperature visualization
* Distillation loss implementation
* Hard vs. soft target comparison
* Teacher–student training concepts
* DeepSeek-R1 distillation discussion

---

## 🎯 Key Takeaway

> Knowledge Distillation allows smaller models to inherit useful knowledge and behavior from much larger models.

Classical KD transfers **probability distributions**, while modern reasoning distillation can transfer **complete reasoning strategies and generated solutions**.
