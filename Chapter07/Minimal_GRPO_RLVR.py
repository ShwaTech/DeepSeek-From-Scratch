"""
Minimal GRPO + RLVR building blocks for Chapter 7.

---------------------------------------------------------------------------------------------------

This file provides a compact educational implementation of two important
techniques used in modern reasoning-model training:
    1. Reinforcement Learning with Verifiable Rewards (RLVR)
    2. Group Relative Policy Optimization (GRPO)

It is not a production-ready trainer. Instead, it focuses on the core ideas
behind how modern language models can learn from automatically verifiable
feedback.

1. RLVR — Reinforcement Learning with Verifiable Rewards:
    RLVR replaces subjective reward models with deterministic verification whenever
    correctness can be automatically checked.

For example, in a mathematical task:
    Prompt → Model generates an answer → Verifier checks correctness → Reward

A correct answer may receive:
    reward = 1
while an incorrect answer receives:
    reward = 0

2. GRPO — Group Relative Policy Optimization:

Instead of generating only one response per prompt, GRPO samples a group of
candidate completions:
    Prompt → {Completion₁, Completion₂, ..., Completion_G}

Each completion receives a reward from the verifier. GRPO then compares the
responses relative to the other responses in the same group.

The group-relative advantage is computed as:

            rᵢ - mean(r)
Aᵢ = ─────────────────────────
            std(r) + ε

where:

rᵢ        = reward of completion i
mean(r)   = average reward within the group
std(r)    = reward standard deviation
Aᵢ        = relative advantage

---------------------------------------------------------------------------------------------------

The file collects the small snippets shown in the chapter into one runnable
script. It is not a production trainer; it is a compact reference
implementation for the core mechanics:

1. deterministic verifier rewards
2. group-relative advantages
3. clipped GRPO loss with reference-policy KL
4. one training-step skeleton
"""

from __future__ import annotations

import re
from typing import Iterable

import torch



# ------------------------------------------------------------
# ------------------- Simple Helpers -------------------------
# ------------------------------------------------------------
def extract_final_number(text: str) -> str | None:
    """
    Return the last integer or decimal number found in a completion.
    
    Args:
        text: The text to extract the number from.
        
    Returns:
        The last integer or decimal number found in the text, or None if no number is found.
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    
    return matches[-1] if matches else None


def verify_math_answer(completion: str, gold: str) -> float:
    """
    A toy RLVR reward for answer-only math tasks.
    
    Args:
        completion: The completion to verify.
        gold: The gold answer.
        
    Returns:
        A reward of 1.0 if the completion is correct, 0.0 otherwise.
    """
    predicted = extract_final_number(completion)
    if predicted is None:
        return 0.0
    
    return 1.0 if predicted == str(gold) else 0.0



# ------------------------------------------------------------
# ------------------- Group Advantage -------------------------
# ------------------------------------------------------------
def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize rewards within each prompt group.

    Args:
        rewards: Tensor of shape [batch_size, group_size].
        eps: Minimum standard deviation for numerical stability.

    Returns:
        Tensor of normalized advantages with the same shape as rewards.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp_min(eps)
    
    return (rewards - mean) / std



# ------------------------------------------------------------
# ------------------- GRPO Loss ------------------------------
# ------------------------------------------------------------
def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """
    Compute the clipped GRPO objective as a minimization loss.

    Args:
        logp:         The log probability of the new policy.
        old_logp:     The log probability of the old policy.
        ref_logp:     The log probability of the reference policy.
        advantages:   The advantages of the new policy.
        eps:          The clipping parameter.
        beta:         The KL penalty parameter.
        
    Returns:
        The clipped GRPO objective as a minimization loss.
    
    Shapes:
        logp, old_logp, ref_logp: [batch_size, group_size, tokens]
        advantages: [batch_size, group_size]
    """
    ratio = torch.exp(logp - old_logp)
    
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps)
    
    adv = advantages.unsqueeze(-1)
    
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    
    kl = logp - ref_logp
    
    return -(surrogate - beta * kl).mean()




# ------------------------------------------------------------
# ------------------- Sample Group ---------------------------
# ------------------------------------------------------------
def sample_group(policy, prompts: Iterable[str], group_size: int):
    """
    Placeholder for policy rollout code.

    Args:
        policy:      The policy to sample from.
        prompts:     The prompts to sample from.
        group_size:  The number of samples to generate for each prompt.
        
    Returns:
        A tuple of (completions, old_logp).
        
    In a real implementation this calls policy.generate(..., do_sample=True)
    and returns completions plus old-policy token log probabilities.
    """
    del policy
    
    completions = [[f"{prompt} answer {i}" for i in range(group_size)] for prompt in prompts]
    
    old_logp = torch.zeros(len(completions), group_size, 4)
    
    return completions, old_logp



# ---------------------------------------------------------------
# ------------------ Sequence Log probabilities -----------------
# ---------------------------------------------------------------
def sequence_logprobs(model, prompts, completions) -> torch.Tensor:
    """
    Placeholder for scoring generated completions under a model.
    
    Args:
        model:         The model to score the completions under.
        prompts:       The prompts to score the completions under.
        completions:   The completions to score.
        
    Returns:
        A tensor of shape [batch_size, group_size, tokens] containing the log probabilities of the completions.
    """
    del model, prompts
    
    return torch.zeros(len(completions), len(completions[0]), 4, requires_grad=True)



# ---------------------------------------------------------------
# --------------------- Training step ---------------------------
# ---------------------------------------------------------------
def train_step(batch, policy, reference, optimizer, group_size: int = 4):
    """
    One GRPO + RLVR training step skeleton.
    
    Args:
        batch:      The batch of data to train on.
        policy:     The policy to train.
        reference:  The reference model.
        optimizer:  The optimizer to use.
        group_size: The number of samples to generate for each prompt.
        
    Returns:
        The loss of the training step.
    """
    prompts, gold_answers = batch
    completions, old_logp = sample_group(policy, prompts, group_size)

    rewards = torch.tensor(
        [
            [verify_math_answer(y, gold) for y in group]
            for group, gold in zip(completions, gold_answers)
        ],
        device=old_logp.device,
    )
    advantages = group_advantages(rewards)

    logp = sequence_logprobs(policy, prompts, completions)
    
    with torch.no_grad():
        ref_logp = sequence_logprobs(reference, prompts, completions)
    
    loss = grpo_loss(logp, old_logp, ref_logp, advantages)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    return {"loss": float(loss.detach()), "reward": float(rewards.mean())}



# ---------------------------------------------------------------
# --------------------- Demo ------------------------------------
# ---------------------------------------------------------------
def _demo() -> None:
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
    advantages = group_advantages(rewards)
    print("↪️ rewards:\n")
    print(rewards)
    print("↪️ advantages:\n")
    print(advantages)

    logp = torch.randn(2, 4, 6) * 0.05
    old_logp = torch.zeros_like(logp)
    ref_logp = torch.zeros_like(logp)
    loss = grpo_loss(logp, old_logp, ref_logp, advantages)
    print(f"demo loss: {loss.item():.4f}")



if __name__ == "__main__":
    _demo()