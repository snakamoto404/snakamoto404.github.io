---
title: "Maximum likelihood reinforcement learning"
date: 2026-02-16
summary: "A highly principled foundational RL paper with easily actionable changes. We derive the continuous generalization."
---

I've always wondered what happens if one applies RL to supervisable tasks. For example, given a binary classification task $(x_j, y_j)$, maximize accuracy as the reward
$$
R := \mathbb E\,[y\,p_\theta(x)+(1-y)(1-p_\theta(x))].
$$
No one does this, presumably for good reasons; how does such a model compare to a normal model trained using cross-entropy?

Turns out that a highly impressive [recent paper](https://arxiv.org/pdf/2602.02710) from CMU goes *exactly* down this rabbit hole -- and comes up with actionable, principled insights. 
They point out that correctness-based RL is optimizing
$$
J_{\mathrm{RL}}(\theta)=\mathbb E_{x\sim\rho}\,p_\theta(x)
$$
where $p_\theta(x)$ is the success probability ("pass rate"), while the likelihood principle suggests optimizing
$$
J_{\mathrm{ML}}(\theta)=\mathbb E_{x\sim\rho}\,\log p_\theta(x),
$$
whose gradient upweights low-pass-rate inputs by a factor $1/p_\theta(x)$. 

### What's in the paper

- A **compute-indexed** family of truncated objectives interpolating between RL and ML,
- A **simple unbiased estimator** whose expected gradient matches the truncated objective,
- Experiments demonstrating strong Pareto improvements over common baselines (e.g. GRPO/RLOO) including up to **(~20x)** test-time scaling efficiency gains in their reasoning setups.

### What's new here 
Luckily (for us), the authors focus on the binary, discrete-reward setting. This post unpacks the following qualitative behavior of MaxRL 

- is *sharper* than (and lower-bounded by) direct objectives, admitting a natural bounded truncation that interpolates RL -> ML with rollout budget,
- fixing a prompt/sample, **upweights the most successful rollouts** (soft-max / log-sum-exp behavior),
- marginalizing over rollouts, **upweights the most difficult prompts** via inverse-probability reweighting.

we also put forward a generalization that abstracts at the level of per-rollout likelihood, admitting application to e.g. regression tasks. We expect this generalization to be highly applicable to regression RL tasks with low signal to noise ratio. 
