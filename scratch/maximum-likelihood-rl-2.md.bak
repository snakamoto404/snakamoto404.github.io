---
title: "Maximum likelihood reinforcement learning"
date: 2026-02-16
summary: "A highly principled foundational RL paper with easily actionable changes. We derive the continuous generalization."
---

I've always wondered what happens if one applies RL to supervisable tasks. For example, in a binary classification task with data $(x_j, y_j)$, we could train with accuracy-like reward
$$
R = \mathbb E\, [y p_\theta(x) + (1-y)(1-p_\theta(x))].
$$
No one really does this in mainstream supervised learning, presumably for good reasons. We typically train the same model class with cross-entropy instead. Why?

A highly impressive [recent paper](https://arxiv.org/pdf/2602.02710) from CMU asks this question in the RL setting and pushes it much further. Core empirical claim: MaxRL Pareto-dominates the RL baselines they test, with up to $20\times$ test-time scaling efficiency gains vs. GRPO on reasoning settings.

Good. Now let's unpack why that should happen mechanistically.

The original paper focuses on binary/discrete correctness rewards. Useful already. But as an exercise in understanding the principle (and for practical low-SNR regression-style tasks), we can generalize the same maximum-likelihood logic to continuous supervised likelihoods.

Qualitatively, the maximum-likelihood RL objective:
- Is sharper, and is lower-bounded by direct expected-likelihood objectives (equivalently, direct objectives are loose upper bounds after an affine shift).
- Fixing a prompt $(x, y)$, upweights the most successful rollouts.
- Marginalizing over rollouts, upweights the most difficult prompts.
- All that "SOTA" policy-based RL methods do is (mostly) reweighting scores; this one is no different. 

## Contents

- [Preamble / ramble on MLE](#preamble--ramble-on-mle)
- [Formulation, notation](#formulation-notation)
- [Direct RL as a Maximum Likelihood approximation](#direct-rl-as-a-maximum-likelihood-approximation)
- [Gradient estimators](#gradient-estimators)
- [Full generalization](#full-generalization)
- [Pseudo-code (with an abstraction barrier)](#pseudo-code-with-an-abstraction-barrier)
- [Optional technical note: estimating $w_T$](#optional-technical-note-estimating-w_t)

## Preamble / ramble on MLE

Maximum likelihood estimation (MLE) is a near-axiomatic principle: given data and a parametric family, choose parameters that maximize the probability (likelihood) of observed data.

- Classification is MLE: data-label pairs $(x, y) \sim \rho$, model $p_\theta(y \mid x)$.
- Regression is MLE: MSE corresponds to Gaussian noise MLE; L1 corresponds to Laplace noise MLE.
- VAEs are MLE-ish too: maximize a tractable lower bound on log-likelihood, while tightening that lower bound.

Given the ubiquity of MLE, it's surprising that Maximum-likelihood RL (MaxRL) is this recent.

Why do we supervise with cross-entropy $\mathbb E \log p_\theta(y\mid x)$ rather than accuracy-like $\mathbb E\,p_\theta(y\mid x)$? One mechanistic lens is the gradient:
$$
\nabla_\theta \log p_\theta(y\mid x) = \frac{1}{p_\theta(y\mid x)}\,\nabla_\theta p_\theta(y\mid x).
$$
So difficult cases (small $p_\theta$) get amplified by $1/p_\theta$.

> Maximum likelihood upweights difficult samples aggressively, updating the model on the frontier of its understanding.

*Addendum:* I would not over-interpret this as the *principle* itself. It's a side-effect interpretation. The principle is still MLE.

### Information-theoretic/compression lens

Maximizing log-likelihood equals minimizing expected code length under the model. In this lens, each example contributes $-\log p_\theta(y\mid x)$ nats of surprise. Hard examples are expensive in description length, so ML naturally spends gradient budget there.

---

## Formulation, notation

Let's start in the standard LLM-RL latent-generation setting.

- Data: $(x, y) \sim \rho$.
- Policy/sequence model: $z \sim m_\theta(\cdot \mid x)$, where $z$ is a rollout.
- Rollout evaluation: decode/postprocess $z$ into a prediction; compare to target $y$.

In binary reasoning with verifier, the typical objective is
$$
J_{\mathrm{RL}} = \mathbb E_{(x,y)\sim\rho}\,\mathbb E_{z\sim m_\theta(\cdot\mid x)}\,\mathbf 1[\hat y(z)=y].
$$

So far this is standard RL setup. Now the useful abstraction.

- A rollout $z$ defines a predictive distribution $\hat P(\cdot\mid z)$ over labels/targets.
- Define per-rollout likelihood
$$
l(y,z) := \hat P(y\mid z).
$$
- Marginalize over rollouts:
$$
p_\theta(y\mid x) := \mathbb E_{z\sim m_\theta(\cdot\mid x)} l(y,z).
$$
- Score function:
$$
S(x,z) := \nabla_\theta \log m_\theta(z\mid x).
$$

This score vector is almost always **the only** intermediate through which policy parameters $\theta$ touch policy-gradient objectives. Most algorithms differ mainly in how they reweight or shift these score vectors.
Unpacking / understanding the latent variable formulation there already goes 60% of the way torwards understanding this post; also note that this formulation subsumes supervised maximum-likelihood when sampling is trivial. 

## Direct RL as a Maximum Likelihood approximation

Here we will show:
- Direct RL $\leftrightarrow$ MaxRL is the same structural story as accuracy $\leftrightarrow$ cross-entropy.
- Direct objectives are first-order approximations; MaxRL is the higher-order MLE completion.

Recall the maximum likelihood objective:
$$
J_{\mathrm{ML}} = \mathbb E_{(x,y)\sim\rho}\log p_\theta(y\mid x)
= \mathbb E_{(x,y)\sim\rho}\log\mathbb E_{z\sim m_\theta(\cdot\mid x)} l(y,z).
$$
Critically, **marginalization over rollouts** happens inside the logarithm. Next, let's substitute two concrete use-cases and see how they compare to standard rewards. 


### Binary reasoning setup

Binary verifier means $l(y,z)=\mathbf 1[\hat y(z)=y]$. Then
$$
J_{\mathrm{pass}} = \mathbb E_{(x,y)} p_\theta(y\mid x)
$$
is the usual expected pass-rate objective, while
$$
J_{\mathrm{ML}} = \mathbb E_{(x,y)}\log p_\theta(y\mid x)
$$
is the MLE analogue (cross-entropy counterpart).

So this is exactly the familiar supervised split: expected correctness vs. expected log-correctness. Same objects, very different gradient geometry.

### Continuous regression setup

Let's swap in a continuous likelihood. With Gaussian observational model (say $\sigma^2=1$):
$$
l(y,z)=\hat P(y\mid z)=\mathcal N\big(y;\hat y(z),1\big)
=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(y-\hat y(z))^2}{2}\right).
$$
Then
$$
J_{\mathrm{ML}}
= \mathbb E_{(x,y)}\log\mathbb E_{z\sim m_\theta(\cdot\mid x)}
\exp\left(-\frac{(y-\hat y(z))^2}{2}\right) + \text{const}.
$$
The direct analogue is minimizing the MSE of each rollout
$$
J_{\mathrm{direct}}
= -\dfrac 1 2 \mathbb E_{(x,y)}\mathbb E_{z\sim m_\theta(\cdot\mid x)}
(y-\hat y(z))^2
$$
So the same gap appears: direct expectation vs. log of expectation.

### Putting them together: Jensen and Taylor

Fix $(x,y)$ and write $p := p_\theta(y\mid x)$ for brevity. Reminder: $p=\mathbb E_{z\sim m_\theta(\cdot\mid x)}l(y,z)$. For $p\in(0,1]$:
$$
\log p = -\sum_{k=1}^\infty \frac{(1-p)^k}{k}.
$$
Define the $T$-th truncation:
$$
J_T(p) := -\sum_{k=1}^{T}\frac{(1-p)^k}{k}.
$$

- $J_1(p)=p-1$, i.e. direct RL up to an additive constant.
- $J_T$ interpolates between direct RL and full ML: $J_1 \ge J_2 \ge \cdots \ge \log p$.

Now differentiate:
$$
\nabla_\theta J_T(p)
= \sum_{k=1}^{T}(1-p)^{k-1}\,\nabla_\theta p
= w_T(p)\,\nabla_\theta p,
$$
with
$$
w_T(p) := \sum_{k=1}^{T}(1-p)^{k-1}
= \frac{1-(1-p)^T}{p}.
$$
Using the score identity again, $\nabla_\theta p = \mathbb E_z[l(y,z)S(x,z)]$:
$$
\nabla_\theta J_T(p)
= w_T(p)\,\mathbb E_z[l(y,z)S(x,z)].
$$

Two important qualitative remarks.

**Remark 1 (within a fixed sample):** the rollout contribution is proportional to $l(y,z)$. So higher-likelihood rollouts are reinforced more. In binary case, only successful rollouts contribute.

**Remark 2 (across samples):** sample-level multiplier is $w_T(p)$. For small $p$, this is large; for $T\to\infty$, $w_\infty(p)=1/p$. So hard prompts get amplified.


## Gradient estimators

Here, you say, "all's great! There's this beautiful thery and a highly principled objective. How are we going about optimizing it?" 

The original paper gives a simple unbiased estimator in the binary case. Then we generalize the same principle.

### Binary case

For fixed $x$, draw $z_1,\dots,z_N\overset{iid}{\sim} m_\theta(\cdot\mid x)$, define
$$
r_j:=\mathbf 1[\hat y(z_j)=y],\qquad S_j:=\nabla_\theta\log m_\theta(z_j\mid x),\qquad K:=\sum_{j=1}^N r_j.
$$
Estimator:
$$
\hat g_N(x)=
\begin{cases}
\dfrac{1}{K}\sum_{j=1}^N r_j S_j, & K\ge 1,\\
0,&K=0.
\end{cases}
$$
Its expectation equals the truncated objective gradient with order $T=N$:
$$
\mathbb E[\hat g_N(x)] = \nabla_\theta J_N(p)
= \left(\frac{1-(1-p)^N}{p}\right)\nabla_\theta p.
$$

This is the one-line implementation-level change: normalize by mean reward (successful count), not by fixed $N$.

## Full generalization

Okay, the binary intuition is clean. Now we do the general case with the same skeleton.
Replace binary $r_j\in\{0,1\}$ by general likelihood samples
$$
l_j := l(y,z_j)\in\mathbb R_+.
$$
From here onward, **unbiased** means unbiased for the exact $T$-th truncation:
$$
\mathbb E[\hat g_T(x,y)] = \nabla_\theta J_T(p).
$$
With leave-one-out construction, this requires enough samples to identify all moments up to order $T-1$, i.e. $N\ge T$.

Reminder on symbols for the rest of this section: $l_j$ is per-rollout likelihood, $S_j$ is the score, and $\omega_j$ is the leave-one-out estimate of the truncation weight.

For each sample,
$$
\nabla_\theta J_T(p) = w_T(p)\,\nabla_\theta p,
\qquad
\nabla_\theta p = \mathbb E[lS].
$$
So we need an unbiased estimator of the product $w_T(p)\nabla_\theta p$.

A naive product of two empirical estimators is generally biased (correlation term). The clean fix is leave-one-out factorization.

For each index $j$:
- Use one rollout for the gradient factor: $g_j := l_j S_j$ (unbiased for $\nabla_\theta p$).
- Use the other $N-1$ rollouts to estimate $w_T(p)$:
$$
\omega_j := \texttt{EstimateWeight}\big(\{l_i\}_{i\ne j},T\big),
$$
with $\mathbb E[\omega_j]=w_T(p)$ exactly (for the chosen truncation order $T$).

Then define
$$
\hat g_T(x,y)
:= \frac{1}{N}\sum_{j=1}^N \omega_j\,l_j\,S_j.
$$
Because $\omega_j$ is built from samples independent of $(l_j,S_j)$,
$$
\mathbb E[\hat g_T(x,y)]
= \frac{1}{N}\sum_{j=1}^N \mathbb E[\omega_j]\,\mathbb E[l_jS_j]
= w_T(p)\,\nabla_\theta p
= \nabla_\theta J_T(p).
$$

This is exactly the abstraction barrier we want: all the combinatorics/statistics for estimating $w_T$ are compartmentalized inside `EstimateWeight`.

---

## Pseudo-code (with an abstraction barrier)

```text
Algorithm: Generalized MaxRL (LOO factorization)
Inputs: batch B, rollouts N, truncation T, policy m_theta
Require: N >= T

for each (x, y) in B:
  sample z_1, ..., z_N ~ m_theta(. | x)

  for j = 1..N:
    l_j <- l(y, z_j)
    S_j <- grad_theta log m_theta(z_j | x)

  for j = 1..N:
    omega_j <- EstimateWeight({l_i}_{i != j}, T)
    # unbiased estimate of w_T(p) from N-1 samples

  g(x, y) <- (1/N) * sum_{j=1}^N omega_j * l_j * S_j

return g_batch <- (1/|B|) * sum_{(x,y) in B} g(x, y)
```

Optional control variate (zero-mean score baseline) can be added in the usual way to reduce variance. The baseline changes variance, not the target expectation.

---

## Optional technical note: estimating $w_T$

If you care about the plumbing details, this is the compartmentalized part.

Given
$$
w_T(p)=\sum_{k=1}^{T}(1-p)^{k-1},
$$
we need unbiased estimators for powers $(1-p)^q$ from iid samples $l_i$ where $\mathbb E[l_i]=p$.

Using $(1-l_i)$ and U-statistics over distinct indices gives minimum-variance unbiased estimators for these polynomial moments. For exact $T$-th order unbiasedness:
- require $N-1\ge T-1$ (equivalently $N\ge T$),
- estimate each $(1-p)^{k-1}$ with its U-statistic on the $N-1$ leave-one-out pool for $k=1,\dots,T$,
- sum those terms to get $\omega_j$ with $\mathbb E[\omega_j]=w_T(p)$.

If $N<T$, you should treat that as a deliberate lower-order approximation (change $T$), not as an unbiased estimator of the original $J_T$.

That section is mostly textbook U-statistics machinery, so it is reasonable to treat `EstimateWeight` as a solved subroutine in the main exposition.