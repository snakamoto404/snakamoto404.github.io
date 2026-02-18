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
- Experiments demonstrating strong Pareto improvements over common baselines (e.g. GRPO/RLOO) including up to **(~20x)** rollout scaling efficiency gains in their reasoning setups.

### What's new here 
Luckily (for us), the authors focus on the binary, discrete-reward setting. This post unpacks the following qualitative behavior of MaxRL 

- is *sharper* than (and lower-bounded by) direct objectives, admitting a natural bounded truncation that interpolates RL -> ML with rollout budget,
- fixing a prompt/sample, **upweights the most successful rollouts** (soft-max / log-sum-exp behavior),
- marginalizing over rollouts, **upweights the most difficult prompts** via inverse-probability reweighting.

we also put forward a generalization that abstracts at the level of per-rollout likelihood, admitting application to e.g. regression tasks. We expect this generalization to be highly applicable to regression RL tasks with low signal to noise ratio. 

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

The maximum likelihood objective (for our latent-generation likelihood) is
$$
J(\theta)=\mathbb E_{(x,y)\sim\rho}\,\log p_\theta(y\mid x)
=\mathbb E_{(x,y)\sim\rho}\,\log\mathbb E_{z\sim m_\theta(\cdot\mid x)} l(y,z).
$$
This is the exact analogue of cross-entropy: the log is outside the expectation because $p_\theta(y\mid x)$ is a *marginal likelihood*.

### Binary reasoning setup

In the binary reasoning setup, typical reward is the expected pass rate:
$$
J_{\mathrm{pass}}
=
\mathbb E_{(x,y)\sim\rho}
\,\mathbb E_{z\sim m_\theta(\cdot\mid x)}\mathbf 1[\hat y(z)=y]
=
\mathbb E_{(x,y)\sim\rho}\,p_\theta(y\mid x).
$$
Max-likelihood wants instead:
$$
J_{\mathrm{ML}}
=
\mathbb E_{(x,y)\sim\rho}\,\log p_\theta(y\mid x)
=
\mathbb E_{(x,y)\sim\rho}\,\log\mathbb E_{z\sim m_\theta(\cdot\mid x)} \mathbf 1[\hat y(z)=y].
$$
This is *exactly* the paper's core distinction: RL maximizes $\mathbb E[p]$, ML maximizes $\mathbb E[\log p]$.

(Also: yes, $\log\mathbb E[\mathbf 1[\cdot]]$ is a little silly-looking, but that's kind of the point: **you're doing maximum likelihood on a Bernoulli observation whose success probability is induced by a non-differentiable latent generator**.)

## Continuous regression

Specialize to a Gaussian noise model with $\sigma=1$: 
$$
l(y,z)=\hat P(y\mid z)=\mathcal N(\hat y(z))(y)
=
\frac{1}{\sqrt{2\pi}}
\exp\left[-\frac{(y-\hat y(z))^2}{2}\right].
$$
Then
$$
J(\theta)
=
\mathbb E_{(x,y)\sim\rho}
\,\log\mathbb E_{z\sim m_\theta(\cdot\mid x)}
\left[\exp\left(-\frac{(y-\hat y(z))^2}{2}\right)\right]
+\text{const}. 
$$

The "direct RL analogue" of pass-rate training is *not* MSE; it is expected *likelihood*:
$$
J_{\mathrm{direct}}
=
\mathbb E_{(x,y)\sim\rho}
\,\mathbb E_{z\sim m_\theta(\cdot\mid x)}\,l(y,z).
$$
This is the clean regression analogue of maximizing expected accuracy $\mathbb E[p]$ instead of log-likelihood $\mathbb E[\log p]$.

(If you want this to literally satisfy $l\in(0,1]$ so the Maclaurin series below is automatic, just pick a scale so the peak density is $\le 1$; multiplying $l$ by a $\theta$-independent constant only adds a constant to $\log p_\theta$, hence doesn't change the ML gradient.)

### Putting them together: Jensen and Taylor

From here on, fix $(x,y)$ and suppress dependencies to reduce clutter:
$$
p := p_\theta(y\mid x)=\mathbb E_{z\sim m_\theta(\cdot\mid x)} l(y,z).
$$

For $p\in(0,1]$, taylor expanding about $p=1$ yields 
$$
\log p
=
-\sum_{k=1}^\infty \frac{(1-p)^k}{k}.
$$
Note that this expansion is about $p=1$ (success), so $p\to 0$ implies greater deviation between the first (direct) and full (maximum-likelihood) orders, another perspective on larger improvements for harder tasks. 

Truncating to order $T$ gives the **compute-indexed MaxRL objective**
$$
J_T(p)
:=
-\sum_{k=1}^T \frac{(1-p)^k}{k}.
$$

- $T=1$: $J_1(p)=-(1-p)=p-1$, i.e. RL / pass-rate training up to an additive constant.
- $T\to\infty$: $J_T(p)\to\log p$, i.e. exact maximum likelihood.

Differentiating: 
$$
\nabla_\theta J_T(p_\theta)
=
\sum_{k=1}^T (1-p)^{k-1} \cdot \nabla p
=: w_T(p)\cdot \nabla p, \quad w_T(p)=\sum_{k=0}^{T-1}(1-p)^k
=\frac{1-(1-p)^T}{p}.
$$
As $T\to\infty$, $w_T(p)\to 1/p$, recovering ML's inverse-probability reweighting.
Now use the log-derivative trick on
$$
p=\mathbb E_{z\sim m_\theta}[l(y,z)]:
\qquad
\nabla_\theta p
= \mathbb E_{z\sim m_\theta(\cdot\mid x)}\big[l(y,z)\,\nabla_\theta\log m_\theta(z\mid x)\big]
=\mathbb E[lS].
$$
Therefore
$$
\nabla_\theta J_T(p)
=
 w_T(p)\,\mathbb E_{z\sim m_\theta(\cdot\mid x)}\big[l(y,z)\,S(x,z)\big].
$$

At this point you can already read off the two qualitative behaviors:

**Remark 1 (per-prompt):** inside $\mathbb E[lS]$, trajectories with larger likelihood $l(y,z)$ get more weight.

- Binary case: only successful rollouts contribute ($l\in\{0,1\}$).
- Gaussian regression: $l\propto \exp(-\mathrm{MSE}/(2\sigma^2))$ so the best rollouts dominate.

**Remark 2 (across prompts):** the scalar multiplier $w_T(p)$ increases when $p$ is small (hard prompts). In the ML limit $T\to\infty$, $w_T(p)=1/p$ reproduces the inverse-pass-rate reweighting.

---

## Gradient estimators

Here, you say: "all's great! There's this beautiful theory and a highly principled objective. How are we going about optimizing it?"

The MaxRL paper works out an unbiased estimator in the binary setting, and (crucially) shows it is unbiased for a **truncated** objective whose order matches your rollout compute.

Below: (i) recap the binary case, then (ii) give a clean generalization that operates at the level of the per-rollout likelihood $l(y,z)$.

### Binary case

In the binary setting $l(y,z)\in\{0,1\}$, draw $N$ trajectories $z_1,\dots,z_N\sim m_\theta(\cdot\mid x)$, define
$$
S_i:=\nabla_\theta\log m_\theta(z_i\mid x),\qquad K:=\sum_{i=1}^N l_i.
$$

- **REINFORCE** (pass@1) uses
$$
\hat g_{\mathrm{RL}}=\frac{1}{N}\sum_{i=1}^N l_i S_i,
$$
which is unbiased for $\nabla_\theta\,\mathrm{pass@}1(x)$.

- **MaxRL's key estimator** is: average scores over *successful* trajectories only,
$$
\hat g_N^{\mathrm{bin}}(x)
:=
\begin{cases}
\frac{1}{K}\sum_{i=1}^N l_i S_i, & K\ge 1,\\
0, & K=0.
\end{cases}
$$

Conditioning on $K\geq 1$, note that 
$$
\mathbb E_z\big[\hat g_N^{\mathrm{bin}}(x)\mid K\geq 1\big] = \mathbb E_z[S\mid l=1] = \dfrac{\mathbb E[l\cdot S]}{\mathbb E[l]} = \dfrac{\nabla p}{p} = \nabla \log p
$$
It's an unbiased ML estimator, conditioning on $K\geq 1$!! The bias comes from $K=0$ which happens with probability $(1-p)^N$. Substituting shows that this exactly equals the $T$-order truncated objective. 
$$
\mathbb E_z\big[\hat g_N^{\mathrm{bin}}(x)\big] = (1-(1-p)^N) \nabla \log p = w_T(p)\cdot \nabla p = \nabla J_T(p)
$$
## Generalization

Now we drop the assumption $l\in\{0,1\}$ and keep only what the derivation above *actually used*:

- $l_i := l(y,z_i)\in[0,1]$ is a per-rollout likelihood/reward signal,
- $p = \mathbb E[l]$,
- $\nabla_\theta p = \mathbb E[lS]$,
- $\nabla_\theta J_T = w_T(p)\,\nabla_\theta p$.

We want to **estimate the last quantity using $N$ i.i.d rollout samples**. The obstacle is subtle but standard:

- You can estimate $p$ and $\nabla_\theta p$ unbiasedly from samples.
- But $w_T(p)\nabla_\theta p$ is a **product** of unknowns.
- Plugging in sample estimates makes bias because the factors are correlated.

The key here is the **leave-one-out** trick. Use one sample to estimate $\nabla_\theta p$, the remaining $N-1$ samples to estimate $w_T(p)$, and average over all leave-one-outs. The product factorizes because samples are conditionally independent. 

Formally, **suppose we have an estimator subroutine $\omega_j$**, built from $\{l_j\}_{j\ne i}$ only, such that
$$
\mathbb E[\omega_i ] = w_T(p).
$$
We know that $l_i S_i$ is an unbiased estimator for $\nabla p = \mathbb E[l\cdot S]$, 
then the leave-one-out product $\omega_i l_i \cdot S_i$ is unbiased for $w_T(p)\nabla_\theta p$. Averaging over $i$ gives the final estimator:
$$
\hat g_T(x,y)
:=
\frac{1}{N}\sum_{i=1}^N \omega_i\,l_i\,S_i.
$$
In addition, this is a particularly neat expression because it is, again, a simple reweighted average of the rollout scores! To reduce variance, we can use the demeaning trick $\mathbb E[S] = 0$ to define 
$$
\hat g_T(x,y)
:=
\frac{1}{N}\sum_{i=1}^N \left(\omega_i\,l_i - \dfrac 1 N\right)\,S_i.
$$

> It remains to solve the problem: given i.i.d samples $(l_0, \dots, l_{N-1})$, estimate 
> $$w_T(p) = \sum_{k=0}^{T-1}(1-p)^k.$$
> Taking $T=N$, it suffices to obtain estimators for $(1-p)^k$ for all $1\leq k\leq N-1$. Fortunately, it turns out that this is a textbook problem with a known MVUE (Minimal Variance Unbiased Estimator) using U-statistics (something something elementary symmetric polynomials, you're welcome to look it up -- drop a comment!). 

## Pseudocode: generalized MaxRL for supervised likelihoods

I'll package **weight estimation** as a subroutine, as requested.

```python
# Generalized truncated Maximum likelihood RL for latent-generation likelihoods
#
# Inputs:
#   - model m_theta(z | x)
#   - per-rollout likelihood l(y, z) in [0, 1]   (can be scaled)
#   - truncation order T
#   - number of rollouts N >= T (T=1 corresponds to REINFORCE)
#


def grad_step(batch):
    # batch: list of (x, y) samples
    total_grad = 0.0

    for (x, y) in batch:
        # 1) sample rollouts
        z = [sample_from_model(m_theta, x) for _ in range(N)]

        # 2) compute per-rollout likelihoods and score vectors
        lvals = [likelihood_l(y, zi) for zi in z]                # l_i
        Svals = [grad_logprob(m_theta, zi, x) for zi in z]       # S_i = grad_theta log m_theta(zi|x)

        # 3) leave-one-out weights + accumulate weighted score average
        g = 0.0
        for i in range(N):
            l_loo = [lvals[j] for j in range(N) if j != i]       # N-1 samples
            omega_i = Estimate_wT(l_loo, T)                      # unbiased for w_T(p)
            g += omega_i * lvals[i] * Svals[i]

        g /= N
        total_grad += g

    total_grad /= len(batch)
    apply_update(theta, total_grad)
```