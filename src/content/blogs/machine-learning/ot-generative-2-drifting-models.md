---
title: "OT for generative modeling 2 — Wasserstein gradients and drifting models"
date: 2026-02-23
summary: "We look at Wasserstein Kaiming He et al.'s Drifting Models, interpret the antisymmetric drifting field as Wasserstein gradient flow on a kernelized discrepancy, and draw the connection to maximum likelihood estimation."
---

Flow matching both split the generative problem into two phases: at *training* time, learn a vector field; at *inference* time, integrate an ODE or SDE through that field to produce a sample. The integration is expensive — the reason we care about "number of function evaluations" (NFE) at all — and a great deal of recent work has gone into compressing it: distillation, consistency models, progressive reduction.

Recent work on **Drifting Models** ([He et al., 2026](https://arxiv.org/abs/2602.04770)) defines a "drifting field" $\mathbf{V}_{p,q}(\mathbf{x})$ that tells each generated sample which direction to move, requires antisymmetry ($\mathbf{V}_{p,q} = -\mathbf{V}_{q,p}$) so that the field vanishes at equilibrium ($p = q$), and train the generator to chase its own drifted targets.

We have a mechanical interpretation -- the loss going to zero produces desirable behavior. But equipped with the Wasserstein machinery we built in [parts 0](/blogs/machine-learning/ot-generative-0-static/)–[1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/), we can say something much sharper. The antisymmetric drifting field is the Wasserstein gradient of a distribution discrepancy; the training dynamics execute gradient descent on the manifold $\mathcal{W}_2$; and the $L^2$ regression loss against drifted targets is, through the Wasserstein bridge, a gradient step on a proper statistical divergence. The whole paradigm admits a precise connection to maximum likelihood.

## Contents

- [Formulation of drifting models](#formulation): mechanics of the paradigm.
- [Otto calculus](#otto-calculus): some theory connecting statistical divergences and euclidean flow.
- [Drifting as MLE](#drifting-as-mle)

## Formulation

The drifting models paradigm consists of:

- a **drifting field** that tells generated samples which direction to move
- a **training loop** that chases drifted targets

In this generative paradigm, we're given samples from $P_{\mathrm{data}} \in \mathcal W_2(\R^n)$, we consider an initial noise distribution $P_{\mathrm{noise}} \in \mathcal W_2(\R^d)$, and a parametric model generator $f_\theta: \mathbb R^d\to \mathbb R^n$. Denote the pushforward measure by $P_{\mathrm{model}} \in \mathcal W_2(\R^n)$. To be consistent with the paper, we abbreviate
$$
    p := P_{\mathrm{data}}, \quad q_\theta := P_{\mathrm{model}, \theta}
$$
The work considers general antisymmetric drift fields $V_{p, q}: \mathbb R^d\to \mathbb R^d$; each field is a vector field on the sample space. The training loop consists of iteratively minimizing

$$
    \mathcal L = \mathbb E_{\epsilon\sim P_{\mathrm{noise}}} \| f_\theta(\epsilon) - \text{stopgrad} \left[
        f_\theta(\epsilon) + V_{p, q_\theta}(f_\theta(\epsilon))
    \right]^2
$$
Note that $p=q \implies \mathcal L=0$.

### The drifting field

Let's consider the author's choice of the drifting field
:::definition[canonical drifting field]

Consider the following antisymmetric drifting field evaluated at sample space $x\in \mathbb R^n$:
$$
\begin{aligned}
    V_{p,q}(x)
    &=
    \underbrace{\frac{1}{Z_p(x)}\E_{y^+\sim p}\!\left[k(x, y^+)(y^+ - x)\right]}_{V_p^+(x):\;\text{attraction to data}} \;-\; \underbrace{\frac{1}{Z_q(x)}\E_{y^-\sim q}\!\left[k(x, y^-)(y^- - x)\right]}_{V_q^-(x):\;\text{repulsion from model}} \\
    &= \dfrac{1}{Z_pZ_q} \mathbb E_{y^+\sim p, y^-\sim q} \left[
        k(x, y^+) k(x, y^-)(y^+ - y^-)
    \right]
\end{aligned}
$$
The kernel is Laplace-weighted
$$
    k(x,y) = \exp(-\|x - y\|/\tau)
$$
The per-point normalization translates to softmax weighting
$$
    Z_p(x) = \E_{y^+\sim p}[k(x, y^+)], \quad Z_q(x) = \E_{y^-\sim q}[k(x, y^-)]
$$
:::
Let's unpack this, $k(x, y^{\pm})$ is a smoothing kernel over samples of $y^\pm - x$. In the large-sample precise limit $\tau\to 0$, we obtain
$$
    \mathbb E_{x\sim q} [V_p^+(x)] = \mathbb E_{x\sim q, y^+\sim p} (y^+ - x)
$$


## Wasserstein Gradient Flow

We take a step back to develop the theory of **Wasserstein gradient flow** (and return to capital letters): given a probability distribution $Q_\theta$, we can assign some loss / preference to it by some functional $\mathcal F:\mathcal W_2\to \R$. What happens to the probability distribution as we try to minimize $\mathcal F(q_\theta)$ by gradient descent?

### The Kullback-Leibler functional

Fixing data distribution $p$, maximimizing likelihood of the data under model is equivalent to maximizing the KL functional:
$$
\begin{aligned}
    \mrm{KL}(P\|Q_\theta)
    &= \mathbb E_{x\sim P} \left[
        \log \dfrac{Q_\theta(x)}{P(x)}
    \right] \\
    &= \mathbb E_{x\sim P} \log Q_\theta(x) - \mathbb E_{x\sim P}\log P(x)\\
    &= \int dP\, \log Q - \int dP\, \log P
\end{aligned}
$$
For more interesting properties of KL divergence, see [these notes](https://nlyu1.github.io/classical-info-theory/kullback-leibler-divergence.html). From a SGD perspective, minimizing KL is equivalent to maximizing likelihood when empirical samples are i.i.d from $P$:
$$
    \nabla_\theta \, \mathrm{KL}(P\|Q_\theta) = \nabla_\theta \mathbb E_{x\sim P} \, \log Q_\theta(x)
$$


### Gradients on manifolds

I like to interpret differential geometry as the "lifting" of Euclidean constructs into locally Euclidean manifolds. Gradients are no different. In Euclidean space, given a curve $\gamma: [0, 1]\to \R^n$ and a linear function $f:\R^n\to \R$, the chain rule yields
$$
    \df d {dt} f(\gamma(t)) = \dot \gamma(t) \cdot \nabla\big|_{\gamma(t)} f
$$
Lift the inner product to the manifold metric, can use this to define gradients on manifolds:

:::definition[Wasserstein gradients]

Given a scalar function $f:\mathcal W_2\to \R$, the gradient of $f$ at the point $P\in \mathcal W_2$ is the unique tangent vector $v=\mathrm{grad}_W f(P)$ such that, for any curve $\gamma(t)$ with $\gamma(t)=x$, we have
$$
    \dfrac{d}{dt} f(\gamma(t)) = \la \dot \gamma(t), v\ra_W = \mathbb E_P \la \dot \gamma(t), v\ra
$$
where $\la\cdot, \cdot\ra_W$ is our familiar Wasserstein metric on $\mathcal W_2$, and $\la\cdot, \cdot\ra$ in the second equality is the familiar Euclidean metric, after we expanded the definition of the Wasserstein metric.
:::

Now, we're equipped to state a major result in [Otto calculus](https://www.math.toronto.edu/mccann/assignments/477/Otto01.pdf). We'll prove it shortly

:::theorem[fundamental theorem of Otto calculus]
Given a probability functional $\mathcal F:\mathcal W_2\to \R$, its Wasserstein gradient can be computed as
$$
    \mathrm{grad}_W \mathcal F(P) = \nabla_x \left(\dfrac{\delta \mathcal F}{\delta P}\right)
$$
The theorem should look fairly intuitive: on the RHS, we compute the pointwise derivative of $\mathcal F$ w.r.t. the point density at $P(x)$ and use this as the potential. The theorem tells us that the direction of steepest $\mathcal F$-ascent is the gradient of this (functional derivative) potential.
:::

Several remarks are in order:
1. The definition is [definition eqref]. This is a theorem (equation), not a definition! The general differential-geometry gradient exists, but its general computation does not usually admit such easy form.
2. The expression $\dfrac{\delta\mathcal F}{\delta P}: \mathbb R^n\to \R$ is **a scalar function on the sample space** that's usually known as the **functional derivative**. Its values are the point-wise derivatives of $F$ w.r.t. $P$.

Again, the functional derivative **is a scalar function on the sample space**. Its definition is best demonstrated by two useful examples:

:::example
For the entropy functional $\mathcal H(P) = -\int dP\, \log P = -\int P(x) \log P(x) dx$, applying the product rule yields:
$$
    \dfrac{\delta \mathcal H(P)}{\delta P}(x) = -(\log P(x) + 1)
$$

The KL functional $\mathrm{KL}(P\|Q_\theta)$ has two arguments:
$$
    \mathrm{KL}(P\|Q_\theta) = \int dP\, \log P - \int dP \log Q_\theta
$$

Taking the functional derivative w.r.t. the data distribution $P$, we treat $Q_\theta$ as a constant:
$$
\begin{aligned}
    \dfrac{\delta \mathrm{KL}(P\|Q_\theta)}{\delta P}(x)
    &= \pd P\Big( P \log P - P \log Q_\theta \Big) \\
    &= \log P(x) - \log Q_\theta(x) + 1
\end{aligned}
$$
Similarly w.r.t. $Q_\theta$:
$$
\begin{aligned}
    \dfrac{\delta \mathrm{KL}(P\|Q_\theta)}{\delta Q_\theta}(x)
    &= \partial_{Q_\theta} \Big( - P \log Q_\theta \Big) = -\dfrac{P(x)}{Q_\theta(x)}
\end{aligned}
$$
:::

:::example[applying Otto's theorem to entropy]

From above, $\frac{\delta \mathcal H}{\delta P} = -(\log P + 1)$. Applying the theorem:
$$
    \mrm{grad}_W \mathcal H(P) = \nabla_x\left(-\log P - 1\right) = -\nabla \log P
$$
The Wasserstein gradient of entropy is the **negative score**. Gradient ascent on entropy has velocity $v = \mrm{grad}_W\mathcal H = -\nabla \log P$. Plugging into the continuity equation:
$$
    \pd t P = \nabla \cdot(P\,\nabla \log P) = \Delta P
$$
This is the **heat equation**: heat diffusion is Wasserstein gradient ascent of entropy.
:::

:::example[applying Otto's theorem to forward KL]

Apply to $Q_\theta$ in $\mrm{KL}(P\|Q_\theta)$. From above, $\frac{\delta\mrm{KL}(P\|Q_\theta)}{\delta Q_\theta} = -P/Q_\theta$. Applying the theorem:
$$
    \mrm{grad}_W \mrm{KL}(P\|Q_\theta)\big|_{Q_\theta} = \nabla_x\!\left(-\frac{P}{Q_\theta}\right)
$$
Gradient descent velocity: $v = \nabla_x(P/Q_\theta)$. Particles flow toward regions where the **density ratio** $P/Q_\theta$ increases. In practice, the density ratio is expensive to estimate, which is one reason forward KL is rarely minimized by direct Wasserstein gradient descent.
:::

:::example[applying Otto's theorem to reverse KL]

Apply to $Q_\theta$ in $\mrm{KL}(Q_\theta\|P)$. The functional derivative is
$$
    \frac{\delta\mrm{KL}(Q_\theta\|P)}{\delta Q_\theta} = \log Q_\theta - \log P + 1
$$
Applying the theorem:
$$
    \mrm{grad}_W \mrm{KL}(Q_\theta\|P)\big|_{Q_\theta} = \nabla \log Q_\theta - \nabla \log P = s_{Q_\theta} - s_P
$$
The Wasserstein gradient is a **score difference**. Gradient descent velocity: $v = s_P - s_{Q_\theta}$. Particles flow in the direction where the data score exceeds the model score. This is exactly the drifting field's structure — but it requires $\nabla \log P$, the data score. With empirical samples (Diracs), this is undefined: the **Dirac trap** that we return to in [the next section](#why-not-kl-the-dirac-trap).
:::

### Proving Otto's theorem

We need to show that $v = \nabla_x \frac{\delta \mathcal F}{\delta P}$ satisfies the gradient definition for all test tangent vectors $u = \nabla\psi \in T_P\mathcal W_2$.

By the gradient definition, $v = \mrm{grad}_W\mathcal F$ is the unique tangent vector satisfying
$$
    \df d{d\epsilon}\mathcal F(P + \epsilon\,\delta P)\Big|_{\epsilon=0} = \la u, v\ra_W = \int \la u, v\ra\, dP
$$
for all test velocities $u$, where $\delta P$ is the perturbation of $P$ induced by flowing along $u$. By the continuity equation, an infinitesimal flow along $u$ produces perturbation $\delta P = \pd t P= -\nabla \cdot (P\, u)$. By the definition of the functional derivative,
$$
    \df d{d\epsilon}\mathcal F(P + \epsilon\,\delta P)\Big|_{\epsilon=0} \equiv \int \frac{\delta \mathcal F}{\delta P}\,\delta P\, dx = -\int \frac{\delta \mathcal F}{\delta P}\,\nabla\cdot(P\, u)\, dx
$$
Apply the divergence theorem, the boundary term vanishes by
$$
\begin{aligned}
    &= \cancel{-\int \nabla \cdot \!\left[\frac{\delta \mathcal F}{\delta P}\, P\, u\right] dx} \;+\; \int u\cdot \nabla \left(\frac{\delta \mathcal F}{\delta P}\right)\, dP \\
    &= \int \la u,\, \nabla\frac{\delta \mathcal F}{\delta P}\ra\, dP \implies \mathrm{grad}_W(\mathcal F) := v = \nabla \left(\dfrac{\delta \mathcal F}{\delta P}\right) \,\, \square
\end{aligned}
$$

## Drifting as MLE

We now connect the drifting field to Wasserstein gradient flow and ask: what functional is being minimized, and what does this have to do with maximum likelihood?

### The antisymmetric constraint as gradient flow signature

The antisymmetry $V_{p,q} = -V_{q,p}$ is not just a convenient property — it is the **structural signature of a gradient flow** on a symmetric discrepancy.

If a functional $\mathcal D(p, q)$ is symmetric in its arguments, then its first variation with respect to $q$ satisfies $\frac{\delta \mathcal D}{\delta q}(p, q) = -\frac{\delta \mathcal D}{\delta q}(q, p)$. Lifting to the tangent space via $\nabla$, the resulting velocity field inherits the antisymmetry. The drifting field's antisymmetry therefore guarantees that it is a **conservative gradient field** — the system flows down a well-defined potential landscape toward the equilibrium $p = q$.

### Why not KL: the Dirac trap

A natural hypothesis: the drifting field is the Wasserstein gradient of the KL divergence, and drifting models are doing MLE. The geometric structure is right — the antisymmetry, the equilibrium — but there is a fatal obstruction.

Consider $\mrm{KL}(q \| p)$ (reverse KL, minimized over $q$). Its Wasserstein gradient involves $\nabla \log p$ — the score of the data distribution. With only empirical samples, $p = \frac{1}{N}\sum_i \delta(x - x_i)$, and $\nabla \log p$ is undefined (the gradient of a Dirac delta is a distribution, not a function). This is the **Dirac trap**: you cannot compute KL gradients from raw samples without smoothing.

This is precisely why diffusion models add noise — convolving with Gaussians smooths the Diracs, making the score $\nabla \log \rho_t$ computable everywhere. Drifting models operate with only empirical samples. The functional **cannot** be KL.

### The functional is MMD

The drifting field's kernel structure matches the Wasserstein gradient of **Maximum Mean Discrepancy**:

:::definition[MMD]
$$
    \mrm{MMD}^2(p, q) = \E_{x,x'\sim p}[k(x, x')] - 2\,\E_{x\sim p,\, y\sim q}[k(x, y)] + \E_{y,y'\sim q}[k(y, y')]
$$
:::

The Wasserstein gradient of $\mrm{MMD}^2$ with respect to the second argument $q$, evaluated at a point $x$ in the support of $q$, is:
$$
    \nabla_W \mrm{MMD}^2\big|_x \;\propto\; \E_{y\sim p}[\nabla_x k(x, y)] - \E_{y\sim q}[\nabla_x k(x, y)]
$$

This is attraction toward data minus repulsion from model — precisely the drifting field's structure. MMD is symmetric, so antisymmetry comes for free. And MMD is computable from empirical samples with no Dirac trap.

:::remark[The normalization]
The vanilla MMD gradient uses uniform averages. The paper's drifting field additionally normalizes by $Z_p(x), Z_q(x)$, converting uniform kernel averages into softmax-weighted mean shifts. This likely corresponds to minimizing a **normalized** kernel discrepancy (a ratio of kernel expectations rather than a difference). The normalization improves numerical stability and may change the implicit functional; pinning down the exact modified functional is an open question.
:::

### Connection to MLE

Drifting models are **not** standard MLE — they minimize a kernel discrepancy (MMD), not a likelihood. But we can still draw precise connections.

**Characteristic kernels and convergence.** If the kernel $k$ is *characteristic* (which exponential kernels are on compact domains), then $\mrm{MMD}(p, q) = 0$ iff $p = q$. At convergence, the model distribution is exact. This is stronger than a bound — it is an exact zero of the divergence.

**The training dynamics on $\mathcal W_2$.** Each training step computes the drifting field (an approximation to the Wasserstein gradient of MMD), shifts the targets, and trains the generator to output the shifted targets. Over training iterations, the pushforward distribution $q_\theta$ physically crawls across $\mathcal W_2$ toward $p_{\mrm{data}}$. The training dynamics are Wasserstein gradient descent on MMD, with the step size absorbed into the kernel bandwidth $\tau$.

**Three paradigms, three functionals.** We can now cleanly compare the three major generative paradigms through the Wasserstein lens:

| | **Functional** | **Training** | **Inference** |
|---|---|---|---|
| **Diffusion** | KL to noise (WGF) | Learn score $\nabla \log \rho_t$ | Reverse SDE/ODE (many NFE) |
| **Flow matching** | Benamou-Brenier action | Learn geodesic velocity field | Integrate ODE (few NFE) |
| **Drifting** | MMD (WGF) | Chase drifted targets | Single forward pass (1 NFE) |

**Open questions.** Can we get a formal convergence rate for the training dynamics? Talagrand-type inequalities could bridge MMD $\to$ $W_2$ $\to$ KL, but the per-point normalization complicates the analysis. The relationship between the kernel bandwidth $\tau$ and the effective step size on $\mathcal W_2$ also deserves formalization.
