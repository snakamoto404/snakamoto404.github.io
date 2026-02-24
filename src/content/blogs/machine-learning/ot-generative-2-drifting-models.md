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

The drifting models paradigm has three ingredients: a **drifting field** that tells generated samples which direction to move, a **training loop** that chases drifted targets, and **1-NFE inference** — a single forward pass through the generator, no ODE integration required.

### The drifting field

Given a data distribution $p$ and a model distribution $q$ (the pushforward of noise through the generator $f_\theta$), the drifting field decomposes into attraction toward data and repulsion from the model's own samples:

:::definition[Drifting field]
$$
    V_{p,q}(x) = \underbrace{\frac{1}{Z_p(x)}\E_{y^+\sim p}\!\left[k(x, y^+)(y^+ - x)\right]}_{V_p^+(x):\;\text{attraction to data}} \;-\; \underbrace{\frac{1}{Z_q(x)}\E_{y^-\sim q}\!\left[k(x, y^-)(y^- - x)\right]}_{V_q^-(x):\;\text{repulsion from model}}
$$
where $k(x,y) = \exp(-\|x - y\|/\tau)$ is an exponential kernel with temperature $\tau$, and $Z_p(x) = \E_{y^+\sim p}[k(x, y^+)]$, $Z_q(x) = \E_{y^-\sim q}[k(x, y^-)]$ are per-point normalizations (softmax weights).
:::

Each term is a kernel-weighted mean shift: $V_p^+$ points $x$ toward nearby data, $V_q^-$ points $x$ toward nearby model samples. Their difference pushes generated samples toward data and away from other generated samples.

Two structural properties are immediate:

:::proposition
**Antisymmetry.** $V_{p,q}(x) = -V_{q,p}(x)$ for all $x$. (Swap the roles of $p$ and $q$; attraction and repulsion trade places.)
:::

:::proposition
**Equilibrium.** $p = q \implies V_{p,q}(x) = 0$ for all $x$. (When model matches data, the two terms cancel.)
:::

### Training

The generator $f_\theta: \R^n \to \R^n$ maps noise $\epsilon\sim p_{\mrm{noise}}$ to samples $x = f_\theta(\epsilon)$. Training proceeds by:

1. Generate a batch: $x = f_\theta(\epsilon)$
2. Compute the drifting field $V = V_{p,q}(x)$ from a data batch $y^+ \sim p$ and the generated batch $y^- = x$
3. Form **drifted targets**: $\tilde x = \mrm{stopgrad}(x + V)$
4. Minimize $\mathcal L = \|f_\theta(\epsilon) - \tilde x\|^2$

The gradient does **not** flow through $V$ — the `stopgrad` is crucial. Without it, the generator could minimize the loss by making $V$ vanish (collapsing the field) rather than by actually matching data.

:::remark
The loss simplifies to $\|V(f_\theta(\epsilon))\|^2$: the squared magnitude of the drifting field at the generator's output. The loss is zero iff the drifting field vanishes everywhere the model places mass — which, by the equilibrium property, happens when $q_\theta = p$.
:::

### Inference

At test time, sampling is a single forward pass: $x = f_\theta(\epsilon)$ for $\epsilon \sim p_{\mrm{noise}}$. This is **1-NFE** — no iterative ODE/SDE integration. The "integration" has been absorbed into the training iterations: over the course of SGD, the pushforward distribution $q_\theta$ physically migrates across distribution space toward $p$.

:::remark[Comparison with flow matching]
Flow matching learns a vector field at training time, then integrates an ODE at inference time. Drifting models learn by iteratively chasing drifted targets at training time — absorbing the transport dynamics into SGD — and require only a single forward pass at inference.
:::

## Otto calculus

We now develop the theory of **Wasserstein gradient flow**: how to compute the "steepest descent direction" of a functional on the distribution manifold $\mathcal W_2$. This is the machinery that will let us interpret the drifting field geometrically.

### Functionals on $\mathcal W_2$

A **functional** $\mathcal F: \mathcal W_2 \to \R$ maps a distribution to a scalar. Gradient flow on $\mathcal W_2$ means finding the velocity field $v$ that makes $\rho_t$ descend $\mathcal F$ steepest under the Wasserstein metric.

The central example is the **KL divergence** to a target distribution $\rho_\infty(x) \propto \exp(-U(x))$:
$$
    \mrm{KL}(\rho \| \rho_\infty) = \underbrace{\int \rho \log \rho\, dx}_{\text{entropy } H[\rho]} \;+\; \underbrace{\int \rho\, U\, dx}_{\text{potential energy}} \;+\; \mrm{const}
$$
The entropy term penalizes concentration; the potential energy term pulls mass toward the minima of $U$ in the sample space.

### The Otto calculus pipeline

Otto calculus is a three-step pipeline that converts a scalar functional on $\mathcal W_2$ into a concrete flow equation on $\R^n$.

**Step 1 — First variation.** Compute the Fréchet derivative $\frac{\delta \mathcal F}{\delta \rho}$: how does $\mathcal F$ change under an infinitesimal perturbation of $\rho$? The result is a **scalar function on $\R^n$**.

**Step 2 — Lift to the tangent space.** Apply the spatial gradient $\nabla$ to obtain a vector field on $\R^n$. This is the **Wasserstein gradient**:
$$
    \nabla_W \mathcal F = \nabla \frac{\delta \mathcal F}{\delta \rho}
$$
Recall from [Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/) that tangent vectors on $\mathcal W_2$ are gradient vector fields on $\R^n$. Step 2 lifts a scalar (the first variation) into a valid tangent vector.

**Step 3 — Continuity equation.** The gradient flow velocity is $v = -\nabla_W \mathcal F$ (negative gradient = steepest descent). The density evolves by the continuity equation:
$$
    \pd t \rho + \nabla \cdot (\rho\, v) = 0 \qquad \Longrightarrow \qquad \pd t \rho = \nabla \cdot \!\left(\rho\, \nabla \frac{\delta \mathcal F}{\delta \rho}\right)
$$

This is the **abstract Wasserstein gradient flow equation**. Everything hinges on computing $\frac{\delta \mathcal F}{\delta \rho}$ for the functional of interest.

### Full derivation: KL $\to$ Fokker-Planck

Let's run the pipeline on $\mathcal F[\rho] = \mrm{KL}(\rho \| \rho_\infty)$ with $\rho_\infty \propto \exp(-U)$.

**Step 1.** The first variation of KL is:
$$
    \frac{\delta}{\delta \rho}\mrm{KL}(\rho \| \rho_\infty) = \log \rho + U(x) + 1
$$
(The $+1$ comes from $\frac{\delta}{\delta \rho}\int \rho \log \rho = \log \rho + 1$; it's a constant and will be killed by $\nabla$ in Step 2.)

**Step 2.** The Wasserstein gradient is:
$$
    \nabla_W \mrm{KL} = \nabla(\log \rho + U) = \underbrace{\nabla \log \rho}_{\text{Stein score}} + \nabla U
$$
Note how the **Stein score** $\nabla \log \rho$ appears naturally — it is not introduced ad hoc but arises inevitably from the entropy term of KL under Otto calculus.

**Step 3.** The gradient flow velocity is $v = -\nabla \log \rho - \nabla U$. Plugging into the continuity equation:
$$
    \pd t \rho = -\nabla \cdot(\rho\, v) = \nabla\cdot\bigl(\rho\,\nabla\log \rho + \rho\,\nabla U\bigr)
$$

The first term simplifies: $\nabla \cdot (\rho\, \nabla \log \rho) = \nabla \cdot (\rho \cdot \frac{\nabla \rho}{\rho}) = \nabla \cdot (\nabla \rho) = \Delta \rho$, giving:

:::theorem[Wasserstein gradient flow of KL]
$$
\begin{equation}
    \pd t \rho = \Delta \rho + \nabla \cdot (\rho\, \nabla U)
    \label{eq:fokker-planck}
\end{equation}
$$
This is the **Fokker-Planck equation**. Setting $U = 0$ (target = uniform / maximum entropy) recovers the **heat equation** $\pd t \rho = \Delta \rho$: diffusion is the Wasserstein gradient flow of entropy.
:::

:::remark[Two descriptions, one physics]
On the manifold $\mathcal W_2$, the Fokker-Planck flow is **deterministic**: the entire distribution $\rho_t$ slides smoothly down the KL landscape toward $\rho_\infty$. In the sample space $\R^n$, the same process looks **stochastic**: individual particles follow Langevin dynamics $dx = -\nabla U\, dt + \sqrt{2}\, dW_t$, kicked by Brownian noise. Same physics, two complementary descriptions.
:::

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
