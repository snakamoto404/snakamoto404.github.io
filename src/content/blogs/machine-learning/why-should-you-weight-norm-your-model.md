---
title: "Why should you weight-norm your model?"
date: 2026-10-01
summary: "Some surprising consequences of abundamently using RMSNorm in frontier architectures and how to fix them."
---

Weight-norm has been a hot gossip topic, and dozens of papers have been written about it w.r.t. different implementations and motivations. Here, I pick my favorite first-principles interpretation and implementation. Tl; dr:

- Since we put norms everywhere, most matrix parameters of our model are (approx) scalar scale-invariant $W\leftrightarrow cW$. This gauge is the first-principles reason for weight-norm tricks.
- For scalar-invariant $W$, the effective lr is actually $\eta/\|W\|$.
- We want to control effective-lr downstream of loss curvature and sharpness regularization, instead of in a coupled manner. This motivates **fixing parameters' weight norms**.
- The hyperball keeps constant weight-norm, gets rid of weight decay, and allows us to choose any optimizer in a decoupled manner
$$
    W \leftarrow R\cdot \mathrm{Normalize}\left(W - \eta \cdot R\cdot \mathrm{Normalize}(U)\right)
$$


<details>
<summary>References</summary>

1. Hyperball's first author [Kaiyue Wen's blog post](https://whenwen.github.io/wd_blog/public/weight-decay-part-2.html)
2. Fantastic PT optimizers and where to find them, [part 1](https://arxiv.org/abs/2509.02046) and [part 2](https://arxiv.org/abs/2606.16899).
3. Lecture 5 from [Stanford's CS312](https://deep-learning-alchemy.github.io/).

</details>

## Some details

The output of a typical linear layer $W$ is usually processed by an RMSNorm:
- Attention Q / K projections $\implies$ QK-norm. There exists exact symmetry here.
- Output-proj $\implies$ pre-norm of subsequent blocks. Coupled, approximate symmetry here.

When $W$ has scalar symmetry, i.e. $L(W) = L(cW)$, there are several nontrivial consequences.

:::proposition[consequences of scalar symmetry]
When $L(W) = L(cW)$:
1. The angular $[\mathrm{numel}(W) - 1]$ degrees of freedom affect the loss, so the effective lr is
$$
 \eta_{\mathrm{eff}} = \eta / \|W\|
$$
2. The **gradient $G = \nabla_W L$ is orthogonal to $W$**.
:::
Recall the standard optimizer update with decoupled weight-decay $\lambda$:
$$
    W' \leftarrow (1 - \eta \lambda)W - \eta G
$$
Schematically, a 2D projection looks like

<figure class="theme-figure">
  <img class="figure-light" src="/plots/weight-norm-light.svg" width="500" height="244" loading="lazy" alt="Scale-invariant loss geometry with a vertically compressed constant-norm arc. The gradient is tangent; weight decay points inward along faint radial contours." />
  <img class="figure-dark" src="/plots/weight-norm-dark.svg" width="500" height="244" loading="lazy" alt="Scale-invariant loss geometry with a vertically compressed constant-norm arc. The gradient is tangent; weight decay points inward along faint radial contours." />
</figure>

Since loss doesn't vary across the radial direction, $\la G, W\ra = 0$. Then
$$
    \|W'\|_2^2 = (1-\lambda)^2 \|W\|^2 + \eta^2 \|G\|^2
$$
When $\lambda=0$, the norm of $W$ will just keep growing and decrease the effective lr. Here's an example dynamic of one such matrix at different lr's, fixing the same weight-decay.

<figure class="theme-figure" style="max-width: 100%;">
  <img class="figure-light" src="/plots/weight-norm-schedule-light.svg" width="1000" height="355" loading="lazy" alt="Three panels versus step: learning rate, weight RMS, and effective learning rate in RMS units. Blue, green, and red denote peak learning rates 0.0015, 0.003, and 0.006. Weight norms grow and then fall, changing the effective learning-rate schedules." />
  <img class="figure-dark" src="/plots/weight-norm-schedule-dark.svg" width="1000" height="355" loading="lazy" alt="Three panels versus step: learning rate, weight RMS, and effective learning rate in RMS units. Blue, green, and red denote peak learning rates 0.0015, 0.003, and 0.006. Weight norms grow and then fall, changing the effective learning-rate schedules." />
</figure>

Some observations:
1. Adjusting the lr also implicitly adjusts the effective lr schedule.
2. The effective lr schedule is very sharp after initial-lr. Not obvious this is desirable behavior.

A good decoupled fix is the hyperball prescription:

1. Choose a good $\mu P$ initialization and norm $R$ (from $\mu P$ or bootstrapping from existing model).
2. Choose any base optimizer $W' \leftarrow (1 - \eta \lambda)W - \eta \, U$.
3. Boostrap the update rule
$$
    W \leftarrow R\cdot \mathrm{Normalize}\left(W - \eta \cdot R\cdot \mathrm{Normalize}(U)\right)
$$
