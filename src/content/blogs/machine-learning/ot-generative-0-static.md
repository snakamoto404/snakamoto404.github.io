---
title: "OT for generative modeling 0 — the static perspective"
date: 2026-02-23
summary: "Why we care about optimal transport (OT), and introduction to the static viewpoint. Variational characterization, and WGAN."
---

At the heart of modern generative modeling, there is a fundamental tension between two spaces. On **sample space** $\mathbb R^n$, Euclidean geometry gives us clean, benign objectives such as $L^2$ mean squared error. On the **space of distributions** $\mathcal P(\mathbb R^n)$ maximum likelihood singles out the Kullback-leibler divergence as the principled objective, and $\mathcal P(\mathbb R^n)$ has its own geometry; however, KL is remarkably agnostic towards the internal geometry of $\mathbb R^n$ and, in particular, the inductive biases associated with it.

**Optimal transport** provides one of the most mathematically beautiful bridge between the two spaces. It has also become the backbone of modern generative modeling. My personal acquaintance with the field began with [Wasserstein GAN](https://arxiv.org/pdf/1701.07875) (more on it later). In school, [information theory](https://nlyu1.github.io/classical-info-theory/) provided the necessary tools, and I've always been intrigued by the connections between e.g. flow matching, wasserstein gradient flow, and even [renormalization](https://arxiv.org/abs/2202.11737). Motivated by understanding a highly impressive recent work on [drifting models](https://arxiv.org/abs/2602.04770), I decided to write these posts to learn, and unpack, some of the concepts.

This series builds the optimal transport toolkit from scratch, with one eye on the mathematics and the other on the generative models it enables. It aspires to be curiosity-driven and hand-waves some epsilons and deltas. To start off, two themes run through everything, we've already seen one:

1. **Bridging sample and distribution geometries.** Optimal transport connects Euclidean geometry to statistical divergences. This empowers us to understand training processes with benign $L^2$  objectives through the lens of proper distributional optimization.
2. **Bridging static and dynamic definitions.** The equivalence between static, closed-form quantities (e.g. transport cost) and dynamic quantities (e.g. integral of kinetic energy) hinges on a clean decomposition: solve the single-particle action-extremizing path, then marginalize over a fixed distribution. This is the engine under Benamou-Brenier and the key to making flow matching computable using conditional flow matching.

Layout of the posts:

- **Part 0** (this post): the *static* picture — transport plans, the static definition of Wasserstein distance, and the variational dual that empowers Wasserstein-GAN.
- **[Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/)**: the *geometric* picture — the Wasserstein distribution manifold provides a *dynamic* definition of $W_2$ distance; gradients, geodesics.
- **[Part 2](/blogs/machine-learning/ot-generative-2-wasserstein-gradients/)**: the *unifying* picture — Benamou-Brenier theorem as the bridge between static and dynamic transport. Conditional and marginal flow matching.
- **[Part 3](/blogs/machine-learning/ot-generative-3-drifting-models/)**: *application* of the optimal transport perspective to drifting models.

## Contents

- [The earth-moving problem and transport plans](#the-earth-moving-problem-and-transport-plans)
- [The static (Kantorovich) formulation](#the-static-formulation)
- [Variational characterization and Wasserstein-GAN](#variational-characterization-and-wasserstein-gan)

## Transport plans

We work on $\mathbb R^d$. Interpret a probability distribution $\rho$ on $\mathbb R^d$ as a snapshot of a fluid[^earth] on $\mathbb R^d$, with $f_\rho(x)$ denoting the height of water at $x\in \mathbb R^d$; water has uniform density, so the mass of water in each region is proportional to the enclosed distribution probability.

[^earth]: "Earth-moving" language is historically standard, but we will prefer the fluid metaphor because later sections use velocity fields, momentum, and continuity equations.

## The static formulation

*Coming soon.*

## Variational characterization and Wasserstein-GAN

*Coming soon.*
