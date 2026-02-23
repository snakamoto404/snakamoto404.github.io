---
title: "OT for generative modeling 0 — the static perspective"
date: 2026-02-23
summary: "Part 0 of a series on optimal transport in generative models. We build simple intuition: the earth-moving problem, the Kantorovich formulation, and why the dual characterization leads directly to Wasserstein-GAN."
---

Generative modeling is, at bottom, a problem about distributions: you have one (noise), you want another (data), and you need a *principled* way to push one toward the other. But what does "principled" mean here? Cross-entropy? KL divergence? Adversarial discrimination?

There is a tension at the heart of the question. On one side, Euclidean geometry gives us clean, computable objectives — $L^2$ regression, mean squared error — but these are defined on *points*, not distributions. On the other side, information-theoretic divergences like KL are proper statistical distances between distributions, but they blow up whenever supports don't overlap (try computing $\text{KL}(p \| q)$ when $p$ puts mass where $q$ doesn't). Generative models live in the gap: we write $L^2$ losses in PyTorch and somehow distributions converge. *Why does this work?*

The answer is **optimal transport**. The question it asks is deceptively simple — given two piles of mass, what is the cheapest way to rearrange one into the other? — but the theory that falls out of it provides the critical bridge between Euclidean geometry and statistical divergences. The Wasserstein distance is well-defined for arbitrary measures (no absolute continuity required), inherits geometric structure from the base space (geodesics, curvature, gradients), and — crucially — allows us to understand $L^2$ regression training objectives as gradient steps on proper statistical divergences in the space of distributions. This is the framework that makes score matching, flow matching, and drifting models *principled*, not just empirically effective.

This series builds the optimal transport toolkit from scratch, with one eye on the mathematics and the other on the generative models it enables. Two themes run through everything:

1. **The Wasserstein bridge.** Optimal transport connects Euclidean geometry to statistical divergences. This empowers us to understand training processes with $L^2$ regression objectives through the lens of proper distributional distances.
2. **Static–dynamic unification.** The equivalence between static transport plans and dynamic fluid flows hinges on a clean decomposition: solve the single-particle Lagrangian (Euler-Lagrange → straight line), then marginalize. This is the engine under conditional flow matching and the key to making the theory computable.

The arc:

- **Part 0** (this post): the *static* picture — transport plans, the Kantorovich formulation, and the variational dual that gives us Wasserstein-GAN.
- **[Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/)**: the *geometric* picture — the space of probability distributions is itself a Riemannian manifold, and optimal transport defines its metric.
- **[Part 2](/blogs/machine-learning/ot-generative-2-wasserstein-gradients/)**: the *dynamic* picture — action, Lagrangians, the Benamou-Brenier bridge between static and dynamic transport, and Otto calculus for doing gradient descent on this manifold.
- **[Part 3](/blogs/machine-learning/ot-generative-3-drifting-models/)**: the *application* — Kaiming He et al.'s Drifting Models, interpreted as Wasserstein gradient flow, and their connection to MLE.

Part 0 is deliberately elementary. The goal is to gain *simple, stable intuition* for what optimal transport is and why it is the right language for comparing distributions — before we touch any geometry or calculus. If you already know the Kantorovich dual, you can safely skip to Part 1.

## Contents

- [The earth-moving problem and transport plans](#the-earth-moving-problem-and-transport-plans)
- [The Kantorovich formulation](#the-kantorovich-formulation)
- [Variational characterization and Wasserstein-GAN](#variational-characterization-and-wasserstein-gan)

## The earth-moving problem and transport plans

*Coming soon.*

## The Kantorovich formulation

*Coming soon.*

## Variational characterization and Wasserstein-GAN

*Coming soon.*
