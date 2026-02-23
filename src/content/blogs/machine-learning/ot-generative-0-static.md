---
title: "OT for generative modeling 0 — the static perspective"
date: 2026-02-23
summary: "Part 0 of a series on optimal transport in generative models. We build simple intuition: the earth-moving problem, the Kantorovich formulation, and why the dual characterization leads directly to Wasserstein-GAN."
---

Generative modeling is, at bottom, a problem about distributions: you have one (noise), you want another (data), and you need a *principled* way to push one toward the other. But what does "principled" mean here? Cross-entropy? KL divergence? Adversarial discrimination?

It turns out that there is an ancient, beautiful, and deeply physical answer: **optimal transport**. The question is deceptively simple — given two piles of mass, what is the cheapest way to rearrange one into the other? — but the theory that falls out of it underpins an enormous fraction of modern generative modeling, from Wasserstein-GANs to flow matching to diffusion.

This series builds the optimal transport toolkit from scratch, with one eye on the mathematics and the other on the generative models it enables. The arc is:

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
