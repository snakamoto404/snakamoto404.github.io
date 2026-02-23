---
title: "OT for generative modeling 1 — the Wasserstein geometry"
date: 2026-02-23
summary: "Part 1 of a series on optimal transport in generative models. We construct the Wasserstein manifold from first principles: probability distributions as points, vector fields as tangent vectors, and the density-weighted inner product that turns optimal transport into Riemannian geometry."
---

In [Part 0](/blogs/machine-learning/ot-generative-0-static/) we defined the Wasserstein distance: the cheapest way to rearrange one distribution into another. We now know *how much* it costs to move mass. But that framing treats distributions as static objects — you compare two of them, get a number, and that's it.

What if distributions could *move*? Not just "compare $p$ to $q$," but "watch $\rho_t$ evolve continuously from $p$ toward $q$"? To make that precise, we need the space of distributions itself to have geometric structure — tangent vectors, inner products, geodesics. In other words: we need a *manifold*.

This is exactly what Felix Otto did in his celebrated 2001 paper, and it is one of the most elegant constructions in modern applied mathematics. The punchline is almost too clean: the space $\mathcal{W}_2(\mathbb{R}^d)$ of probability distributions with finite second moments is an infinite-dimensional Riemannian manifold, and the Wasserstein-2 distance we defined in Part 0 is its geodesic distance.

This post builds that manifold from first principles — each definition motivated by the question "what structure do we *need* to talk about distributions evolving in time?" If you are comfortable with the Kantorovich problem from Part 0 and have some familiarity with differential geometry (tangent spaces, metrics, geodesics), this should feel like a natural next step.

## Contents

- [The $\mathcal{W}_2$ manifold](#the-mathcalw_2-manifold)
- [Continuity equation and vector fields](#continuity-equation-and-vector-fields)
- [The Wasserstein metric](#the-wasserstein-metric)

## The $\mathcal{W}_2$ manifold

*Coming soon.*

## Continuity equation and vector fields

*Coming soon.*

## The Wasserstein metric

*Coming soon.*
