---
title: "OT for generative modeling 1 — the Wasserstein geometry"
date: 2026-02-23
summary: "Part 1 of a series on optimal transport in generative models. We construct the Wasserstein manifold from first principles: probability distributions as points, vector fields as tangent vectors, and the density-weighted inner product that turns optimal transport into Riemannian geometry."
---

In [Part 0](/blogs/machine-learning/ot-generative-0-static/) we defined the Wasserstein distance: the cheapest way to rearrange one distribution into another. We now know *how much* it costs to move mass. But that framing treats distributions as static objects — you compare two of them, get a number, and that's it. From now on, we focus on $W_2$ with quadratic penalty.

However, transport is an inherently dynamical process; recall our water analogy, probability distributions can continuously flow. Guiding questions for this section:

- How to describe the dynamical aspects of transport? <blue We need the geometric structure of tangent space (velocity vectors), inner products, geodesics -- in short, a "manifold". We'll disentangle the common confusion between **sample domain** $\mathbb R^n$ and the distribution manifold ...>
- What is this term "Wasserstein Gradient Flow"? <blue Otto calculusj -- we'll see how to compute Wasserstein gradients, which manifest as vector fields on the sample domain $\mathbb R^n$. >
- How is the static $W_2$ definition in [Part 0](/blogs/machine-learning/ot-generative-0-static/) related to the continuous evolution of probability distributions? <... Benamou-Brenier formula. We'll see how the $W_2$ distance is just the geodesic distance on ... (manifold)>

Part 1 is notably denser than part 0, but also much the more beautiful. Here, we'll see physics meet statistics: the continuity equation in action, the Fokker-Planck equation falling out as a corollary, and the free-particle Lagrangian action providing the the key bridge between static and dynamical perspectives on optimal transport.

## Contents

- [The $\mathcal{W}_2$ manifold](#the-mathcalw_2-manifold): which **spaces** are we working in?
- [Continuity equation and the tangent space](#continuity-equation-and-vector-fields): what describes flow
- [The Wasserstein metric](#the-wasserstein-metric): dynamic definition of distance

## The $\mathcal{W}_2$ manifold

*Coming soon.*

## Continuity equation and vector fields

*Coming soon.*

## The Wasserstein metric

*Coming soon.*
