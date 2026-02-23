---
title: "OT for generative modeling 2 — Wasserstein gradients"
date: 2026-02-23
summary: "Part 2 of a series on optimal transport in generative models. We connect Lagrangian particle mechanics to Eulerian fluid dynamics via Benamou-Brenier, unifying the static and dynamic perspectives, and develop Otto calculus for gradient descent on the space of distributions."
---

We now have a manifold ([Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/)) and a distance ([Part 0](/blogs/machine-learning/ot-generative-0-static/)). The natural next question is: *how do we optimize on it?*

In finite dimensions, gradient descent is straightforward — compute $\nabla f$, step in the negative direction. But our "point" is an entire probability distribution, our "tangent vector" is a vector field on $\mathbb{R}^d$, and our "function" is a functional like KL divergence. What does "steepest descent" even look like in this setting?

The answer turns out to hinge on a beautiful duality between two perspectives on moving mass:

- The **Lagrangian** (particle) view: track individual grains of sand from origin to destination, each following its own least-action trajectory.
- The **Eulerian** (fluid) view: stand at a fixed point in space and observe the aggregate velocity field of all the passing mass.

The Benamou-Brenier theorem tells us these are *the same thing* — the static Kantorovich cost equals the dynamic fluid kinetic energy — and the bridge between them is precisely the Euler-Lagrange equation from classical mechanics. Once we have that bridge, Otto calculus gives us concrete rules for computing Wasserstein gradients: take the first variation of your functional, apply the spatial gradient, and you have a velocity field that pushes your distribution downhill.

This is the post where the physics and the probability finally fuse. It is also the post that makes [Part 3](/blogs/machine-learning/ot-generative-3-drifting-models/) (drifting models as Wasserstein gradient flow) possible.

## Contents

- [Action and Lagrangian](#action-and-lagrangian)
- [Unified dynamic and static perspectives — Benamou-Brenier](#unified-dynamic-and-static-perspectives--benamou-brenier)
- [Otto calculus](#otto-calculus)

## Action and Lagrangian

*Coming soon.*

## Unified dynamic and static perspectives — Benamou-Brenier

*Coming soon.*

## Otto calculus

*Coming soon.*
