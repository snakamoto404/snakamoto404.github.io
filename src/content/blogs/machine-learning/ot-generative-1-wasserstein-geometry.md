---
title: "OT for generative modeling 1 — the Wasserstein geometry"
date: 2026-02-23
summary: "Part 1 of a series on optimal transport in generative models. We construct the Wasserstein manifold from first principles: probability distributions as points, vector fields as tangent vectors, and the density-weighted inner product that turns optimal transport into Riemannian geometry."
---

In [Part 0](/blogs/machine-learning/ot-generative-0-static/) we defined the Wasserstein distance: the cheapest way to rearrange one distribution into another. We now know *how much* it costs to move mass. But that framing treats distributions as static objects — you compare two of them, get a number, and that was it. However, transport is an inherently dynamical process; recall our water analogy, probability distributions can continuously flow. Guiding questions for this section:

- <span class="question">How to describe the dynamical aspects of transport?</span> We need the geometric structure of a manifold: distributions as points, velocity fields as tangent vectors, and an inner product. We'll carefully disentangle the **sample domain** $\R^d$ from the **distribution manifold** $\mathcal{W}_2$.
- <span class="question">What is this term "Wasserstein Gradient Flow"?</span> Otto calculus lets us compute gradients of functionals over distributions; these gradients manifest as vector fields on $\R^d$. The Fokker-Planck equation falls out as a corollary.
- <span class="question">How is the static $W_2$ definition in [Part 0](/blogs/machine-learning/ot-generative-0-static/) related to the continuous evolution of probability distributions?</span> The Benamou-Brenier formula shows that $W_2$ is the geodesic distance on $\mathcal{W}_2$: a nested action decomposition connecting static coupling costs to dynamical kinetic energy.

Part 1 is notably denser than part 0, but also much the more beautiful. From now on, we focus on $W_2$ with quadratic penalty; we'll see physics meet statistics: the continuity equation in action, the Fokker-Planck equation falling out as a corollary, and the free-particle Lagrangian action providing the key bridge between static and dynamical perspectives on optimal transport.

## Contents

- [The $\mathcal{W}_2$ manifold](#the-mathcalw_2-manifold): which **spaces** are we working in?
- [The Wasserstein metric](#the-wasserstein-metric): dynamic definition of distance

## The $\mathcal{W}_2$ manifold

Fixing a sample space $\R^d$, we consider the set $\mathcal W_2$ of all **probability distributions over $\R^d$ with finite variance**. For example, this could be a distribution over all images in $\R^{H\times W}$. $\R^d$ is endowed with the standard Euclidean topology. Two perspectives on a point $\rho \in \mathcal W_2$:

- A snapshot of water in $\R^d$ of total mass $1$, with $\rho(x)$ density at each point.
- Normal $\R^d$ vectors are functions $\{1, \dots, d\}\to \R$; think of each $\rho\in \mathcal W_2$ as an infinite-dimensional vector with one component at each $x\in \R^d$, of value $\rho(x)$. It's subject to the additional non-negativity and integrate-to-one constraints.

In this space, **a point $P\in \mathcal W_2$** is **an entire distribution $P(x)$ over $x \in \R^d$**. There are two spaces at play — the sample domain $\R^d$ where data lives, and the distribution manifold $\mathcal W_2$ where each point is itself a distribution — and it's crucial to separate them.

### Continuity equation, tangent space, vector fields

We next identify derivatives on $\mathcal W_2$. Note that $\mathcal W_2$ is a subset of the ambient space $L^2(\R^d)$ of square-integrable functions $\R^d\to \R$.

Velocities (fancily called **tangent vectors**) for general vectors are intuitive: just take the component-wise derivative! However, when we're restricting ourselves to probability distributions, we'll "slide off" the manifold if we follow general velocities, even if infinitesimally.

Let's go back to the fluid perspective. Generally, fluid density $\rho$ evolves according to the **continuity equation**
$$
\begin{equation}
\partial_t \rho + \nabla \cdot (\rho\, v) = 0
\label{eq:continuity}
\end{equation}
$$
This is a local conservation law which dictates that probability density (mass) **cannot evolve by teleporting in the sample space $\R^d$**: the change $\partial_t \rho$ equals the negative divergence of the mass flux $\rho\, v$ for some vector field $v: \R^d\to \R^d$ **in the sample space**.

> The continuity equation provides a **many-to-one map** from (smooth) sample-space vector fields $v: \R^d\to \R^d$ to **permissible density evolutions** on $\mathcal W_2$.

We can do better. The [Helmholtz decomposition](https://en.wikipedia.org/wiki/Helmholtz_decomposition) says that any (regular) vector field on $\R^d$ splits as $v = \nabla \varphi + w$ where $w$ is divergence-free -- this is the familiar curl-divergence decomposition in 3D. The divergence-free component swirls mass along the contours of $\rho$ without changing it — it satisfies $\nabla \cdot (\rho\, w) = 0$[^weighted] and contributes nothing to $\partial_t \rho$ in $\eqref{eq:continuity}$. Quotienting out these invisible components:

:::definition[Tangent space of $\mathcal W_2$]
The space $T_\rho\, \mathcal W_2$ of probability density velocities on the Wasserstein manifold is one-to-one [^closure] with the space of gradient vector fields $\{v: \mathbb R^n\to \mathbb R^n\mid v = \nabla \varphi\}$ on the sample space.
:::

Intuitively, sample-space vector fields are like wind that can blow the fluid around. However, we don't care about the component of the wind that makes water go around in infinitesimal circles (these don't change water density); the remaining vector field degree of freedom can always be identified as a gradient field.

[^weighted]: More precisely, the relevant decomposition is into $\nabla \varphi$ and $w$ with $\nabla \cdot (\rho\, w) = 0$ ($\rho$-weighted divergence-free), which are orthogonal under the $\rho$-weighted $L^2$ inner product.
[^closure]: Technically, $T_\rho\, \mathcal W_2$ is the $L^2(\rho)$-closure of $\{\nabla \varphi : \varphi \in C_c^\infty(\R^d)\}$.

## The Wasserstein metric

*Coming soon.*
