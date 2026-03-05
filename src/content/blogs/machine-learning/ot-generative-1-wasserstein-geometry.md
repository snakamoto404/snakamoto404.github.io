---
title: "OT for generative modeling 1 — the Wasserstein geometry"
date: 2026-02-23
summary: "We construct the Wasserstein manifold from first principles: probability distributions as points, sample-space vector fields as tangent vectors, the density-weighted inner product that endows optimal transport with a rich Riemannian geometry. Physics-intuition on the Benamou-Brenier theorem which unifies static and Riemannian definitions."
---

In [Part 0](/blogs/machine-learning/ot-generative-0-static/) we defined the Wasserstein distance: the cheapest way to rearrange one distribution into another. We now know *how much* it costs to move mass. But that framing treats distributions as static objects — you compare two of them, get a number, and that was it. However, transport is an inherently dynamical process; recall our water analogy, probability distributions can continuously flow. Guiding questions for this section:

- <span class="question">How to describe the dynamical aspects of transport?</span> We need the geometric structure of a manifold: distributions as points, velocity fields as tangent vectors, and an inner product. We'll carefully disentangle the **sample domain** $\R^n$ from the **distribution manifold** $\mathcal{W}_2$.
- <span class="question">How is the static $W_2$ definition in [part 0](/blogs/machine-learning/ot-generative-0-static/) related to the continuous evolution of probability distributions?</span> The Benamou-Brenier formula shows that $W_2$ is the geodesic distance on $\mathcal{W}_2$: a nested action decomposition connecting static coupling costs to dynamical kinetic energy.

Part 1 is notably denser than part 0, but also much the more beautiful. From now on, we focus on $W_2$ with quadratic penalty; we'll see physics meet statistics: the continuity equation in action, the Fokker-Planck equation falling out as a corollary, and the free-particle Lagrangian action providing the key bridge between static and dynamical perspectives on optimal transport.

## Contents

- [The $\mathcal{W}_2$ manifold](#the-mathcalw_2-manifold)
  - [Continuity equation, tangent space, vector fields](#continuity-equation-tangent-space-vector-fields)
- [The Wasserstein metric](#the-wasserstein-metric)
  - [Wasserstein length as free-fluid action](#wasserstein-length-as-free-fluid-action)
- [Unifying static and dynamical perspectives](#unifying-static-and-dynamical-perspectives)
  - [Single-particle least action](#single-particle-least-action)
  - [Boundary conditions vs. transport plans](#boundary-conditions-vs-transport-plans)
  - [The nested decomposition](#the-nested-decomposition)
  - [From particles to fluid](#from-particles-to-fluid)

## The $\mathcal{W}_2$ manifold

Fixing a sample space $\R^n$, we consider the set $\mathcal W_2$ of all **probability distributions over $\R^n$ with finite variance**. For example, this could be a distribution over all images in $\R^{H\times W}$. $\R^n$ is endowed with the standard Euclidean topology. Two perspectives on a point $\rho \in \mathcal W_2$:

- A snapshot of water in $\R^n$ of total mass $1$, with $\rho(x)$ density at each point.
- Normal $\R^n$ vectors are functions $\{1, \dots, n\}\to \R$; think of each $\rho\in \mathcal W_2$ as an infinite-dimensional vector with one component at each $x\in \R^n$, of value $\rho(x)$. It's subject to the additional non-negativity and integrate-to-one constraints.

In this space, **a point $P\in \mathcal W_2$** is **an entire distribution $P(x)$ over $x \in \R^n$**. There are two spaces at play — the sample domain $\R^n$ where data lives, and the distribution manifold $\mathcal W_2$ where each point is itself a distribution — and it's crucial to separate them.

### Continuity equation, tangent space, vector fields

We next identify derivatives on $\mathcal W_2$. Note that $\mathcal W_2$ is a subset of the ambient space $L^2(\R^n)$ of square-integrable functions $\R^n\to \R$.

Velocities (fancily called **tangent vectors**) for general vectors are intuitive: just take the component-wise derivative! However, when we're restricting ourselves to probability distributions, we'll "slide off" the manifold if we follow general velocities, even if infinitesimally.

Let's go back to the fluid perspective. Generally, fluid density $\rho$ evolves according to the **continuity equation**
$$
\partial_t \rho + \nabla \cdot (\rho\, v) = 0
$$
This is a local conservation law which dictates that probability density (mass) **cannot evolve by teleporting in the sample space $\R^n$**: the change $\partial_t \rho$ equals the negative divergence of the mass flux $\rho\, v$ for some vector field $v: \R^n\to \R^n$ **in the sample space**.

> The continuity equation provides a **many-to-one map** from (smooth) sample-space vector fields $v: \R^n\to \R^n$ to **permissible density evolutions** on $\mathcal W_2$.

We can do better. The [Helmholtz decomposition](https://en.wikipedia.org/wiki/Helmholtz_decomposition) says that any (regular) vector field on $\R^n$ splits as $v = \nabla \varphi + w$ where $w$ is divergence-free -- this is the familiar curl-divergence decomposition in 3D. The divergence-free component swirls mass along the contours of $\rho$ without changing it — it satisfies $\nabla \cdot (\rho\, w) = 0$[^weighted] and contributes nothing to $\partial_t \rho$ in the continuity equation above. Quotienting out these invisible components:

:::definition[Tangent space of $\mathcal W_2$]
The space $T_\rho\, \mathcal W_2$ of probability density velocities on the Wasserstein manifold is one-to-one [^closure] with the space of gradient vector fields $\{v: \mathbb R^n\to \mathbb R^n\mid v = \nabla \varphi\}$ on the sample space.
:::

Intuitively, sample-space vector fields are like wind that can blow the fluid around. However, we don't care about the component of the wind that makes water go around in infinitesimal circles (these don't change water density); the remaining vector field degree of freedom can always be identified as a gradient field.

[^weighted]: More precisely, the relevant decomposition is into $\nabla \varphi$ and $w$ with $\nabla \cdot (\rho\, w) = 0$ ($\rho$-weighted divergence-free), which are orthogonal under the $\rho$-weighted $L^2$ inner product.
[^closure]: Technically, $T_\rho\, \mathcal W_2$ is the $L^2(\rho)$-closure of $\{\nabla \varphi : \varphi \in C_c^\infty(\R^n)\}$.

## The Wasserstein metric

The last section was ankle-deep in differential geometry, now let's go knee-deep by introducing a **Riemannian metric**; this makes $\mathcal W_2$ a Riemannian manifold. A Riemannian metric endows a manifold with notions of angles and length.

The metric is a bilinear form $\la \cdot, \cdot\ra_\rho: T_\rho \mathcal W_2 \times T_\rho \mathcal W_2 \to \R$ that takes in two tangent vectors and computes an inner product.

- By integrating the metric of the velocity along a curve, we obtain the **length** of a curve.
- Given two points, a **geodesic** is a minimal-length curve between them.
- The **(geodesic) distance** between two points is the length of the geodesic.

:::remark[Euclidean geometry]
The standard Euclidean metric on $\R^n$ consumes two vectors and computes $\la u, v\ra = \sum_j u_j v_j$. The length of a curve $\gamma: [0, 1]\to \R^n$ is
$$
    \mathcal L(\gamma) = \int_0^1 \|\pd t \gamma(t)\|\, dt
$$
Geodesics are straight lines $\gamma(t) = (1-t)x + ty$, and the geodesic distance is $\|x - y\|$.
:::

What's a natural metric on $\mathcal W_2$ for two tangent vectors at density $\rho$, represented by gradient vector fields $u=\nabla \varphi, v=\nabla \psi$? A natural candidate is their $\rho$-weighted inner product on the sample space:

:::definition[Wasserstein metric]
$$
    \la u, v\ra_\rho = \int_{\R^n} \rho(x)\cdot \la u(x),\, v(x)\ra\, dx = \int_{\R^n} \rho\cdot \la \nabla\varphi,\, \nabla\psi\ra\, dx
$$
:::

:::definition[Wasserstein length]
Given a curve $\gamma: [0, 1]\to \mathcal W_2$ connecting $\gamma_0=P$ to $\gamma_1=Q$, we identify $\pd t \gamma_t$ with a gradient vector field $v_t:\R^n\to \R^n$. The **Wasserstein length** of the curve is
$$
    \mathcal L(\gamma) = \int_0^1\|\pd t \gamma_t\|_{\gamma_t}^2\, dt = \int_0^1 \left[\int_{\R^n} \gamma_t(x) \|v_t(x)\|^2\, dx\right]\, dt
$$
The **Wasserstein distance** $W_2(P, Q)$ is the infimum of $\sqrt{\mathcal L(\gamma)}$ over all such curves.
:::

Let's unpack this. We have a smooth, locally continuous deformation from distribution $P$ to $Q$ given by the family of distributions $\gamma_t$, where $\gamma_t$ is generated by the flow of $v_t$ on the sample space. The Wasserstein length **of the curve $\gamma$** is the integral over time and space of the infinitesimal "action", which is just the mass $\gamma_t(x)$ multiplied by the velocity squared $\|v_t(x)\|^2$.

### Wasserstein length as free-fluid action

For a single **free particle** of mass $m$ traveling with velocity $v$, the Lagrangian is purely kinetic: $L = \frac{1}{2}m\|v\|^2$. The action over a trajectory $x(t)$ is
$$
    S_{\text{particle}} = \int_0^1 \frac{1}{2}m\,\|\dot x(t)\|^2\, dt
$$

Now promote this to a **free fluid** with density $\rho$. Each infinitesimal parcel at $x$ carries mass $\rho(x)\,dx$ and moves with velocity $v(x)$. The total action of the fluid is
$$
    S_{\text{fluid}} = \int_0^1\!\!\int_{\R^n} \frac{1}{2}\,\rho(x)\,\|v(x)\|^2\, dx\, dt
$$

Up to a factor of $1/2$, **the Wasserstein length of a curve $\gamma:[0, 1]\to \mathcal W_2$ is exactly the action of a free fluid flowing from $\gamma_0=P$ to $\gamma_1=Q$ under velocity field $v_t \equiv \dot \gamma_t: \R^n\to \R^n$.** This is the mechanical meaning of the Wasserstein metric.

:::definition[Dynamical definition of $W_2$]
The Wasserstein-2 distance between $P, Q\in \mathcal W_2$ is equivalently the minimum fluid action between configurations $P$ and $Q$:
$$
\begin{equation}
W_2^2(P, Q) = \inf_{\rho_t,\, v_t} \int_0^1\!\!\int_{\R^n} \rho_t(x)\,\|v_t(x)\|^2\, dx\, dt
\label{eq:w2-dynamical}
\end{equation}
$$

$$
\begin{align*}
\text{s.t.}\quad & \pd t \rho + \nabla \cdot (\rho\, v) = 0 \\
& \rho_0 = P,\quad \rho_1 = Q
\end{align*}
$$
:::

## Unifying static and dynamical perspectives

There's an elephant in the room: we have the [static definition of Wasserstein distance](/blogs/machine-learning/ot-generative-0-static/#eq-w2-static) and the dynamical definition $\eqref{eq:w2-dynamical}$, and they had better agree. The Kantorovich formulation optimizes over transport plans; the Riemannian formulation optimizes over fluid flows. Beautiful theories should have unique, canonical definitions — and these two are the same.

The unifying result is the **Benamou-Brenier theorem**. This theorem is pivotal because **it identifies the linear optimal transport plan that realizes the Wasserstein distance** — the engine at the heart of flow matching.

:::theorem[Benamou-Brenier]
The Kantorovich (static) and Riemannian (dynamical) definitions of $W_2$ coincide:
$$
\begin{aligned}
    W_2^2(P, Q)
    &= \inf_{\pi \in \Pi(P, Q)} \int_{\R^n\times\R^n} \|y - x\|^2\, d\pi(x, y) \\
    &= \inf_{\substack{\rho_t,\, v_t \\ \pd t\rho + \nabla\cdot(\rho v) = 0 \\ \rho_0=P,\;\rho_1=Q}} \int_0^1\!\!\int_{\R^n} \rho_t\,\|v_t\|^2\, dx\, dt
    \label{eq:benamou-brenier}
\end{aligned}
$$
The optimum of the dynamical formulation is achieved by straight-line transport under the optimal plan $\pi^*$: each mass element $(x, y)\sim\pi^*$ follows trajectory $\gamma(t) = (1-t)\,x + t\,y$ with velocity $y - x$. The optimal marginal velocity field is
$$
    v_t^*(z) = \E_{\pi^*}\!\left[y - x \;\middle|\; (1-t)\,x + t\,y = z\right]
$$
:::

We state the result first, then prove it in four steps:

1. [Single-particle least action](#single-particle-least-action): Euler-Lagrange gives straight-line trajectories with the static quadratic cost $\|y - x\|^2$.
2. [Single-particle transport plans](#boundary-conditions-vs-transport-plans): a transport plan $\pi$ assigns endpoints to each mass element; classical mechanics takes over.
3. [The nested decomposition](#the-nested-decomposition): the macroscopic particle ensemble action splits into an inner infimum over single-particle trajectories, and outer infimum over coupling plans, recovering the static definition.
4. [From particle ensemble to fluid](#from-particles-to-fluid): the marginal velocity field is the conditional expectation of particle velocities; a variance-drop argument shows particle ensemble and fluid actions coincide for $\pi^*$.

### Single-particle least action

A single free particle of unit mass travels from $x_0$ at $t=0$ to $x_1$ at $t=1$. The Lagrangian is purely kinetic: $L = \frac{1}{2}\|\dot x\|^2$. By the Euler-Lagrange equation ($\ddot x = 0$), the action-minimizing trajectory is a straight line at constant velocity:
$$
    x(t) = (1-t)\,x_0 + t\,x_1, \qquad \dot x = x_1 - x_0
$$
Plugging back, the minimum action of this single particle is exactly
$$
    S^*(x_0, x_1) = \|x_1 - x_0\|^2
$$
(dropping the conventional $1/2$). This is another perspective on **the quadratic static cost** from Part 0, i.e. the minimal action of free-particle travel between two endpoints.

### Boundary conditions vs. transport plans

Now scale up to the fluid. The **boundary conditions** are the marginal distributions: $P$ at $t=0$ and $Q$ at $t=1$. The fluid-level boundary condition is underspecified: it tells us the shape of the final distributions, but not **which densities go where**. Infinitely many arrangements are compatible.

A **transport plan** $\pi \in \Pi(P, Q)$ resolves this ambiguity. It's a coupling that assigns specific endpoints to every infinitesimal unit of mass: "$\pi(x, y)$ mass travels from $x$ to $y$." Once a plan is fixed, classical mechanics takes over — every mass element independently follows its own Euler-Lagrange straight-line path.

### The nested decomposition

Here is the core of Benamou-Brenier. Let's first consider the dynamical fluid as a macroscopic ensemble of particles, its action decomposes into two nested minimization problems:

$$
\begin{aligned}
S_{\text{fluid as particles}}^* &= \int_{\R^n\times \R^n} \left[\inf_{\substack{\gamma(t):\; \gamma(0)=x \\ \gamma(1)=y}} \int_0^1 \|\dot \gamma(t)\|^2\, dt\;\right] d\pi(x, y) \\
&= \inf_{\pi \in \Pi(P, Q)}\; \int \|x-y\|^2 d\pi(x, y)
= W_2^2(P, Q)
\label{eq:nested-action}
\end{aligned}
$$

1. **Inner infimum (classical mechanics):** Fix endpoints $(x, y)$ from the plan. The action-minimizing trajectory is a straight line; the resulting cost is $\|y-x\|^2$.
2. **Outer infimum (static OT):** Substitute the inner solution. What remains is $\inf_\pi \int \|x_0 - x_1\|^2\, d\pi$ — precisely the [static Kantorovich definition in Part 0](/blogs/machine-learning/ot-generative-0-static/#eq-w2-static).

There're three beautifully interleaved viewpoints at work here:

- **Static OT** is the outer infimum alone. It asks: given $P$ and $Q$, what plan minimizes aggregate pairwise cost? Time and dynamics are absent.
- **Classical mechanics** is the inner infimum alone. It asks: fixing endpoints and mass, what path minimizes action?
- **Dynamic OT** is both simultaneously. Minimizing the global fluid action discovers the optimal particle trajectories *and* the optimal transport plan that binds them.

### From particles to fluid

There remains a subtle yet important gap: we decomposed the action into individual particle costs under a plan $\pi$, but the dynamical definition $\eqref{eq:w2-dynamical}$ is written in terms of a macroscopic velocity field $v_t$, not individual particle trajectories. How do we reconcile the particle and fluid perspectives?

:::remark

This reconciliation is the key engine behind being able to optimize the desired marginal flow matching objective using the tractable conditional flow matching objective.
:::

Given a transport plan $\pi$, each mass element follows its straight-line trajectory $x(t) = (1-t)x_0 + tx_1$ with velocity $\dot x = x_1 - x_0$. Multiple particles may pass through the same point $x$ at time $t$, potentially with different velocities. The momentum density and mass density at $x$ at time $t$ is:

$$
    \mu_t(x) = \int 1_{ty + (1-t) z = x} (z-y)\, d\pi(y, z), \quad \rho_t(x) = \int 1_{ty + (1-t)z = x}\, d\pi(y, z)
$$
Divide the two to get the macroscopic fluid velocity, which turns out to be the **conditional expectation** of particle velocities given position:
$$
    v_t(x) = \E[\dot x \mid x(t) = x]
$$

By the law of total variance (bias-variance tradeoff on other contexts), the total particle action decomposes as:
$$
    \underbrace{\E_\pi \|\dot x\|^2}_{\text{particle action}} \;=\; \underbrace{\int \rho_t \|v_t\|^2\, dx}_{\text{fluid action}} \;+\; \underbrace{\int \rho_t\,\mrm{Var}[\dot x \mid x(t) = x]\, dx}_{\geq\, 0}
$$

**Fluid action is upper-bounded by particle ensemble action**. Equality holds when the variance vanishes — i.e., when **no two particles cross at the same point at the same time with different velocities**. Under the optimal transport plan $\pi^*$, this is guaranteed by cyclical monotonicity: if two particles' trajectories crossed, swapping their destinations would reduce total cost [^cyclicality]. Therefore, for the optimal plan, particle and fluid actions coincide exactly, completing the bridge.

[^cyclicality]: Minimal-cost implies that for any two support pairs $(a_1, b_1), (a_2, b_2)$, $\|b_1-a_1\|^2+\|b_2-a_2\|^2 \le \|b_2-a_1\|^2+\|b_1-a_2\|^2$, equivalently $\langle a_1-a_2,\; b_1-b_2\rangle \ge 0$. You can draw some diagrams to convince yourself that this implies no crossing.
