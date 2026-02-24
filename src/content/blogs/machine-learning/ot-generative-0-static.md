---
title: "OT for generative modeling 0 — the static perspective"
date: 2026-02-23
summary: "Why we care about optimal transport (OT), and introduction to the static viewpoint. Variational characterization, and WGAN."
---

At the heart of modern generative modeling, there is a fundamental tension between two spaces. On **sample space** $\mathbb R^n$, Euclidean geometry gives us clean, benign objectives such as $L^2$ mean squared error. On the **space of distributions** $\mathcal P(\mathbb R^n)$ maximum likelihood singles out the Kullback-Leibler divergence as the principled objective, and $\mathcal P(\mathbb R^n)$ has its own geometry; however, KL is remarkably agnostic towards the internal geometry of $\mathbb R^n$ and, in particular, the inductive biases associated with it.

**Optimal transport** provides one of the most mathematically beautiful bridges between the two spaces. It has also become the backbone of modern generative modeling. My personal acquaintance with the field began with [Wasserstein GAN](https://arxiv.org/pdf/1701.07875) (more on it later). In school, [information theory](https://nlyu1.github.io/classical-info-theory/) provided the necessary tools, and I've always been intrigued by the connections between e.g. flow matching, wasserstein gradient flow, and even [renormalization](https://arxiv.org/abs/2202.11737). Motivated by understanding a highly impressive recent work on [drifting models](https://arxiv.org/abs/2602.04770), I decided to write these posts to learn, and unpack, some of the concepts.

This series builds the optimal transport toolkit from scratch, with one eye on the mathematics and the other on the generative models it enables. The posts aspire to be curiosity-driven and hand-waves some epsilons and deltas. To start off, two themes run through everything, we've already seen one:

1. **Bridging sample and distribution geometries.** Optimal transport connects Euclidean geometry to statistical divergences. This empowers us to understand training processes with benign $L^2$  objectives through the lens of proper distributional optimization.
2. **Bridging static and dynamic definitions.** The equivalence between static, closed-form quantities (e.g. transport cost) and dynamic quantities (e.g. integral of kinetic energy) hinges on a clean decomposition: solve the single-particle action-extremizing path, then marginalize over a fixed distribution. This is the engine under Benamou-Brenier and the key to making flow matching computable using conditional flow matching.

Layout of the posts:

- **Part 0** (this post): the *static* picture — transport plans, the static definition of Wasserstein distance, and the variational dual that empowers Wasserstein-GAN.
- **[Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/)**: the *geometric* picture — the Wasserstein distribution manifold, the *dynamic* definition of $W_2$ distance, gradients, geodesics, and flow matching.
- **[Part 2](/blogs/machine-learning/ot-generative-2-wasserstein-gradients/)**: the *unifying* picture — Benamou-Brenier theorem as the bridge between static and dynamic transport. Conditional and marginal flow matching.
- **[Part 3](/blogs/machine-learning/ot-generative-3-drifting-models/)**: *application* of the optimal transport perspective to drifting models.

## Contents

- [Static definition of Wasserstein distance](#static-definition-of-wasserstein-distance)
- [Variational characterization and Wasserstein-GAN](#variational-characterization-and-wasserstein-gan)

## Static definition of Wasserstein distance

We work on $\mathbb R^d$. Interpret a probability distribution $P$ on $\mathbb R^d$ as a snapshot of a fluid[^earth] on $\mathbb R^d$, with $P(x)$ denoting the height of water at $x\in \mathbb R^d$; water has uniform density, so the **mass** of water in each region is proportional to the enclosed distribution probability. There, **mass $\leftrightarrow$ probability**.

We trivially associate the **cost** of moving mass $\rho$ from $x$ to $y$ with $\rho (x-y)^2$. You're welcome to pick other powers or cost functions. In plain words:

:::definition
The **Wasserstein distance** between distributions $P, Q$ is the minimum cost it takes to transport $P$ to $Q$ according to the quadratic cost model above.
:::

Great! Nice definition, let's go about rigorously defining it.

[^earth]: people like to use "dirt" instead of "water," for the mental model, as suggested by the alternative naming of Wasserstein distance as earth-moving distance. But we'll be talking momentum and physics later, so water seems to make much more sense.

What kind of mathematical object conveniently describes a **transport plan** from $P$ to $Q$? Kantorovich thought about this and came up with the set $\Pi(P, Q)$:

:::definition
The **coupling set** $\Pi(P, Q)$ is the set of all joint distributions on $\R^d \times \R^d$ whose marginals are $P$ and $Q$:

$$
\Pi(P, Q) = \bigl\{\, \pi \in \mathcal{P}(\R^d \times \R^d) \;\big|\; \pi_1 = P,\; \pi_2 = Q \,\bigr\}.
$$

Here $\pi_1, \pi_2$ denote the first and second marginals of $\pi$: for every measurable $A \subseteq \R^d$,

$$
\pi_1(A) = \pi(A \times \R^d), \qquad \pi_2(A) = \pi(\R^d \times A).
$$

Such a $\pi$ is called a **coupling** (or **transport plan**) of $P$ and $Q$.
:::

A joint distribution $\pi \in \Pi(P, Q)$ denotes that **we will transport $\pi(x, y)$ mass from $x$ to $y$**. We have defined point-to-point cost. The static definition of Wasserstein distance is consequently manifest:

:::definition
The **Wasserstein-2 distance** between $P$ and $Q$ is

$$
\begin{equation}
W_2^2(P, Q) \;=\; \inf_{\pi \in \Pi(P, Q)} \int_{\R^d \times \R^d} \|x - y\|^2 \, d\pi(x, y).
\label{eq:w2-static}
\end{equation}
$$
:::

We're taking an infimum over transport plans $\pi$, of the transport cost associated with $\pi$. Thus the name optimal transport.

:::remark
We call the definition *static* because the transport plan $\pi$ only tells you *how much* to transfer from here to there, not what trajectory the transported mass travels through. The dynamic version will be introduced in the next post.
:::

:::remark
Also recognize the Wasserstein distance as a linear program: the constraints and cost are both linear in $\pi$. It's screaming duality, which we'll explore next.
:::

:::remark
On finite-dimensional spaces, $\Pi(P, Q)$ is represented by doubly-stochastic matrices. Look up Sinkhorn-Knopp and the Birkhoff polytope (convex hull of permutation matrices).
:::

## Variational characterization and Wasserstein-GAN

When we say **variational** characterization [^variational] of something, we generally mean writing "something" as a minimum or maximum. Such characterizations are extremely useful because (1) theoretically, they provide bounds, (2) SGD loves extremizing things, and (3) such characterizations provide valuable insight into what the quantity is doing. Recall

:::definition
A function $f : \R^d \to \R$ is **$1$-Lipschitz** if for all $x, y \in \R^d$, $|f(x) - f(y)| \le \|x - y\|$.
We write $\mrm{Lip}_1$ for the set of all such functions.
:::

The key insight is that $W_1$ — the Wasserstein-1 distance (replace $\|x-y\|^2$ with $\|x-y\|$ in \eqref{eq:w2-static}) — admits a clean dual formulation.

:::theorem[Kantorovich-Rubinstein duality]
$$
W_1(P, Q) = \sup_{f \in \mrm{Lip}_1} \left[ \E_{x \sim P}\, f(x) - \E_{y \sim Q}\, f(y) \right].
$$
:::

In words: instead of searching over couplings (the primal / static formulation), we search over $1$-Lipschitz "critic" functions. The supremum is attained by the function that best separates $P$ from $Q$ — subject to the smoothness constraint.

:::remark
Why does this matter for generative modeling? Consider training a generator $G_\theta$ that pushes a noise distribution $Z$ toward data $P_{\mrm{data}}$. Let $Q_\theta = G_{\theta\#} Z$ be the generated distribution. Then

$$
W_1(P_{\mrm{data}}, Q_\theta) = \sup_{f \in \mrm{Lip}_1} \left[ \E_{x \sim P_{\mrm{data}}}\, f(x) - \E_{y \sim Q_\theta}\, f(y) \right].
$$

This is exactly the [**Wasserstein-GAN**](https://arxiv.org/pdf/1701.07875) (WGAN) objective: train a neural-network critic $f_w$ to approximate the supremum, and train $G_\theta$ to minimize the resulting distance. The Lipschitz constraint is enforced by weight clipping or gradient penalty. Compared to the original GAN's Jensen-Shannon divergence, $W_1$ provides gradients even when the supports of $P_{\mrm{data}}$ and $Q_\theta$ don't overlap — which is precisely the mode-collapse pathology that plagued early GANs.
:::

[remark: connection to TV] (don't modify yet)

[^variational]: Variational characterizations of mutual information and $f$-divergences are some of the most beautiful results in information theory.