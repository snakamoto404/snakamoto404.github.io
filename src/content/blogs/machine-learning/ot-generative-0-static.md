---
title: "OT for generative modeling 0 — the static perspective"
date: 2026-02-23
summary: "Why we care about optimal transport (OT), the static (Kantorovich) definition of Wasserstein definition, the linear programming (Kantorovich-Rubinstein) dual formulation, and WGAN."
---

There are two spaces of mathematical objects at the heart of generative modeling. On the **sample space** $\R^d$, Euclidean geometry gives us clean, benign objectives such as $L^2$ mean squared error. On the **space of distributions** $\mathcal P(\R^d)$ maximum likelihood singles out the Kullback-Leibler divergence as the principled objective, and $\mathcal P(\R^d)$ has its own geometry. However, KL is agnostic towards the internal geometry of $\R^d$ and, consequently, the inductive biases associated with it; $D(P\|Q)$ also blows up when $P$'s support is outside of $Q$.

**Optimal transport** provides one of the most mathematically beautiful bridges between the two spaces. It has also become the backbone of modern generative modeling. My personal acquaintance with the field began with [Wasserstein GAN](https://arxiv.org/pdf/1701.07875). In school, [information theory](https://nlyu1.github.io/classical-info-theory/) provided the necessary tools, and I've always been intrigued by the connections between e.g. flow matching, wasserstein gradient flow, and even [renormalization](https://arxiv.org/abs/2202.11737). Motivated by understanding a highly impressive recent work on [drifting models](https://arxiv.org/abs/2602.04770), I decided to write these posts to learn, and unpack, some of the concepts.

This series **builds the optimal transport toolkit from scratch**, with one eye on the mathematics and the other on the generative models it enables. The posts aspire to be curiosity-driven and hand-waves some epsilons and deltas. To start off, two themes run through everything, we've already seen one:

1. **Bridging sample and distribution geometries.** Optimal transport connects Euclidean geometry to statistical divergences. This empowers us to understand training processes with benign $L^2$  objectives through the lens of proper distributional optimization.
2. **Bridging static and dynamic definitions.** The equivalence between static, closed-form quantities (e.g. transport cost) and dynamic quantities (e.g. integral of kinetic energy) hinges on a clean decomposition: solve the single-particle action-extremizing path, then marginalize over a fixed distribution. This is the engine under Benamou-Brenier and the key to making flow matching computable using conditional flow matching.

Layout of the posts:

- **Part 0** (this post): the *static* picture — static definition of Wasserstein distance, and the variational dual that empowers Wasserstein-GAN.
- **[Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/)**: the *geometric* picture — the Wasserstein distribution manifold, the *dynamic* definition of $W_2$ distance, and unification with the static picture.
- **[Part 2](/blogs/machine-learning/ot-generative-2-drifting-models/)**: *application* of the optimal transport perspective to drifting models.

## Contents

- [Wasserstein distance](#wasserstein-distance)
- [Variational characterization and W-GAN](#variational-characterization-and-w-gan)

## Wasserstein distance

We work on $\mathbb R^d$. Interpret a probability distribution $P$ on $\mathbb R^d$ as a snapshot of a fluid[^earth] on $\mathbb R^d$, with $P(x)$ denoting the height of water at $x\in \mathbb R^d$; water has uniform density, so the **mass** of water in each region is proportional to the enclosed distribution probability. There, **mass $\leftrightarrow$ probability**.

We trivially associate the **cost** of moving mass $\rho$ from $x$ to $y$ with $\rho \cdot \|x-y\|^2$. You're welcome to pick other powers or cost functions ($\rho \cdot \|x-y\|$ is another popular one). In plain words:

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

<span id="eq-w2-static"></span>

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

## Variational characterization and W-GAN

When we say **variational** characterization [^variational] of something, we generally mean writing "something" as a minimum or maximum. Such characterizations are extremely useful because (1) theoretically, they provide bounds, (2) SGD loves extremizing things, and (3) such characterizations provide valuable insight into what the quantity is doing.

Let's rewrite the Wasserstein distance $\eqref{eq:w2-static}$, generalized to $c(x, y)$ from $\|x-y\|$, by replacing the hard constraint $\pi \in \Pi(P, Q)$ with nested optimization:

Conceptualize this as an adversarial equilibrium between two players

- the **primal** player controls $\pi$ and wants to minimize $W$ as in $\eqref{eq:w2-static}$ subject to a constraint.
- Instead of a hard constraint, we equivalently enforce the constraint by introducing an adversarial **dual** who controls $\varphi, \psi$.
- The dual player can crank up any constraint deviation at $\infty$ cost to the primal player. This theme will appear very shortly, **$\infty$ penalty is used consistently in nested extremization to enforce constraints**.
- Note: in nested extremization, since the inner optimization gets to react to the outer-choice as a constant, so chronologically, the **outer operator moves first**.

As a first pass, we give the dual player the power to choose arbitrary $\varphi, \psi$:
$$
\begin{equation}
\begin{split}
W_c(P, Q) = \inf_{\pi\geq 0} \sup_{\varphi, \psi}\biggl[
    &\int c(x, y) \, d\pi(x, y) + \int \varphi(x) \left[ dP(x) - d\pi(x, y)\right] \\
    &+ \int \psi(y) \left[ dQ(y) - d\pi(x, y)\right]
\biggr]
\end{split}
\label{eq:lagrangian}
\end{equation}
$$

Let's re-interpret this game by swapping $\inf$ and $\sup$ (regularity conditions needed), regroup,
and specialize to $c(x, y) = \|x-y\|$ in the $W_1$ distance:
$$
\begin{align}
W_1(P, Q) &= \sup_{\varphi, \psi} \left[
    \int \varphi\, dP + \int \psi\, dQ + \mathcal L(\varphi, \psi)
\right] \label{eq:w1-dual-raw} \\
\mathcal L(\varphi, \psi) &= \inf_{\pi\geq 0} \int \left[\|x-y\| - \varphi(x) - \psi(y)\right]\, d\pi(x, y) \label{eq:penalty}
\end{align}
$$

In this interpretation of the game (minimax guarantees equivalent value), the dual player moves first, and the primal player reacts to minimize their penalty. Note that the value of the $\inf\sup$ and the $\sup\inf$ are the same — the optimizing parameters have changed.

Looking at $\mathcal L$ in $\eqref{eq:penalty}$: if the dual player chose $\varphi, \psi$ such that $\varphi(x) + \psi(y) > \|x-y\|$ anywhere, then the integrand is negative there and the primal player will happily put infinite mass on it, sending $\mathcal L \to -\infty$. Mathematically, **the inner infimum evaluates to an indicator function of the constraint**[^dual-cone]

$$
\mathcal L(\varphi, \psi) = \begin{cases} 0 & \text{if } \varphi(x) + \psi(y) \leq \|x-y\| \text{ for all } x,y \\ -\infty & \text{otherwise} \end{cases}
$$

[^dual-cone]: In convex analysis, $\inf_{\pi \geq 0} \int f\, d\pi$ equals $0$ if $f \geq 0$ everywhere and $-\infty$ otherwise. This is the support function of the nonnegative-measure cone, equivalently the (negative) indicator of its dual cone $\{f : f \geq 0\}$.

Because the dual player is trying to maximize the overall objective, they'll avoid any choices that trigger $\infty$ penalty. Therefore, we can safely restrict the domain of the supremum to the subset of functions where the penalty is zero:

$$
    W_1(P, Q) = \sup_{\varphi(x) + \psi(y) \leq \|x-y\|} \left[\int \varphi\, dP + \int \psi\, dQ\right]
$$

This collapses to a single degree of freedom. Setting $x = y$ in the constraint gives $\varphi(x) + \psi(x) \leq 0$, and since the dual player wants to maximize $\int \varphi\, dP + \int \psi\, dQ$, at optimality $\psi = -\varphi$. Substituting back: the constraint $\varphi(x) - \varphi(y) \leq \|x-y\|$ for all $x, y$ is exactly the 1-Lipschitz condition (by symmetry in $x, y$).

:::definition
A function $f : \R^d \to \R$ is **$1$-Lipschitz** if for all $x, y \in \R^d$, $|f(x) - f(y)| \le \|x - y\|$.
We write $\mrm{Lip}_1$ for the set of all such functions.
:::

Rearranging, we obtain the **Kantorovich-Rubinstein dual** of $W_1$ (similar result exists for $W_2$):
$$
\begin{equation}
W_1(P, Q) = \sup_{\varphi\in \mrm{Lip}_1} \left[\int \varphi\, dP - \int \varphi\, dQ\right] = \sup_{\varphi\in \mrm{Lip}_1} \left[\E_P\, \varphi - \E_Q \, \varphi\right]
\label{eq:kr-dual}
\end{equation}
$$

:::remark
Why does this matter for generative modeling? Consider training a generator $G_\theta$ that pushes a noise distribution $p_{\mrm{noise}}$ toward data $p_{\mrm{data}}$. Let $p_\theta = G_{\theta\#} p_{\mrm{noise}}$ be the generated distribution. Then

$$
W_1(p_{\mrm{data}}, p_\theta) = \sup_{f \in \mrm{Lip}_1} \left[ \E_{x \sim p_{\mrm{data}}}\, f(x) - \E_{y \sim p_\theta}\, f(y) \right].
$$

This is exactly the [**Wasserstein-GAN**](https://arxiv.org/pdf/1701.07875) (WGAN) objective: train a neural-network critic $f_w$ to approximate the supremum, and train $G_\theta$ to minimize the resulting distance. The Lipschitz constraint is enforced by weight clipping or gradient penalty. Compared to the original GAN's Jensen-Shannon divergence, $W_1$ provides gradients even when the supports of $p_{\mrm{data}}$ and $p_\theta$ don't overlap — which is precisely the mode-collapse pathology that plagued early GANs.

[^variational]: Variational characterizations of mutual information and $f$-divergences are some of the most beautiful results in information theory.
