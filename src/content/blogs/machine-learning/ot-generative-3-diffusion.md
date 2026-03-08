---
title: "OT for generative modeling 3 — Diffusion as Maximum Likelihood Estimation"
date: 2026-03-01
summary: "We derive the interpretation of physical diffusion as Wasserstein gradient flow, noise-spectrum decomposition of KL-divergence, diffusion models as MLE, and first-principles analysis of flow matching scalability. Honorable mentions to Fokker-Planck equation, Anderson's theorem, de Bruijin identity, and Tweedie's formula."
---

This part uses the Wasserstein toolkit we've developed in parts 0 and 1 (links) to unpack the dominant generative paradigm from first principles.
We begin with some physics on Brownian motion and stochastic processes, highlights include:

- **Brownian motion $\leftrightarrow$ score flow** provides the fundamental bridge between SDE and ODE formulations as well as a valuable perspective on the score-ODE which shows up everywhere in generative modeling.
  - As a corollary, we prove **Anderson's theorem** which allows one to run a SDE backwards in time.
- Unifying microscopic particle movement with macroscopic, thermodynamic optimization: diffusion with drift (Fokker-Planck) as Wasserstein gradient descent.

We also deep-dive into the two absolutely foundational pillars of scalable diffusion / flow matching. In my opinion, they are the first-principles reason why flow matching dominate modern generative modeling:

- **Tweedie's formula** provides a dimension-scalable solution to the density estimation problem, the universal problem to all generative modeling. It turns density estimation -- which nonparametrically scales with dimension -- into function estimation, which scales with the amount and internal structure of the data.
- The **dynamic de Bruijin identity** provides a canonical noise-spectral decomposition of KL-divergence. It bridges score matching (exactly what Tweedie's provide), with MSE; it also provides a canonical spectrum over which to commit bias-variance tradeoffs.

Later todo for myself: look into diffusion as optimal Bayes engine (Polyanskiy).

## Contents

- [Contents](#contents)
- [Physics of diffusion](#physics-of-diffusion)
  - [Brownian motion as score flow](#brownian-motion-as-score-flow)
  - [Brownian motion with drift, Fokker-Planck](#brownian-motion-with-drift-fokker-planck)
  - [Diffusion as Wasserstein gradient flow](#diffusion-as-wasserstein-gradient-flow)
  - [Bonus: Anderson's theorem](#bonus-andersons-theorem)
- [MLE interpretation of diffusion models](#mle-interpretation-of-diffusion-models)
  - [the dynamic de Bruijn identity](#the-dynamic-de-bruijn-identity)
  - [Application to various processes.](#application-to-various-processes)
    - [Heat process](#heat-process)
    - [Variance-preserving process](#variance-preserving-process)
- [Tweedie's formula and flow matching](#tweedies-formula-and-flow-matching)
  - [Tweedie's formula](#tweedies-formula)
  - [The flow matching process](#the-flow-matching-process)
  - [Flow matching in practice](#flow-matching-in-practice)

<!-- A clarifying point: diffusion is a heavily loaded word, it could mean any of the following:

1. Brownian motion of particles $dX_t = g\, dW_t$.
2. Brownian motion of particles with drift.
3. The diffusion equation (Fokker-Planck) [eqref]
4. Fokker-planck based SDE generative models.
5. The wide class of flow matching ODE and diffusion SDE generative models. -->

## Physics of diffusion

We begin by delving into some physics. We first adopt a **microscopic, particle-level** description of the diffusive process, then unify it with a **macroscopic, information-theory level** description:
- Microscopic language: brownian motion, vector fields, and score.
- Macroscopic language: KL minimization, entropy.

The keystone unifying these two perspectives is the Wasserstein gradient flow.

### Brownian motion as score flow

:::remark[Brownian motion]
Formally, a standard Brownian motion $W_t$ is a continuous-time stochastic process characterized by:
- **Independent increments**: future displacements $W_{t+u} - W_t$ are entirely independent of past states
- **Gaussian increments**: $W_{t+u} - W_t \sim \mathcal{N}(0, u)$.

The infinitesimal increment $dW_t$ behaves as a zero-mean Gaussian variable with variance $dt$. It is the **CLT limit of a discrete binomial random walk**. We'll deep-dive in a later series on the connections between Black-Scholes options pricing, heat diffusion, and quantum mechanics, stay tuned!
:::

Consider the standard heat-diffusion process where particles are purely driven by Brownian jittering:

<span id="eq-brownian"></span>

$$
\begin{equation}
    dX_t = g\, dW_t
\label{eq:brownian}
\end{equation}
$$

Over time $\Delta t$, particles jump according to $\mathcal{N}(0, g^2\Delta t)$.

:::theorem[Brownian motion as score flow]
The stochastic evolution $\eqref{eq:brownian}$ is equivalent at the distribution level to deterministic evolution under the velocity field

<span id="eq-score-flow"></span>

$$
\begin{equation}
    v(x) = -\dfrac{g^2}{2} \nabla \log \rho(x)
\label{eq:score-flow}
\end{equation}
$$

We're saying that the net, distribution-level effect of Brownian motion can be described by probability mass evolving deterministically according to the score gradient $\nabla \log \rho(x)$. Equivalently,
$$
    dX_t = g\, dW_t \quad \cong \quad dX_t = -\dfrac{g^2}{2} \nabla \log \rho
$$
:::

:::remark
This is the engine behind reducing SDE (diffusion) models to ODE (flow matching) models.
:::

<details>
<summary>Proof sketch: 1D discretization argument</summary>

Imagine bins of width $\Delta x$ at points $x_j$, each with probability mass $p_j$. Over time interval $\Delta t$, Brownian particles have equal probability of hopping left or right. The net flux of mass from bin $j-1$ to $j$ is:
$$
    J_j = \dfrac{1}{2\Delta t}(p_{j-1} - p_j)
$$
The flux from $j$ to $j+1$ is $J_{j+1} = \frac{1}{2\Delta t}(p_j - p_{j+1})$. The net change in mass at bin $j$ is:
$$
    \pd t p_j = J_j - J_{j+1} = \dfrac{1}{2\Delta t}(p_{j-1} + p_{j+1} - 2p_j)
$$

Converting to continuous density $\rho_j \approx p_j/\Delta x$:
$$
    \pd t \rho_j = \dfrac{\Delta x}{2\Delta t}(\rho_{j-1} + \rho_{j+1} - 2\rho_j)
$$

Recognizing the central difference approximation to the Laplacian $\Delta \rho := \sum_j \pd{j}^2 \rho$:
$$
    \Delta \rho_j \approx \dfrac{\rho_{j+1} - 2\rho_j + \rho_{j-1}}{(\Delta x)^2}
$$

we obtain:
$$
    \pd t \rho_j = \dfrac{(\Delta x)^2}{2\Delta t}\, \Delta \rho_j
$$

The variance of Brownian motion $g\, dW_t$ over time $\Delta t$ is $g^2\Delta t = (\Delta x)^2$, yielding:
$$
    \pd t \rho = \dfrac{g^2}{2}\, \Delta \rho = -\nabla \cdot(\rho v)
$$
for $v = -\frac{g^2}{2}\nabla \log \rho$. $\square$

</details>

### Brownian motion with drift, Fokker-Planck

Now add drift to the Brownian motion:

<span id="eq-sde"></span>

$$
\begin{equation}
    dX_t = f_t(X_t)\, dt + g_t\, dW_t
\label{eq:sde}
\end{equation}
$$

By linearity, the induced probability density $\rho_t\in \mathcal W_2$ evolves under velocity field:

<span id="eq-particle-velocity"></span>

$$
\begin{equation}
    v_t = f_t - \dfrac{g_t^2}{2} \nabla \log \rho_t
\label{eq:particle-velocity}
\end{equation}
$$

Applying the continuity equation $\pd t \rho_t = -\nabla \cdot (\rho_t v_t)$ yields the **Fokker-Planck equation**:

<span id="eq-fokker-planck"></span>

$$
\begin{equation}
    \pd t \rho_t = -\nabla \cdot (\rho_t f_t) + \dfrac{g_t^2}{2} \Delta \rho_t
\label{eq:fokker-planck}
\end{equation}
$$
Since curl components don't affect density evolution, write $f_t = \nabla V_t$,
and we're ready to state the celebrated Fokker-Planck equation.

:::theorem[Fokker-Planck equation]
Particles evolving under the SDE
$$
    dX_t = f_t(X_t)\, dt + g_t\, dW_t
$$
induce probability density evolution given by the **Fokker-Planck equation**:
$$
\begin{equation}
    \pd t \rho_t = -\nabla \cdot (\rho_t f_t) + \dfrac{g_t^2}{2} \Delta \rho_t
\label{eq:fokker-planck-pdd}
\end{equation}
$$
Writing $f_t = \nabla V_t$ (curl components vanish under divergence), this is equivalently deterministic flow under:
$$
\begin{equation}
    v_t = \nabla V_t - \dfrac{g_t^2}{2} \nabla \log \rho_t = \nabla\left(V_t - \dfrac{g_t^2}{2}\log \rho_t\right)
\label{eq:fokker-planck-ode}
\end{equation}
$$
:::

**Proof.** By linearity and the Brownian motion as score flow $\eqref{eq:score-flow}$, the following SDE and ODE are interchangeable:
$$
    dX_t = f_t(X_t)\, dt + g_t\, dW_t \leftrightarrow v_t = f_t - \dfrac{g_t^2}{2} \nabla \log \rho_t
$$

Applying the continuity equation $\pd t \rho_t = -\nabla \cdot(\rho_t v_t)$:
$$
\begin{aligned}
    \pd t \rho_t
    &= -\nabla \cdot\left(\rho_t v_t \right) \\
    &= -\nabla \cdot(\rho_t f_t) + \dfrac{g_t^2}{2}\nabla \cdot(\rho_t \nabla \log \rho_t) \\
    &= -\nabla \cdot(\rho_t f_t) + \dfrac{g_t^2}{2}\Delta \rho_t \quad  \square
\end{aligned}
$$

### Diffusion as Wasserstein gradient flow

In this subsection, we'll endow the Fokker-Planck (diffusion) equation with a macroscopic interpretation.
This exemplifies the theme that **Wasserstein geometry connects microscopic, particle-level evolution
with distribution-level extremization**. Let's begin with a special case:

:::corollary[Brownian motion maximizes entropy]
Pure Brownian diffusion $dX_t = g\, dW_t$ executes Wasserstein gradient ascent on the entropy functional $\mathcal{H}(\rho) = -\int \rho \log \rho$.
:::

**Proof.** The functional derivative of entropy is:
$$
    \dfrac{\delta \mathcal{H}(\rho)}{\delta \rho} = -(\log \rho + 1)
$$
By Otto's theorem (see [Part 2](/blogs/machine-learning/ot-generative-2-drifting-models/)), the Wasserstein gradient is:
$$
    \mathrm{grad}_W \mathcal{H}(\rho) = \nabla_x\left(\dfrac{\delta \mathcal{H}}{\delta \rho}\right) = -\nabla \log \rho
$$
Wasserstein gradient ascent follows $v = \mathrm{grad}_W \mathcal{H} = -\nabla \log \rho$, which matches $\eqref{eq:fokker-planck-pdd}$ with $f_t = 0$ and $g_t = \sqrt 2$. $\square$

:::remark
This corollary bridges the microscopic interpretation of **heat equation as Brownian motion** (particle jittering) with the macroscopic interpretation of **heat equation as process that minimizes entropy along the Wasserstein geometry**.

Why the Wasserstein geometry? There exists other geometries of probability distributions, Fisher-Rao being a notable one. Is the Wasserstein geometry, in some sense, canonical? For one, Fisher-Rao is not physical because it's agnostic towards rearrangements of the base space $\R^n$; mass can be teleported around. The rigorous physical grounding of the Wasserstein geometry in non-equilibrium dynamics is a deep rabbit hole which we'll not explore at the moment.
:::

Now the general case. For simplicity, we suppress time dependence on $V_t$ and $g_t$:

:::theorem[Fokker-Planck as Wasserstein gradient flow]
Define the **Boltzmann equilibrium distribution** from the potential $V_t$:
$$
    \sigma(x) = \dfrac{1}{Z} e^{-2V(x)/g^2}, \quad Z_t = \int e^{-2V/g^2}\, dx
$$
Define the scaled KL divergence:
$$
    \mathcal{F}(\rho) = \dfrac{g^2}{2} \, \mathrm{KL}(\rho \| \sigma) = \dfrac{g^2}{2} \int \rho \log \dfrac{\rho}{\sigma}\, dx
$$
The Fokker-Planck equation $\eqref{eq:fokker-planck}$ is Wasserstein gradient **descent** on $\mathcal F$:
$$
    \pd t \rho_t = -\nabla \cdot(\rho_t \cdot \mathrm{grad}_W \mathcal{F}(\rho_t))
$$
:::

<details>
<summary>Proof</summary>

Expanding the KL divergence:
$$
    \mathcal{F}(\rho) = \dfrac{g_t^2}{2} \int \rho \log \rho\, dx - \dfrac{g_t^2}{2} \int \rho \log \sigma_t\, dx
$$
Using $\log \sigma_t = -2V_t/g_t^2 - \log Z_t$:
$$
    \mathcal{F}(\rho) = \int \rho V_t\, dx + \dfrac{g_t^2}{2}\int \rho \log \rho\, dx + \text{const}
$$
The functional derivative is:
$$
    \dfrac{\delta \mathcal{F}}{\delta \rho} = V_t + \dfrac{g_t^2}{2}(\log \rho + 1) = \dfrac{g_t^2}{2} \log \dfrac{\rho}{\sigma_t} + \text{const}
$$
By Otto's theorem, the Wasserstein gradient is:
$$
    \mathrm{grad}_W \mathcal{F}(\rho) = \nabla_x\left(V_t + \dfrac{g_t^2}{2}\log \rho\right) = \nabla V_t + \dfrac{g_t^2}{2}\nabla \log \rho
$$
Gradient descent follows $v = -\mathrm{grad}_W \mathcal{F}$. From $\eqref{eq:particle-velocity}$ with $f_t = \nabla V_t$:
$$
    v_t = \nabla V_t - \dfrac{g_t^2}{2}\nabla \log \rho_t = -\left(-\nabla V_t + \dfrac{g_t^2}{2}\nabla \log \rho_t\right) = -\mathrm{grad}_W \mathcal{F}
$$
Applying the continuity equation $\pd t \rho_t = -\nabla \cdot(\rho_t v_t)$ completes the proof. $\square$

</details>

:::remark[Interpretation]
Fokker-Planck evolution minimizes KL divergence to the **Boltzmann equilibrium distribution** $\sigma_t \propto e^{-2V_t/g_t^2}$, where $V$ plays the role of energy and $g^2/2$ acts as temperature. The gradient $\nabla V$ pulls mass towards low potential while diffusion $\frac{g^2}{2}\Delta \rho_t$ spreads it out, balancing until $\rho_\infty = \sigma$.
:::

:::remark[Perspectives on the equilibrium distribution]
The Boltzmann equilibrium distribution $\sigma(x) = \frac{1}{Z}e^{-2V(x)/g^2}$ admits three equivalent interpretations:

1. **Thermodynamic principle**: $\sigma$ is the distribution that maximizes entropy subject to fixed expected energy:
   $$
   \sigma = \arg\max_{\rho} \left\{-\int \rho \log \rho\, dx \;\Big|\; \int \rho V\, dx = E_0\right\}
   $$
   The Lagrange multiplier enforcing the energy constraint gives the inverse temperature $2/g^2$.

2. **Dynamical equilibrium**: $\sigma$ is the unique stationary solution to the Fokker-Planck equation $\eqref{eq:fokker-planck}$. Setting $\pd t \rho = 0$:
   $$
   0 = -\nabla \cdot(\rho \nabla V) + \dfrac{g^2}{2}\Delta \rho = -\nabla \cdot\left(\rho \left(\nabla V - \dfrac{g^2}{2}\nabla \log \rho\right)\right)
   $$
   This holds when the drift and diffusion balance: $\nabla V = \frac{g^2}{2}\nabla \log \rho$, i.e., $\rho \propto e^{-2V/g^2}$.

3. **Optimal control**: In reinforcement learning, $\sigma$ is the **reward-optimal policy** that maximizes expected reward $-\int \rho V$ subject to KL proximity to a reference policy:
   $$
   \pi^* = \arg\max_{\pi} \left\{\mathbb{E}_\pi[-V] - \lambda\, \mathrm{KL}(\pi \| \pi_{\text{ref}})\right\}
   $$
   The solution is $\pi^* \propto \pi_{\text{ref}} \cdot e^{-V/\lambda}$, recovering the Boltzmann form when $\pi_{\text{ref}}$ is uniform and $\lambda = g^2/2$.
:::

### Bonus: Anderson's theorem

The Brownian motion as score flow equivalence $\eqref{eq:score-flow}$ allows us to elegantly prove a foundational theorem, **Anderson's theorem**, which converts between forward and backward processes. Incidentally, it will also be helpful when we consider options pricing as a reverse-time heat diffusion process. To motivate it, consider the following question:

1. We know how to describe forward-time trajectories of particles using the heat equation (or Fokker-Planck when there's potential).
2. If we want to magically play the tape backwards, which particle evolution equation will replicate the reverse-time behavior?

:::remark
This is not a trivial problem because while $-dt = d(-t)$, the Brownian drift term doesn't simply reverse: $-dW_t \neq dW_{-t}$. Our proof consists of converting the forward and backward SDEs into ODEs via $\eqref{eq:score-flow}$ and matching the velocity components.

Additionally, the choice of the same $g_t$ is canonical because forward and backward drift should be able to make use of the same amount of diffusion rate. Since we can microscopically tell the differences between Brownian motion with different diffusive constants and forward and reverse-time microscopic dynamics should be indistinguishable, we require $g_t\, d\tilde W_t$ in the reverse-time process.
:::

:::theorem[Anderson's theorem]
Given a forward stochastic process on $t \in [0, T]$ governed by the SDE
$$
    dX_t = f_t(X_t)\, dt + g_t \, dW_t
$$
Let $\rho_t$ be the marginal probability density of $X_t$. The reverse-time process which traces the exact same marginal distributions backwards from $T$ to $0$ is driven by the reverse SDE
$$
    d\tilde X_t = \left[f_t(\tilde X_t) - g_t^2 \nabla \log \rho_t(\tilde X_t) \right]\, dt + g_t\, d\tilde W_t
$$
:::

**Proof.** By the Brownian motion as score flow theorem $\eqref{eq:score-flow}$, the forward SDE is equivalent to the ODE:
$$
    dX_t = f_t\, dt + g_t\, dW_t \quad \leftrightarrow \quad dX_t = \left(f_t - \dfrac{g_t^2}{2} \nabla \log \rho_t\right)\, dt
$$
For the reverse-time process, time flows backwards: $d\tau = -dt$ where $\tau = T - t$. The reverse velocity must be the negative of the forward velocity to retrace the path:
Converting back to forward time $dt = -d\tau$:
$$
    d\tilde X_t = \left(\dfrac{g_t^2}{2}\nabla \log \rho_t - f_t\right)\, dt
$$

Converting this ODE back to an SDE using $\eqref{eq:score-flow}$: the reverse process is driven by an ODE under velocity field $\tilde v = \frac{g_t^2}{2}\nabla \log \rho_t - f_t$.

We have postulated from first principles that the $\tilde v$-ODE is equivalent to a SDE with drift $\tilde f_t$ and diffusion rate $g_t$. Applying the gauge degree of freedom to find $\tilde f_t$:
$$
    \tilde f_t - \dfrac{g_t^2}{2}\nabla \log \rho_t = \tilde v = \dfrac{g_t^2}{2}\nabla \log \rho_t - f_t
$$
The LHS corresponds to $\tilde v\, dt \leftrightarrow \tilde f_t\, dt + g_t\, d\tilde W_t$. The RHS comes from ODE reversal. Solving for $\tilde f_t$ yields
$$
    d\tilde X_t = \tilde f_t\, dt + g_t\, d\tilde W_t = \left(f_t - g_t^2 \nabla \log \rho_t\right)\, dt + g_t\, d\tilde W_t \quad \square
$$

## MLE interpretation of diffusion models

In this section, we provide a first-principles, maximum likelihood interpretation of diffusion generative models by applying the (dynamic) de Bruijn identity. This is the crux behind the celebrated paper [maximum likelihood training of score-based diffusion models](https://arxiv.org/pdf/2101.09258).

The maximum likelihood objective $D(p_{\mathrm{data}} \| q_{\mathrm{model}})$ is prone to divergence when the support of $q_{\mathrm{model}}$ does not cover $p_{\mathrm{data}}$. The de Bruijn identity **decomposes maximum likelihood into score matching over a noise spectrum**, allowing us to attenuate divergence in a controllable limit of the integral.

### the dynamic de Bruijn identity

Recall the definition of the Riemannian gradient:
$$
    \la \nabla \mathcal F(\gamma_t), \dot \gamma_t\ra = \dfrac{d}{dt} \mathcal F(\gamma_t)
$$
Consider the product manifold $\mathcal W \times \mathcal W$ equipped with the standard product metric, where $\dot p_1, \dot q_1, \dot p_2, \dot q_2$ are tangent vectors on the components:
$$
    \la \dot p_1\oplus \dot q_1, \dot p_2\oplus \dot q_2\ra_{\mathcal W\times \mathcal W} = \la \dot p_1, \dot q_1\ra + \la \dot p_2, \dot q_2\ra
$$
Define the functional $D(p \| q)$ on the product manifold $\mathcal W\times \mathcal W$. The Wasserstein gradient factors (note that $\nabla_p$ denotes the Wasserstein gradient w.r.t. $p$):
$$
    \nabla_{pq} D(p \| q) = \nabla_p D(p \| q) \oplus \nabla_q D(p \| q)
$$
Given trajectories $p_t, q_t$, we have
$$
\begin{aligned}
    \dfrac{d}{dt} D(p_t \| q_t)
    &=
    \la \nabla_p D(p_t \| q_t) \oplus \nabla_q D(p_t \| q_t)
    , \dot p_t \oplus \dot q_t \ra  \\
    &= \la \nabla_p D(p_t \| q_t), \dot p_t\ra + \la \nabla_q D(p_t \| q_t), \dot q_t\ra \\
    &= \la \nabla \log p_t - \nabla \log q_t, \dot p_t\ra - \la \dfrac{q_t}{p_t} \left(\nabla \log p_t - \nabla \log q_t\right), \dot q_t\ra \\
    &= \int \la \nabla \log p_t - \nabla \log q_t, \dot p_t\ra\, dp_t - \int \la \nabla \log p_t - \nabla \log q_t, \dot q_t\ra\, dp_t \\
    &= \mathbb E_{p_t} \la \nabla \log p_t - \nabla \log q_t, \dot p_t - \dot q_t\ra
\end{aligned}
$$
Note that the integral and gradient are over $x\in \R^n$ in sample space.
This is the general dynamic de Bruijn's identity: it's just the gradient expansion of the KL-divergence on product manifold.

:::theorem[the dynamic de Bruijn identity]
Given distribution trajectories $p_t, q_t\in \mathcal W$, the KL divergence rate of change is given by

<span id="eq-debruijn"></span>

$$
\begin{equation}
    \dfrac d {dt} D(p_t \| q_t) = \mathbb E_{p_t} \la \nabla \log p_t - \nabla \log q_t, \dot p_t - \dot q_t\ra
\label{eq:debruijn}
\end{equation}
$$
:::

### Application to various processes.

The variance exploding process injects Gaussian noise without attenuating the original signal. It's simply the heat equation with variable diffusion constant.

In this section, we consider two common process and the application of theorem $\eqref{eq:debruijn}$ to each:

1. The <u>heat process</u> with noise schedule $g_t$. it's defined on $t\in [0, \infty)$ and the variance of the marginal explodes.
2. The <u>variance preserving process</u>, defined on $t\in [0, \infty)$ and $p_\infty\to \mathcal N(0, I)$.

#### Heat process

:::definition[heat process]
The variance-exploding process is defined to obey the SDE:
$$
    dX_t = g_t\, dW_t
$$
Also recall its Wasserstein gradient
$$
    \pd t \rho_t = -\nabla \cdot (\rho v), \quad v(x) = -\dfrac{g_t^2}{2} \nabla \log \rho(x)
$$
Here, $g_t$ is also known as **the noise schedule**. Denote the accumulated variance as $\sigma^2_t$, then the marginal distribution is
$$
    \rho(x_t\mid x_0) = \mathcal N(x_t; x_0, \sigma^2_tI), \quad \sigma^2_t = \int_0^t\, g_s^2\, ds
$$
If $p_t, q_t$ are both subject to the VE process, substituting the dynamic de Bruijn identity yields

<span id="eq-heat-debruijn"></span>
$$
\begin{equation}
\begin{aligned}
    D(p_0 \| q_0)
    &= D(p_T \| q_T) - \int_0^T \mathbb E_{p_t} \la \nabla \log p_t - \nabla \log q_t, \dot p_t - \dot q_t\ra\, dt \\
    &= D(p_T \| q_T) + \dfrac 1 2 \int_0^T g_t^2 \cdot \mathbb E_{x\sim p_t} \| \nabla \log p_t(x) - \nabla \log q_t(x) \|^2\, dt
\end{aligned}
\label{eq:heat-debruijn}
\end{equation}
$$
Sometimes this specialized result is known as the dynamic de Bruijn identity.
:::

:::remark
The equation $\eqref{eq:heat-debruijn}$ demonstrates a fundamental interpretation of the de Bruijn identity. Note the Euclidean score matching loss $\|\nabla \log p_t - \nabla \log q_t\|$. This term is well-defined for all $t>0$. The divergence of disjoint support is hidden in the tail $t\to 0$. We have expanded the KL divergence as score matching over a full noise spectrum.
:::

#### Variance-preserving process

:::definition[variance-preserving process]
The variance-preserving (VP) process is defined by the following SDE with noise schedule $\beta_t$:
$$
    dX_t = -\dfrac 1 2 \beta_t X_t\, dt + \sqrt{\beta_t}\, dW_t
$$
By Fokker-Planck, the equivalent velocity field dictating Wasserstein gradient flow is
$$
    v(x) = -\dfrac 1 2 \beta_t \left(x - \nabla \log \rho\right)
$$
It's related to the heat process up to a time-dependent rescaling of sample space scale. The conditional distribution is
$$
    \rho(x_t \mid x_0) = \mathcal N\left(x_t; \sqrt{\alpha_t} x_0, (1-\alpha_t) I\right), \quad \alpha_t = e^{-\int_0^t \beta_s\, ds}
$$
Substituting into the dynamic de Bruijn identity, the drift terms magically cancel and terminal divergence vanishes at $t\to \infty$, yielding
$$
    D(p_0 \| q_0) = \dfrac 1 2 \int_0^\infty \beta_t \cdot \mathbb E_{p_t} \| \nabla \log p_t - \nabla \log q_t \|^2\, dt
$$
:::

:::remark[interpretation as OU-process]
The variance-preserving process is related to the Ornstein-Uhlenbeck process, which is the canonical continuous-time model for mean-reverting behavior given by
$$
    dX_t = \theta(\mu - X_t)\, dt + \sigma\, dW_t, \quad \theta, \sigma>0
$$
Here $\mu$ is long-term mean, $\theta$ is restorative strength, and $\sigma$ volatility.
:::

## Tweedie's formula and flow matching

Here, we clarify some of the most heavily overloaded deconcepts in flow matching / diffusion, prove that flow matching is MLE by Tweedie's formula, and conclude with some possibly backfit first-principles analysis on why flow matching have come to dominate generative modeling. We'll see that Tweedie's formula is the engine behind high-dimensional scalability of flow matching models.

### Tweedie's formula

Tweedie's formula states that for processes with Gaussian noise, the true posterior mean (optimal denoiser) can be computed purely from **the score of the marginal distribution**.

Application of this formula sheds light on the fundamental unification of ODE-style flow matching and SDE-style diffusion as maximum likelihood estimation via flow matching. It unifies the following objectives:

1. Predicting the score
2. Predicting the posterior mean (denoising)
3. Predicting the denoising vector field.

<span id="thm-tweedie"></span>
:::theorem[Tweedie's formula]
Let $z\sim P_z$ be an unobserved true signal with an arbitrary prior, then
$$
\begin{equation}
    x\mid z\sim \mathcal N(\alpha z, \Sigma) \implies \alpha \mathbb E[z\mid x] = x + \Sigma \nabla \log \rho
\label{eq:tweedie}
\end{equation}
$$
where $\rho$ is the marginal distribution of $x$ and $\alpha\neq 0$.
:::

<details>
<summary>Proof sketch: substitute the Gaussian conditional score formula</summary>

Without loss of generality we can substitute $z=\alpha z$ so $\alpha=1$. First compute the conditional distribution and score:
$$
\begin{aligned}
    p(x\mid z)
    &\propto \exp \left[-\dfrac 1 2 (x-z)^T \Sigma^{-1}(x-z)\right]  \\
    \nabla p(x\mid z) &= -p(x\mid z) \Sigma^{-1}(x-z)
\end{aligned}
$$
Substitute into the score:
$$
\begin{aligned}
    \nabla \log \rho(x)
    &= \df 1 {\rho(x)} \int \nabla p(x\mid z) p(z)\, dz \\
    &= - \df 1 {\rho(x)} \int \Sigma^{-1}(x-z) \cdot p(x\mid z) p(z)\, dz \\
    &= \int \Sigma^{-1}(z-x) p(z\mid x)\, dz \\
    &= \mathbb E_{z\mid x} \Sigma^{-1}(x - z) = \Sigma^{-1}\left(x - \mathbb E[z\mid x]\right)
\end{aligned}
$$

Rearranging yields the result. $\square$

</details>

:::remark
This is a highly nontrivial theorem. In general, posterior quantities depend on the prior distribution of $z$. Here, the key property $\nabla_x p(x\mid z) = -p(x\mid z)\Sigma^{-1}(x-z)$ provided the closed-form $p(x\mid z)$ which can be rearranged into $p(z\mid x)$.
:::

:::remark[application, *IMPORTANT*!]
Flipping the theorem on its head, Tweedie reduces a **density estimation** problem $\nabla \log \rho$ into a **mean-estimation** problem $\mathbb E[z\mid x]$. Suppose $z$ is an arbitrary distribution, we can apply $\sigma$ amount of noise and solve the mean-estimation problem to approximate $\log \rho_z \approx \log \rho_x$; here $\sigma$ controls the bias-variance tradeoff.
:::

Next, let's consider compact $t\in [0, 1]$ with $t=0$ being data / model boundary distribution, and $t=1$ being $\mathcal N(0, I)$. Recall in [Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/#unifying-static-and-dynamical-perspectives) that the Wasserstein-2 geodesic distance is realized by straight-line transport. Solving this ODE corresponds to flow matching, and we derive the de Bruijin MLE interpretation below:

### The flow matching process

<!-- Recall two critical results from Part 1:

1. Given a transport plan (coupling) $\pi(x, y)$, the optimal-transport flow which realizes the plan realizes straight-line transport.
2. The Wasserstein-2 distance is realized by the optimal transport plan. By Banamou-Brenier, the straight-line transports which realize the optimal transport plan never intersect. -->

We analyze the flow matching process by applying the following construction:

1. We <span class="question">define</span> an independent coupling between data and i.i.d. Gaussian noise.
2. Given the independent coupling, we <span class="question">define</span> straight-line transport <span class="question">conditioned upon endpoints</span>. This supplies the trivial endpoints-conditioned vector field.
3. Using 1-2 above, we can derive the <span class="question">data-conditioned</span> vector field $v_t(x_t\mid x_0)$. This is a straight vector field.
4. Using 3, we derive the marginal vector field $v_t(x_t)$. This <span class="question">is not a straight vector field</span>.
5. Note that the <span class="question">noise-conditioned vector field</span> $v_t(x_t\mid x_1)$ is also not straight.

It's extremely important to differentiate between vector fields by what they're conditioned on.

:::definition[the flow matching ODE process]
Fixing data (or model) distribution $p_0$ and stationary noise distribution $p_1=\mathcal N(0, I)$, flow matching uses **independent coupling** between noise and data:
$$
    \pi(x_0, x_1) = p_0(x_0) p_1(x_1)
$$
Recalling our discussion of the optimal transport field from [Part 1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/#from-particles-to-fluid), the OT process **for the conditional coupling** implements straight line transport between endpoints.
The conditional density implements linear interpolation between data sample and Gaussian noise:
$$
    \rho_t(x_t\mid x_0) = \mathcal N\left(x_t; (1-t) x_0, t^2I\right)
$$
:::

The next step is deriving the conditional and marginal vector fields.

Fixing both data $x_0$ and noise $x_1$, the vector field $v(x_t\mid x_0, x_1) = x_1 - x_0$ on valid interpolation $x_t = \bar t x_0 + tx_1$ [^elsewhere].

:::proposition[data-conditioned flow-matching vector field]
Fixing data $x_0$, the conditional vector field is straight-line: at $x_t$, the trajectory endpoint is $\mathbb E[x_1\mid x_t, x_0]$. The projected displacement, which gives velocity under straight-line transport, is the current displacement scaled by the time progress $1/t$.
$$
\begin{equation}
\begin{aligned}
    v_t(x_t\mid x_0)
    &= \dfrac 1 t (x_t - x_0)
\end{aligned}
\label{eq:data-conditional-fn}
\end{equation}
$$
:::
For general endpoints-conditioned transport to initial position conditioned transport, apply
$$
    v_t(x_t\mid x_0) = \int v_t(x_t\mid x_0, x_1)\, d\rho(x_1\mid x_t, x_0)
$$

Let's proceed to derive the marginal vector field. Recall the decomposition:
$$
\begin{equation}
    v_t(x_t) = \int v_t(x_t\mid x_0) \, d\rho(x_0\mid x_t)
\label{eq:marginal-vec}
\end{equation}
$$
- LHS denotes the macroscopic fluid velocity at position $x_t$, time $t$.
- Fluid at this space-time are composed of fluid initially starting across a range of initial positions $x_0$, each with their own velocity.
- The RHS denotes an ensemble of particle velocity $v_t(x_t\mid x_0)$ from initial positions $x_0$, averaged by their constituent ratio $\rho(x_0\mid x_t)$.

<span id="prp-fm"></span>
:::proposition[flow matching formulas]
Applying $\eqref{eq:marginal-vec}$ with $\eqref{eq:data-conditional-fn}$ yields
$$
\begin{equation}
\begin{aligned}
    v_t(x_t)
    &= \int \dfrac 1 t (x_t - x_0)\, d\rho(x_0\mid x_t) = \dfrac 1 t \left(x_t - \mathbb E[x_0 \mid x_t]\right)
\end{aligned}
\label{eq:fm-marginal-tweedie}
\end{equation}
$$
This looks difficult, but luckily, the independent coupling + straight-line endpoint transport implies conditional Gaussian noise $x_t\mid x_0 \sim \mathcal N(\bar t x_0, t^2 I)$. Applying Tweedie $\eqref{eq:tweedie}$ yields
$$
    \mathbb E[x_0\mid x_t] = \dfrac{x_t + t^2 \nabla \log p_t(x_t)}{1-t}
$$
Simplify to obtain
$$
\begin{equation}
    v_t(x_t) = -\left(
        \dfrac{1}{1-t} x_t + \dfrac{t}{1-t} \nabla \log p_t(x_t)
    \right)
\label{eq:fm-marginal-velocity}
\end{equation}
$$
The drift terms cancel in $v^p_t - v^q_t$ and $D(p_1 \| q_1)=0$, yielding the KL decomposition
<span id="eq-fm-debruijn"></span>
$$
\begin{equation}
    D(p_0 \| q_0)
    = \int_0^1 \dfrac{t}{1-t} \mathbb E_{p_t} \|\nabla \log p_t - \nabla \log q_t\|^2\, dt
    \label{eq:fm-debruijn}
\end{equation}
$$
:::

:::remark[estimate v. transport]
Note that if $x_0\sim p$ are e.g. images, then $x_0$ are generally sharp images, while the estimates $\hat x_0$ are generally blurry means. In particular, $\hat x_0(x_1\sim p_1)$ is just the unconditional mean.
:::

:::remark[straight vector fields]
Note that the **data-conditioned** vector field $v_t(x_t\mid x_0)$ is straight. Similarly, the **noise-conditioned vector field** $v_t(x_t\mid x_1) = (x_1 - x_t) / \bar t$ is also straight. However, the marginal vector field $v_t(x_t)$ is not straight.
:::

### Flow matching in practice

Let's look at the [preceding proposition](#prp-fm) operationally:

1. Straight-line ODE transport equation $\eqref{eq:fm-marginal-velocity}$ tells us that given the (noise) spectrum-indexed family of scores $\nabla \log \rho_t$, we can integrate $v_t(x_t)$ from $x_1$ to $x_0$ to generate a sample.
2. de Bruijn formula $\eqref{eq:fm-debruijn}$ tells us that minimizing score MSE yields maximum likelihood.

If we're happy generating samples by integrating a vector field, we only need to approximate the score $\nabla \log p_t(x)$. This is a parameric density estimation problem. But the score target $\nabla \log p_t$ looks untractable.

Our escape hatch is equation $\eqref{eq:fm-marginal-tweedie}$. Reparameterize $f_\theta(x, t) \approx \mathbb E_p[x_0\mid x_t]$, then the score of our generative model is
$$
    \nabla \log q_t(x_t) = \dfrac 1 {t^2} \left[\bar t f_\theta(x_t) - x_t\right]
$$
:::remark
Note that $q_t$ is implicitly defined by the sampling process where we push $x_1\sim \mathcal N(0, I)$ backwards through the ODE $\eqref{eq:fm-marginal-velocity}$ with $\nabla \log p_t(x_t) \mapsto f_\theta(x, t)$.
:::
Rewriting $\nabla \log p_t$ using $\mathbb E_p[x_0\mid x_t]$ via Tweedie, and $\nabla \log q_t$ using the implicit parameterization above, the de Bruijn equation $\eqref{eq:fm-debruijn}$ yields
$$
\begin{aligned}
    D(p_0\| q_0)
    &= \int_0^1 \dfrac{t}{1-t} \cdot \left(\dfrac {\bar t} {t^2}\right)^2 \mathbb E_{x_t \sim p_t}\|\mathbb E_{x_0\sim p(x_0\mid x_t)}[x_0\mid x_t] - f_\theta(x_t, t) \|^2\, dt \\
    &= \mathbb E_{t\sim \mathrm{Unif}[0, 1]} \left[\dfrac {\bar t} {t^3} \mathbb E_{x_t} \| \mathbb E_{x_0}[x_0\mid x_t] - f_\theta(x_t, t) \|^2 \right]
\end{aligned}
$$
Note that sampling are all from the data $p$-process. But mean estimation is very easy, applying MSE decomposition while fixing $x_t$:
$$
\begin{aligned}
    \mathbb E_{x_0\sim p(\cdot \mid x_t)} \| f_\theta(x_t, t) - x_0 \|^2
    &= \mathbb E_{x_0} \| f_\theta(x_t, t) - \mathbb E[x_0\mid x_t] - (x_0 - \mathbb E[x_0\mid x_t]) \|^2 \\
    &= \mathbb E_{x_0} \| \mathbb E_{x_0}[x_0\mid x_t] - f_\theta(x_t, t) \|^2 + \mathrm{Var}[x_0 \mid x_t]
\end{aligned}
$$
The cross term vanishes; this is the law of total variation.

:::remark
This analysis shows that the irreducible noise at level $t$ is $\mathrm{Var}[x_0\mid x_t]$.
:::

[^elsewhere]: It doesn't matter where $v_t$ is elsewhere because $\rho(x_t\mid x_0, x_1)$ does not have mass elsewhere.
