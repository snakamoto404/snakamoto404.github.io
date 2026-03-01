---
title: "OT for generative modeling 3 — Diffusion as Wasserstein Gradient Flow"
date: 2026-03-01
summary: "We unpack diffusion under the Wasserstein geometry and show that it's maximum likelihood."
---



## Contents

- [Contents](#contents)
- [Physics of diffusion](#physics-of-diffusion)
  - [Brownian motion as score flow](#brownian-motion-as-score-flow)
  - [Brownian motion with drift, Fokker-Planck](#brownian-motion-with-drift-fokker-planck)
  - [Diffusion as Wasserstein gradient flow](#diffusion-as-wasserstein-gradient-flow)
  - [Bonus: Anderson's theorem](#bonus-andersons-theorem)
- [MLE interpretation of diffusion models](#mle-interpretation-of-diffusion-models)
  - [Gradient flow dissipation](#gradient-flow-dissipation)
  - [Single-variable warmup](#single-variable-warmup)
  - [De Bruijin full proof](#de-bruijin-full-proof)
  - [Perspectives on diffusion models](#perspectives-on-diffusion-models)

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

In this subsection, we'll endow the Fokker-Planck equation with macroscopic purpose.
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

The maximum likelihood objective $D(p_{\mathrm{data}} \| q_{\mathrm{model}})$ is prone to divergence when the support of $q_{\mathrm{model}}$ does not cover $p_{\mathrm{data}}$. The de Bruijn identity breaks down maximum likelihood into score matching over a trajectory and "hides" this divergence in a controllable limit of the integral.


:::theorem[the dynamic de Bruijn Identity]
Let $p_t, q_t$ be probability distributions over $\mathbb{R}^n$ evolving over time $t \in [0, 1]$. Assume both distributions evolve according to the same heat equation with scaling factor $g(t)$:
$$
\partial_t p_t = \frac{g(t)^2}{2} \Delta p_t \quad \text{and} \quad \partial_t q_t = \frac{g(t)^2}{2} \Delta q_t
$$
with boundary conditions $p_1 = p$, $q_1 = q$, and $p_0 = q_0$. Then the KL divergence decomposes as:
<span id="eq-de-bruijn"></span>
$$
\begin{equation}
D(p_1 \| q_1) = \frac{1}{2} \int_0^1 g(t)^2 \mathbb{E}_{x \sim p_t} \left\| \nabla \log p_t(x) - \nabla \log q_t(x) \right\|^2\, dt
\label{eq:de-bruijn}
\end{equation}
$$
:::

This identity transforms the **distribution-level divergence** $D(p \| q)$, which may be infinite when supports don't overlap, into a **time-averaged score-matching objective** that remains well-defined along the diffusion path.

We prove this by noting a trivial gradient flow dissipation identity and applying it to the special case of Wasserstein geometry. We prove the single-variate de Bruijn identity then allow both $p_t, q_t$ to vary using a product manifold.

### Gradient flow dissipation

The dynamic De Bruijin identity can look much more daunting than the general statement, so we begin with the more intuitive general statement:

:::definition[gradient flow dissipation]
Given a Riemann manifold $\mathcal M$ with gradient $\nabla$, and a scalar function $\mathcal F:\mathcal M\to \R$, for a gradient flow trajectory $\gamma_{t\in [0, 1]}\in \mathcal M$ satisfying $\dot \gamma_t = -\nabla \mathcal F(\gamma_t)$, the scalar function dissipates according to:
$$
    \dfrac{d}{dt} \mathcal F(\gamma_t) = - \| \nabla \mathcal F(\gamma_t) \|^2
$$
Note that the norm $\|\cdot \|^2$ is under the manifold metric, and that the gradient is the Riemann gradient defined by the metric:
$$
    \la \nabla \mathcal F(\gamma_t), \dot \gamma_t\ra = \dfrac{d}{dt} \mathcal F(\gamma_t)
$$
:::
The proof is a trivial application of the standard chain rule (think in Euclidean space):
$$
    \dfrac{d}{dt} \mathcal F(\gamma_t) = \la \nabla \mathcal F(\gamma_t), \dot \gamma_t \ra = -\| \nabla \mathcal F(\gamma_t)\|^2
$$

### Single-variable warmup

### De Bruijin full proof

### Perspectives on diffusion models