---
title: "OT for generative modeling 2 — Wasserstein gradients and drifting models"
date: 2026-02-23
summary: "We look at Kaiming Deng et al.'s Drifting Models, interpret the antisymmetric drifting field as Wasserstein gradient flow on the reverse KL between kernel-smoothed distributions, and develop the connection to maximum likelihood estimation."
---

Diffusion and flow matching split the generative problem into two phases: at *training* time, learn a vector field; at *inference* time, integrate an ODE or SDE through that field to produce a sample. The integration is expensive, and a great deal of recent work has gone into compressing it: distillation, consistency models, progressive reduction.

Recent work on **Drifting Models** ([Deng et al., 2026](https://arxiv.org/abs/2602.04770)) defines a "drifting field" $\mathbf{V}_{p,q}(\mathbf{x})$ that tells each generated sample which direction to move, requires antisymmetry ($\mathbf{V}_{p,q} = -\mathbf{V}_{q,p}$) so that the field vanishes at equilibrium ($p = q$), and train the generator to chase its own drifted targets.

We currently have a mechanical interpretation -- the loss going to zero produces desirable behavior. But equipped with the Wasserstein machinery we built in [parts 0](/blogs/machine-learning/ot-generative-0-static/)–[1](/blogs/machine-learning/ot-generative-1-wasserstein-geometry/), we can say something much sharper. The antisymmetric drifting field is the Wasserstein gradient of a distribution discrepancy; the training dynamics execute gradient descent on the manifold $\mathcal{W}_2$; and the $L^2$ regression loss against drifted targets is, through the Wasserstein bridge, a gradient step on a proper statistical divergence. The whole paradigm admits a precise connection to maximum likelihood.

For those looking for novel content, the main results of this post are as follows:

1. [**Statistical interpretation of Gaussian drifting**](#gaussian-kernel-smoothing-implements-reverse-kl): we show that drifting with a Gaussian kernel implements Wasserstein gradient descent on the reverse, mode-seeking KL divergence $\mrm{KL}(\tilde q_\theta \| \tilde p_{\mathrm{data}})$ between KDE-smoothed distributions. The stop-grad loss implements gradient pullback from sample to parameter space.
2. [**Maximum likelihood modification**](#proposition-maximum-likelihood-drifting): we derive the drifting field for the maximum likelihood (forward KL) objective $\mrm{KL}(\tilde p_{\mathrm{data}} \| \tilde q_\theta)$. The changes to the current paradigm are minimal: reweigh by the density ratio $Z_p/Z_q$ and use the Gaussian (instead of Laplace) kernel. The resulting drifting field is notably not antisymmetric.

## Contents

- [Contents](#contents)
- [Formulation](#formulation)
  - [The drifting field](#the-drifting-field)
- [Wasserstein Gradient Flow](#wasserstein-gradient-flow)
  - [The Kullback-Leibler functional](#the-kullback-leibler-functional)
  - [Gradients on manifolds](#gradients-on-manifolds)
  - [Proving Otto's theorem](#proving-ottos-theorem)
- [Statistical interpretation of drifting](#statistical-interpretation-of-drifting)
  - [Gaussian kernel smoothing implements Reverse KL](#gaussian-kernel-smoothing-implements-reverse-kl)
  - [Implementing forward KL](#implementing-forward-kl)
  - [Proposition: maximum likelihood drifting](#proposition-maximum-likelihood-drifting)

## Formulation

The drifting models paradigm consists of:

- a **drifting field** that tells generated samples which direction to move
- a **training loop** that chases drifted targets

In this generative paradigm, we're given samples from $P_{\mathrm{data}} \in \mathcal W_2(\R^n)$, we consider an initial noise distribution $P_{\mathrm{noise}} \in \mathcal W_2(\R^d)$, and a parametric model generator $f_\theta: \mathbb R^d\to \mathbb R^n$. Denote the pushforward measure by $P_{\mathrm{model}} \in \mathcal W_2(\R^n)$. To be consistent with the paper, we abbreviate
$$
    p := P_{\mathrm{data}}, \quad q_\theta := P_{\mathrm{model}, \theta}
$$
The work considers general antisymmetric drift fields $V_{p, q}: \mathbb R^n\to \mathbb R^n$; each field is a vector field on the sample space. The training loop consists of iteratively minimizing

$$
    \mathcal L = \mathbb E_{\epsilon\sim P_{\mathrm{noise}}} \| f_\theta(\epsilon) - \text{stopgrad} \left[
        f_\theta(\epsilon) + V_{p, q_\theta}(f_\theta(\epsilon))
    \right] \|^2
$$
Note that $p=q \implies \mathcal L=0$. Further note that if $V = \nabla \varphi$, then
$$
\begin{aligned}
    \nabla_\theta \mathcal L
    &\propto
    \nabla_\theta \mathbb E_{\epsilon \sim P_{\mathrm{noise}}} \left[J_{f_\theta}^T\left[
        f_\theta(\epsilon) - f_\theta(\epsilon) + \nabla_{f_\theta(\epsilon)} \varphi(f_\theta(\epsilon))
    \right]\right] \\
    &= \nabla_\theta \mathbb E_{\epsilon \sim P_{\mathrm{noise}}} \left[
        J_{f_\theta}^T \nabla \varphi
    \right]
\end{aligned}
$$
> The stop-grad loss exactly implements the pullback of the gradient. If $V$ is a gradient on sample space $\R^n$, then $\nabla_\theta \mathcal L$ is the pullback of the gradient in parameter space.

### The drifting field

Let's consider the authors' choice of the drifting field
:::definition[canonical drifting field]

Consider the following antisymmetric drifting field evaluated at sample space $x\in \mathbb R^n$:
$$
\begin{aligned}
    V_{p,q}(x)
    &=
    \underbrace{\frac{1}{Z_p(x)}\E_{y^+\sim p}\!\left[k(x, y^+)(y^+ - x)\right]}_{V_p^+(x):\;\text{attraction to data}} \;-\; \underbrace{\frac{1}{Z_q(x)}\E_{y^-\sim q}\!\left[k(x, y^-)(y^- - x)\right]}_{V_q^-(x):\;\text{repulsion from model}} \\
    &= \mathbb E_{y^+\sim p, y^-\sim q} \left[
        \dfrac{k(x, y^+) k(x, y^-)}{Z_pZ_q}(y^+ - y^-)
    \right]
\end{aligned}
$$
The authors chose the Laplace kernel
$$
    k(x,y) = \exp(-\|x - y\|/\tau)
$$
The per-point normalization translates to softmax weighting
$$
    Z_p(x) = \E_{y^+\sim p}[k(x, y^+)], \quad Z_q(x) = \E_{y^-\sim q}[k(x, y^-)]
$$
:::


## Wasserstein Gradient Flow

We take a step back to develop the theory of **Wasserstein gradient flow** (and return to capital letters): given a probability distribution $Q_\theta$, we can assign some loss / preference to it by some functional $\mathcal F:\mathcal W_2\to \R$. What happens to the probability distribution as we try to minimize $\mathcal F(q_\theta)$ by gradient descent?

### The Kullback-Leibler functional

Fixing data distribution $p$, maximizing likelihood of the data under the model is equivalent to minimizing the KL divergence:
$$
\begin{aligned}
    \mrm{KL}(P\|Q_\theta)
    &= \mathbb E_{x\sim P} \left[
        \log \dfrac{P(x)}{Q_\theta(x)}
    \right] \\
    &= \mathbb E_{x\sim P} \log P(x) - \mathbb E_{x\sim P}\log Q_\theta(x)\\
    &= \int dP\, \log P - \int dP\, \log Q
\end{aligned}
$$
For more interesting properties of KL divergence, see [these notes](https://nlyu1.github.io/classical-info-theory/kullback-leibler-divergence.html). From a SGD perspective, minimizing KL is equivalent to maximizing likelihood when empirical samples are i.i.d from $P$:
$$
    \nabla_\theta \, \mathrm{KL}(P\|Q_\theta) = -\nabla_\theta \mathbb E_{x\sim P} \, \log Q_\theta(x)
$$


### Gradients on manifolds

I like to interpret differential geometry as the "lifting" of Euclidean constructs into locally Euclidean manifolds. Gradients are no different. In Euclidean space, given a curve $\gamma: [0, 1]\to \R^n$ and a linear function $f:\R^n\to \R$, the chain rule yields
$$
    \df d {dt} f(\gamma(t)) = \dot \gamma(t) \cdot \nabla\big|_{\gamma(t)} f
$$
Lifting the inner product to the manifold metric, we can use this to define gradients on manifolds:

:::definition[Wasserstein gradients]

Given a scalar function $f:\mathcal W_2\to \R$, the gradient of $f$ at the point $P\in \mathcal W_2$ is the unique tangent vector $v=\mathrm{grad}_W f(P)$ such that, for any curve $\gamma(t)$ with $\gamma(0)=P$, we have
$$
    \dfrac{d}{dt} f(\gamma(t)) = \la \dot \gamma(t), v\ra_W = \mathbb E_P \la \dot \gamma(t), v\ra
$$
where $\la\cdot, \cdot\ra_W$ is our familiar Wasserstein metric on $\mathcal W_2$, and $\la\cdot, \cdot\ra$ in the second equality is the familiar Euclidean metric, after we expanded the definition of the Wasserstein metric.
:::

Now, we're equipped to state a major result in [Otto calculus](https://www.math.toronto.edu/mccann/assignments/477/Otto01.pdf). We'll prove it shortly.

:::theorem[fundamental theorem of Otto calculus]
Given a probability functional $\mathcal F:\mathcal W_2\to \R$, its Wasserstein gradient can be computed as
$$
    \mathrm{grad}_W \mathcal F(P) = \nabla_x \left(\dfrac{\delta \mathcal F}{\delta P}\right)
$$
The theorem should look fairly intuitive: on the RHS, we compute the pointwise derivative of $\mathcal F$ w.r.t. the point density at $P(x)$ and use this as the potential. The theorem tells us that the direction of steepest $\mathcal F$-ascent is the gradient of this (functional derivative) potential.
:::

Several remarks are in order:
1. Despite appearing like a definition, this is a *theorem*! The general differential-geometry gradient exists, but its general computation does not usually admit such easy form.
2. The expression $\dfrac{\delta\mathcal F}{\delta P}: \mathbb R^n\to \R$ is **a scalar function on the sample space** that's usually known as the **functional derivative**. Its values are the point-wise derivatives of $F$ w.r.t. $P$.

Again, the functional derivative **is a scalar function on the sample space**. Its definition is best demonstrated by two useful examples:

:::example
For the entropy functional $\mathcal H(P) = -\int dP\, \log P = -\int P(x) \log P(x) dx$, applying the product rule yields:
$$
    \dfrac{\delta \mathcal H(P)}{\delta P}(x) = -(\log P(x) + 1)
$$

The KL functional $\mathrm{KL}(P\|Q_\theta)$ has two arguments:
$$
    \mathrm{KL}(P\|Q_\theta) = \int dP\, \log P - \int dP \log Q_\theta
$$

Taking the functional derivative w.r.t. the data distribution $P$, we treat $Q_\theta$ as a constant:
$$
\begin{aligned}
    \dfrac{\delta \mathrm{KL}(P\|Q_\theta)}{\delta P}(x)
    &= \pd P\Big( P \log P - P \log Q_\theta \Big) \\
    &= \log P(x) - \log Q_\theta(x) + 1
\end{aligned}
$$
Similarly w.r.t. $Q_\theta$:
$$
\begin{aligned}
    \dfrac{\delta \mathrm{KL}(P\|Q_\theta)}{\delta Q_\theta}(x)
    &= \partial_{Q_\theta} \Big( - P \log Q_\theta \Big) = -\dfrac{P(x)}{Q_\theta(x)}
\end{aligned}
$$
:::

:::example[applying Otto's theorem to entropy]

From above, $\frac{\delta \mathcal H}{\delta P} = -(\log P + 1)$. Applying the theorem:
$$
    \mrm{grad}_W \mathcal H(P) = \nabla_x\left(-\log P - 1\right) = -\nabla \log P
$$
The Wasserstein gradient of entropy is the **negative score**. Gradient ascent on entropy has velocity $v = \mrm{grad}_W\mathcal H = -\nabla \log P$. Plugging into the continuity equation:
$$
    \pd t P = \nabla \cdot(P\,\nabla \log P) = \Delta P
$$
This is the **heat equation**: heat diffusion is Wasserstein gradient ascent of entropy.
:::

:::example[applying Otto's theorem to forward KL]

Apply to $Q_\theta$ in $\mrm{KL}(P\|Q_\theta)$. From above, $\frac{\delta\mrm{KL}(P\|Q_\theta)}{\delta Q_\theta} = -P/Q_\theta$. Applying the theorem:
$$
\begin{aligned}
    \mrm{grad}_W \mrm{KL}(P\|Q_\theta)\big|_{Q_\theta}
    = \nabla_x\!\left(-\frac{P}{Q_\theta}\right)
\end{aligned}
$$
Gradient descent velocity: $v = \nabla_x(P/Q_\theta)$.
:::

<span id="ex-otto-reverse-kl"></span>
:::example[applying Otto's theorem to reverse KL]

Apply to $Q_\theta$ in $\mrm{KL}(Q_\theta\|P)$. The functional derivative is
$$
    \frac{\delta\mrm{KL}(Q_\theta\|P)}{\delta Q_\theta} = \log Q_\theta - \log P + 1
$$
Applying the theorem:
$$
\begin{equation}
    \mrm{grad}_W \mrm{KL}(Q_\theta\|P)\big|_{Q_\theta} = \nabla \log Q_\theta - \nabla \log P = s_{Q_\theta} - s_P
\label{eq:reverse-kl-gradient}
\end{equation}
$$
The Wasserstein gradient is a **score difference**. Gradient descent velocity: $v = s_P - s_{Q_\theta}$. Particles flow in the direction where the data score exceeds the model score. This is exactly the drifting field's structure — but it requires $\nabla \log P$, the data score. With empirical samples (Diracs), this is undefined: the **Dirac trap** that we return to in [the next section](#gaussian-kernel-smoothing-implements-reverse-kl).
:::

### Proving Otto's theorem

We need to show that $v = \nabla_x \frac{\delta \mathcal F}{\delta P}$ satisfies the gradient definition for all test tangent vectors $u = \nabla\psi \in T_P\mathcal W_2$.

By the gradient definition, $v = \mrm{grad}_W\mathcal F$ is the unique tangent vector satisfying
$$
    \df d{d\epsilon}\mathcal F(P + \epsilon\,\delta P)\Big|_{\epsilon=0} = \la u, v\ra_W = \int \la u, v\ra\, dP
$$
for all test velocities $u$, where $\delta P$ is the perturbation of $P$ induced by flowing along $u$. By the continuity equation, an infinitesimal flow along $u$ produces perturbation $\delta P = \pd t P= -\nabla \cdot (P\, u)$. By the definition of the functional derivative,
$$
    \df d{d\epsilon}\mathcal F(P + \epsilon\,\delta P)\Big|_{\epsilon=0} \equiv \int \frac{\delta \mathcal F}{\delta P}\,\delta P\, dx = -\int \frac{\delta \mathcal F}{\delta P}\,\nabla\cdot(P\, u)\, dx
$$
Applying the divergence theorem (the boundary term vanishes since $P$ and $u$ decay at infinity):
$$
\begin{aligned}
    &= \cancel{-\int \nabla \cdot \!\left[\frac{\delta \mathcal F}{\delta P}\, P\, u\right] dx} \;+\; \int u\cdot \nabla \left(\frac{\delta \mathcal F}{\delta P}\right)\, dP \\
    &= \int \la u,\, \nabla\frac{\delta \mathcal F}{\delta P}\ra\, dP \implies \mathrm{grad}_W(\mathcal F) := v = \nabla \left(\dfrac{\delta \mathcal F}{\delta P}\right) \,\, \square
\end{aligned}
$$

## Statistical interpretation of drifting

We now connect the drifting field to Wasserstein gradient flow and ask: what functional is being minimized, and what does this have to do with maximum likelihood?

Recall the antisymmetry property $V_{p,q} = -V_{q,p}$. If you have carefully followed through all of the previous examples, this should remind you of the Wasserstein gradient of reverse KL $\eqref{eq:reverse-kl-gradient}$. Let's explore this.

### Gaussian kernel smoothing implements Reverse KL

The reverse KL example above gave us $v(x) = \nabla \log p(x) - \nabla \log q_\theta(x)$ — a score difference matching the drifting field's attraction-minus-repulsion structure. But with empirical Diracs, $\nabla \log p$ is undefined: the **Dirac trap**. Diffusion models resolve this by convolving with noise. Drifting models resolve it differently — via implicit **kernel density estimation** (KDE). Let's see how this works; define the KDE-smoothed distributions:

:::definition[KDE-smoothed distributions]
Given empirical distributions $p, q$ and the Gaussian kernel $k(x, y) = \exp\!\left(-\frac{\|x-y\|^2}{2\tau^2}\right)$, define
$$
    \tilde p(x) = \E_{y^+\sim p}[k(x, y^+)], \quad \tilde q(x) = \E_{y^-\sim q}[k(x, y^-)]
$$
Note that $\tilde p(x) = Z_p(x)$ and $\tilde q(x) = Z_q(x)$. The per-point normalization constants in the drifting field **are** the KDE densities.
:::

Now consider the reverse KL between the smoothed distributions: $\mathcal F(\tilde q) = \mrm{KL}(\tilde q \| \tilde p)$. From the [reverse KL example](#applying-ottos-theorem-to-reverse-kl) above, the Wasserstein gradient descent velocity is:
$$
    v(x) = \nabla_x \log \tilde p(x) - \nabla_x \log \tilde q(x)
$$

We expand each score using the log-derivative trick $\nabla \log f = \nabla f / f$. For the smoothed data score:
$$
\begin{aligned}
    \nabla_x \log \tilde p(x)
    &= \frac{1}{\tilde p(x)}\nabla_x \E_{y^+\sim p}[k(x, y^+)]
    = \frac{1}{Z_p(x)}\E_{y^+\sim p}[\nabla_x k(x, y^+)]
\end{aligned}
$$
For the Gaussian kernel, $\nabla_x k(x, y) = k(x, y)\frac{y - x}{\tau^2}$. Absorbing the $1/\tau^2$ into the learning rate:
$$
    \nabla_x \log \tilde p(x) = \frac{1}{Z_p(x)}\E_{y^+\sim p}\!\left[k(x, y^+)(y^+ - x)\right] = V_p^+(x)
$$
This is precisely the data attraction field. The same calculation with $q$ yields $\nabla_x \log \tilde q(x) = V_q^-(x)$, the model repulsion field. Therefore:

:::theorem[drifting field as Wasserstein gradient]
The canonical drifting field is the negative Wasserstein gradient of the reverse KL between KDE-smoothed distributions:
$$
    V_{p,q} = \nabla \log \tilde p - \nabla \log \tilde q = V_p^+ - V_q^- = -\mrm{grad}_W\, \mrm{KL}(\tilde q \| \tilde p)
$$
Each training step executes the Jacobian pullback $J^T_\theta g$ of the Wasserstein gradient descent $g$ w.r.t. $\mrm{KL}(\tilde q \| \tilde p)$.
:::

:::remark[the Laplace deviation]
The derivation above assumes a Gaussian kernel throughout. The actual paper specifies a Laplace kernel $k(x,y) = \exp(-\|x-y\|/\tau)$, whose true spatial gradient is
$$
    \nabla_x k_{\mrm{Lap}}(x, y) = k(x, y)\frac{y - x}{\tau\|x - y\|}
$$
This is a **unit-direction** pull of constant magnitude $1/\tau$. The paper's drifting field uses **Laplace scalar weights** $k_{\mrm{Lap}}(x, y^+)$ but **Gaussian vector gradients** $(y^+ - x)$ — a deliberate hybrid. This might have been an empirically successful choice because Laplace weights have heavier tails than Gaussian, preventing kernel starvation in high dimensions.
:::

### Implementing forward KL

The reverse KL functional $\mrm{KL}(\tilde q \| \tilde p)$ is **mode-seeking**: particles collapse into well-defined modes and ignore the rest, because the cost of generating samples where data density is zero is infinite. Under the maximum likelihood estimate principle, we typically prefer **forward KL** $\mrm{KL}(\tilde p \| \tilde q)$, which is **mass-covering**: the model is forced to stretch over the entire support of the data.

Apply Otto calculus to $\mathcal F(\tilde q) = \mrm{KL}(\tilde p \| \tilde q)$. The functional derivative w.r.t. the flowing model $\tilde q$ is $\frac{\delta}{\delta \tilde q}\mrm{KL}(\tilde p \| \tilde q) = -\tilde p / \tilde q$ (from the [forward KL example](#applying-ottos-theorem-to-forward-kl) above). The Wasserstein gradient descent velocity is:
$$
    v_{\mrm{MLE}}(x) = \nabla_x \frac{\tilde p(x)}{\tilde q(x)}
    = \frac{\tilde q\,\nabla_x \tilde p - \tilde p\,\nabla_x \tilde q}{\tilde q^2}
    = \frac{\tilde p(x)}{\tilde q(x)}\Big(\nabla_x \log \tilde p - \nabla_x \log \tilde q\Big)
$$
The term in parentheses is exactly the reverse KL velocity — the canonical drifting field $V_{p,q}$:

:::theorem[forward KL via density ratio scaling]
$$
    v_{\mrm{MLE}}(x) = \frac{\tilde p(x)}{\tilde q(x)} \cdot V_{p,q}(x) = \frac{Z_p(x)}{Z_q(x)} \cdot V_{p,q}(x)
$$
Since $Z_p, Z_q$ are already computed to evaluate the drifting field, the density ratio is **free**.
:::

### Proposition: maximum likelihood drifting

Combining the results above, we can state concretely what a maximum-likelihood variant of drifting looks like.

:::theorem[MLE drifting field]
The Wasserstein gradient descent velocity for the forward KL $\mrm{KL}(\tilde p \| \tilde q)$ between KDE-smoothed distributions, using the Gaussian kernel, is
$$
    V_{\mrm{MLE}}(x) = \frac{Z_p(x)}{Z_q(x)} \cdot V_{p,q}^{\mrm{Gauss}}(x)
$$
where $V_{p,q}^{\mrm{Gauss}}$ is the canonical drifting field evaluated with Gaussian kernel weights.
:::

What does this mean in practice? The changes to the existing training protocol are minimal. Here is the current Deng et al. procedure:

**Existing protocol (Deng et al.):**
1. Sample a batch of data points $\{y_i^+\} \sim p$ and generate model outputs $\{x_j\} = \{f_\theta(\epsilon_j)\}$.
2. For each model sample $x_j$, compute the Laplace kernel values $k(x_j, y_i^+)$ and $k(x_j, x_{j'})$ against all data and model samples respectively. Normalize to obtain $Z_p(x_j)$, $Z_q(x_j)$, and evaluate the drifting field $V_{p,q}(x_j) = V_p^+(x_j) - V_q^-(x_j)$.
3. Form drifted targets $\hat x_j = x_j + V_{p,q}(x_j)$.
4. Minimize $\|f_\theta(\epsilon_j) - \mathrm{stopgrad}(\hat x_j)\|^2$.

**MLE modification (two changes):**
1. Same.
2. Same, but replace the **Laplace** kernel with the **Gaussian** kernel.
3. Form drifted targets $\hat x_j = x_j + \frac{Z_p(x_j)}{Z_q(x_j)} \cdot V_{p,q}(x_j)$. That is, scale the drifting field by the density ratio.
4. Same.

|                      | **Deng et al.**                     | **MLE modification**              |
| -------------------- | ----------------------------------- | --------------------------------- |
| Kernel               | Laplace $e^{-\|x-y\|/\tau}$         | Gaussian $e^{-\|x-y\|^2/2\tau^2}$ |
| Drifting field       | $V_{p,q}$                           | $\frac{Z_p}{Z_q} \cdot V_{p,q}$   |
| Antisymmetric?       | Yes                                 | No                                |
| Functional minimized | $\approx$ Reverse KL (mode-seeking) | Forward KL (mass-covering, MLE)   |

Several qualitative differences are worth noting:

**Antisymmetry is lost.** The density ratio breaks the symmetry: $\frac{Z_p}{Z_q} V_{p,q} \neq -\frac{Z_q}{Z_p} V_{q,p}$ in general. But equilibrium is preserved — when $p = q$, we have $Z_p = Z_q$ and $V_{p,q} = 0$, so the field still vanishes.

**Mode-covering vs mode-seeking.** This is the main qualitative shift. Forward KL penalizes the model for assigning low density where data is present: dropped modes are actively hunted. The density ratio $Z_p/Z_q$ acts as a spatially varying learning rate — particles in under-represented regions ($Z_q$ small, $Z_p$ large) take larger steps, while particles in over-crowded regions slow down.
