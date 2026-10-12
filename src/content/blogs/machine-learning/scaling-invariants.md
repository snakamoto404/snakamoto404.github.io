---
title: "Scaling invariants: width-μP"
date: 2026-10-11
---

Consider a linear layer $W:\mathbb R^n\to \mathbb R^p$ with input $x$ and output $z$. As we continuously scale up $n\mapsto kn$ and potentially $p\mapsto kp$, we want the layer to have **desirable scaling properties / symmetries** with respect to width.

Read [Everett et al.](https://arxiv.org/abs/2407.05872) and [assignment 2 + optimization-1 lecture of CS312](https://deep-learning-alchemy.github.io/) for details.

:::proposition[abc parameterization]{#abc-parameterization}
Define the **forward multiplier**, **initialization scale**, and **learning-rate** degrees of freedom.
$$
    z = n^{-a} Wx, \quad W_{jk} \sim \mathcal N(0, \sigma^2 n^{-2b}), \quad \eta_W = \eta \, n^{-c}
$$
:::

:::proposition[μP axioms]{#MuP-axioms}
Model (width) scaling should have the following behaviors:
1. **Stability**: activations and logits are bounded at initialization and throughout training.
$$
    \|z_0\|_{\mathrm{rms}} = \Theta(1), \quad \|z_t - z_0\|_{\mathrm{rms}} = O(1)
$$
For the final readout layer, we only require $\|z_0\|=O(1)$.

2. **Feature learning** (aka maximality) imposes the stronger constraint that, **for each layer**
$$
    \|z_t - z_0\|_{\mathrm{rms}} = n^{-a} \|\Delta Wx + W\Delta x + \Delta W \Delta x\|_{\mathrm{rms}} = \Theta(1)
$$
:::

## Activation behavior

Consider $x$ with typical size $1$ and suppress $t=0$ subscript to reduce clutter, by $z = n^{-a} Wx$:
$$
\begin{aligned}
    \mathbb E[z_j^2]
    &= n^{-2a} \mathbb E[(W_{jk}x_k)^2] = n^{-2a} \mathbb E[W_{jk}x_k W_{jl}x_l]
\end{aligned}
$$
By **random independent initialization**, cross-terms cancel so $k = l$, leaving
$$
    \mathbb E[z_j^2] = n^{-2a} \mathbb E[W_{jk}^2] x_k^2 = \sigma^2 n^{-2(a+b)} \|x\|^2
$$
Note that $\|x\|^2 = \Theta(n)$ for intermediate and readout layers (but not for the initial layer!!)

:::proposition[activation stability]{#activation-stability}
Given $\|x\|_{\mathrm{rms}} = \Theta(1)$, the typical size of the activation is
$$\|z\|_{\mathrm{rms}}^2 = \sigma^2 n^{1-2(a+b)}$$. Then
$$
    \|z\|_{\mathrm{rms}} = \Theta(1), \quad \|x\|_{\mathrm{rms}} = \Theta(1) \implies a + b = 1/2
$$
Replace with $a+b\geq1/2$ for $\|z\|_{\mathrm{rms}} = O(1)$.
:::
The [Kaiming initialization](https://arxiv.org/abs/1502.01852) chooses $a=0, b=1/2$ and $\sigma^2=2$ to compensate for ReLU.

|                         | Input layer           | Hidden layers          | Readout layer          |
| ----------------------- | --------------------- | ---------------------- | ---------------------- |
| $\Theta(n)$ scaling     | fan-out               | fan-in and fan-out     | fan-in                 |
| $\|x\|^2$ (divide by $n$ for $\|x\|_{\mathrm{rms}}^2$)     | $\Theta(1)$           | $\Theta(n)$            | $\Theta(n)$            |
| $\mathbb E[z_j^2]$      | $\Theta(n^{-2(a+b)})$ | $\Theta(n^{1-2(a+b)})$ | $\Theta(n^{1-2(a+b)})$ |
| Required $\|z\|_{\mathrm{rms}}$ | $\Theta(1)$       | $\Theta(1)$            | $O(1)$                 |
| Stability at init  constraint     | $a + b = 0$           | $a + b = 1/2$          | $a + b \geq 1/2$       |

### Gradient behavior

For well-behaved loss functions, the first cotangent has
$\|\delta'\|_{\mathrm{rms}} = \Theta(1)$.

Write $\delta' = dL/dz$ and $\delta := dL/dx$, then
$$
    z_j = n^{-a} W_{jk} x_k \implies \delta_j = n^{-a} \delta'_k W_{kj}
$$
in matrix shorthand, this is just the standard matrix-vector chain rule $\delta = n^{-a} W^T \delta'$. Working out the norms
$$
    \mathbb E[\delta_j^2] = n^{-2a} \mathbb E[(W_{kj} \delta'_k)^2] = n^{-2a} \mathbb E[W_{kj} \delta'_k W_{lj} \delta'_l]
$$
Again, invoking **weight independence at init** yields $k=l$, and further invoking **cotangent-weight independence** $\delta'_k \perp W_{kj}$ (this is not necessarily true!) yields
$$
    \mathbb E[\delta_j^2] = \sigma^2\, p\, n^{-2(a+b)} \|\delta'\|_{\mathrm{rms}}^2
$$
Similarly for weight gradients, $\partial_{W_{jk}} L = n^{-a} x_k \delta'_j$.

:::proposition[gradient inductive relations]
High-level takeaway: each cotangent backprop introduces a factor of $1/2 - (a + b)$, and the final weight a factor of $-a$.

For cotangents
$
    \|\delta\|_{\mathrm{rms}} = \sigma \sqrt p \, n^{-(a+b)} \|\delta'\|_{\mathrm{rms}}
$.

For weights $\|\partial_{W_{jk}} L\|_{\mathrm{rms}} = n^{-a} \|x\|_{\mathrm{rms}} \|\delta'\|_{\mathrm{rms}}$.

Note that this all assumes cotangent-weight independence.
:::

Recall some optimizer basics. For Muon, just rotate to SVD coordinates.

:::proposition[post-abc scaling weight dynamics]
- for Adam, $\|\Delta W\|_{\mathrm{rms}} = \Theta(n^{-c})$.
- For Muon, $\|\Delta W\|_{\mathrm{rms}} = \Theta\left(\dfrac{n^{-c}}{\sqrt{\max(n, p)}}\right)$.

Write $\|\Delta W\|_{\mathrm{rms}} = \Theta(n^{-c-\gamma})$, where $\gamma=0$ for Adam and $\gamma \approx 1/2$ for Muon.
:::

### Activation dynamics

Now, we can work out the change in representations. Decompose activation change into
$$
    \Delta z = n^{-a} (W \Delta x + \Delta W x + \Delta W \Delta x)
$$
:::definition[alignment exponents]
These are alignment exponents which depend empirically on the problem.
- **dW alignment** exponent: $\|\Delta W x\|_{\mathrm{rms}} \sim n^{\alpha} \|\Delta W\|_{\mathrm{rms}} \|x\|_{\mathrm{rms}}$.
- **dx alignment** exponent: $\|W \Delta x\|_{\mathrm{rms}} \sim n^{\omega} \|W\|_{\mathrm{rms}} \|\Delta x\|_{\mathrm{rms}}$.
- **dWdx alignment** exponent: $\|\Delta W \Delta x\|_{\mathrm{rms}} \sim n^{u} \|\Delta W\|_{\mathrm{rms}} \|\Delta x\|_{\mathrm{rms}}$.
:::

:::proposition[alignment exponent intuitions]
For general $A:\mathbb R^n\to \mathbb R^p$ and $x\in \mathbb R^n$, write RMS to $L_2$ norms.
$$
    \|Av\|_{\mathrm{rms}} = \dfrac{\|Av\|}{\sqrt{p}}, \quad \|A\|_{\mathrm{rms}} = \dfrac{\|A\|}{\sqrt{np}}, \quad \|x\|_{\mathrm{rms}} = \dfrac{\|x\|}{\sqrt{n}}
$$
Expand the alignment exponent in $L_2$ norms:
$$
    \alpha(A, v) = \log_n \dfrac{\|Av\|_{\mathrm{rms}}}{\|A\|_{\mathrm{rms}}\|v\|_{\mathrm{rms}}} = 1 + \log_n g, \quad g := \dfrac{\|Av\|}{\|A\|\|v\|}
$$
Interpreting, $g$ denotes what portion of $A$'s $L_2$ weight can a unit vector capture.
:::

:::corollary[alignment by rank]
Consider general SVD $A = \sigma_j |u_j\rangle \langle v_j|$.
W.l.o.g, let $v = \lambda_j|v_j\rangle$ unit norm, then
$$
    g = \sqrt{\sum (\lambda_j \sigma_j)^2} / \sqrt{\sum \sigma_j^2}
$$
Consider special cases. Tl; dr is that under regular assumptions, the only important quantity is the **asymptotic concentration of $v$ w.r.t. the row space of $A$**.
:::

<details>
<summary> worked examples </summary>

:::remark[examples]
1. When $A$ is constant-rank, $\alpha \to 1$ as long as $\Theta(1)$ of $v$ is in $A$'s row-space.
2. When $A$ is full effective rank, $\alpha \to 1/2$ regardless.
- $A = |u_1\rangle \langle v_1|$ **single-rank**.
    - $v=|v_1\rangle$ *in-rank* $\implies \alpha=1$.
    - $v$ is *full-spectrum deconcentrated* $\langle v_1|v\rangle = 1/\sqrt n \implies \alpha = 1/2$
    - $v$ out of rank $\langle v_1|v\rangle \to 0 \implies \alpha \to -\infty$.
- $A = |u_{j=1\dots q}\rangle \langle v_j|$ **constant-rank $q$ flat-spectrum**.
    - $v = \tfrac{1}{\sqrt q} |v_{j=1\dots q} \rangle$ *in-rank deconcentrated* implies $g = 1/\sqrt q \implies \alpha \to 1$ in increasing $n$.
    - $v = |v_1\rangle$ *concentrated in-rank* implies $g=1/\sqrt q \implies \alpha\to 1$.
    - $v = \tfrac{1}{\sqrt n} |v_{j=1\dots n}\rangle$ *full-space deconcentrated*. $g = 1 / \sqrt n \implies \alpha \to 1/2$.
- $A = |u_j\rangle \langle v_j|$ **full-rank flat-spectrum**. Special case of the above
    - Both deconcentrated and concentrated $g=1/\sqrt n\implies \alpha = 1/2$.
:::
</details>

### Deriving width-μP

We have all the ingredients to derive $\mu P$. Let's begin with a gauge observation

:::proposition[abc-gauge]
For Adam and Muon, $(a, b, c)\sim (a-\theta, b+\theta, c+\theta)$.
More generally,
$$
    U(\lambda G) = \lambda^k U(G)\implies (a, b, c) \sim \left(a-\theta, b+\theta, c+(1+k)\theta\right)
$$
:::

<details>
<summary>Detailed analysis</summary>

1. From the forward's perspective, downscaling initialization by $b\mapsto b+\theta$ and upscaling forward multiplier $a\mapsto a-\theta$ is covariant.
2. From the backward's perspective, the downscaled parameter's gradients were upscaled by $n^\theta$, while we want a corresponding $n^{-\theta}$ to have covariant post-update parameters in this scaled setting.

If updates are invariant to gradient scale, we just need  to upscale $c\mapsto c+\theta$. If $U\propto G$ as in the SGD case, then $c\mapsto c+2\theta$ for covariance.

</details>

Without loss of generality, we only need to consider three types of layers: embedding $W_1$, hidden $W_2$, and readout $W_3$. With one gauge relation for each layer type, there are $6$ remaining degrees of freedom. The key decomposition into init and change throughout training:
$$
    \|\Delta z\|_{\mathrm{rms}} \lesssim n^{-a} \left(
    \underbrace{n^{\omega - b}}_{W \Delta x}
    + \underbrace{n^{\alpha - c - \gamma}}_{\Delta W x}
    + \underbrace{n^{u - c - \gamma_2}}_{\Delta W \Delta x}\right)
$$
1. **Activation stability** (3 constraints): stability at init specifies $a_1 + b_1 = 0, a_2 + b_2 = 1/2$, and *logit stability throughout training + maximality of feature-change passthrough* specifies $a_3 + b_3 = 1$. This leaves $3$ dof.

<details>
<summary>Elaborating on the readout constraint</summary>

Stability at init gives $a_3 + b_3 \geq 1/2$. However, we also want

$$
    \|W_3 \Delta x\| = O(n^{\omega_3 - a_3 - b_3}) = \Theta(1)
$$

Note that $\Delta x$ was computed from $W_3$, and $W_3$ is crucially constant-rank (num-output) compared to $n$, so each output-basis of $W_3$ has $\Theta(1)$ coherence with $\Delta x$, resulting in $\omega_3 \to 1 \implies a_3 + b_3 \geq 1$.

We impose equality to ensure that feature-change-passthrough is a nontrivial component of logit change.
</details>

2. **Embedding maximality** ($1$ constraint): for the embedding, $\Delta x = 0$ and crucially note that the fan-in is not a function of width (the alignment-exponent $\alpha_1 = 0$), then $a_1 + c_1 = \gamma$ from
$$
\|\Delta z_1\|_{\mathrm{rms}} = n^{-a_1} \|\Delta W_1 x\|_{\mathrm{rms}} \propto n^{-a_1} \|\Delta W_1\|_{\mathrm{rms}} \|x\|_{\mathrm{rms}} = n^{-a_1 - c_1 - \gamma}
$$

3. **Hidden-layer maximality** ($1$ constraint): bounding each term of $\Delta z_2$, only $\Delta W_2 x$ has bite and yields
$$
    a_2 + c_2 = \alpha_2 - \gamma_2
$$
<details>
<summary>elaboration</summary>

- $W_2\Delta x$ is bounded by the stability constraint and Gaussian-init flat spectrum: a square Gaussian matrix has $\|W_2\|_{\mathrm{op}} \approx 2\sqrt n\, \|W_2\|_{\mathrm{rms}} \implies \omega_2 \leq 1/2$ (I trusted AI on this). With $a_2 + b_2 = 1/2$ from stability, this term is non-dominant.
- $\Delta W_2 x$: maximality implies $a_2 + c_2 = \alpha_2 - \gamma_2$. Treat $\alpha_2$ and $\gamma_2$ as constants.
- $\Delta W_2 \Delta x$: using maximality, $u_2 - a_2 - c_2 - \gamma_2 = u_2 - \alpha_2$. No additional constraint here.

</details>

4. **Readout-layer maximality** ($1$ constraint): $a_3 + c_3 = \alpha_3 - \gamma_3$.

<details>
<summary> details </summary>

- $W_3 \Delta x$: discussed above with activation stability.
- $\Delta W_3 x$: maximality requires $a_3 + c_3 = \alpha_3 - \gamma_3$.
- $\Delta W_3 \Delta x$: use maximality constraint above to obtain the constant constraint $u_3 - \alpha_3$. No constraint here.

</details>

The analysis above assumes no catastrophic cancellation of leading terms.

### Concrete prescriptions

Let's pick the $a = 0$ gauge with no forward multipliers. This gives the largest gradients and is most robust to $\epsilon$-choices; this is also the most natural gauge for constraints, making clear that **stability implies initialization** and **maximality implies lr** scaling. This gauge couples width to weight-decay, but let's use weight-norm in 2026, alas. From now on, let's assume Muon / AdamW.
$$
    \tilde a = 0, \quad \tilde b = b + a, \quad \tilde c = c + a
$$
Let's further simplify $\alpha := \alpha_2 = \alpha_3 = \{\tfrac 1 2, 1\}$ corresponding to no / full alignment, respectively.

|          | Embedding |  | Hidden |  | Readout |  |
| -------- | --------- | --- | ------ | --- | ------- | --- |
|          | $\tilde b_1$ | $\tilde c_1$ | $\tilde b_2$ | $\tilde c_2$ | $\tilde b_3$ | $\tilde c_3$ |
| **MuP** | $0$ | $-\gamma$ | $\tfrac12$ | $\alpha - \gamma$ | $1$ | $\alpha - \gamma$ |
| **SP**  | $0$ | $\alpha - \gamma + \tfrac12$ | $\tfrac12$ | $\alpha - \gamma + \tfrac12$ | $\tfrac12$ | $\alpha - \gamma + \tfrac12$ |
| **NTK** | $0$ | $\alpha - \gamma$ | $\tfrac12$ | $\alpha - \gamma + \tfrac12$ | $\tfrac12$ | $\alpha - \gamma + \tfrac12$ |
| **SP\*, no-alignment** | $0$ | $\tfrac12$ | $\tfrac12$ | $\tfrac12$ | $\tfrac12$ | $\tfrac12$ |
| **MuP\*, no-alignment** | $0$ | $0$ | $\tfrac12$ | $0$ | $1$ | $\tfrac12$ |
| **SP\*, full-alignment** | $0$ | $1$ | $\tfrac12$ | $1$ | $\tfrac12$ | $1$ |
| **MuP\*, full-alignment** | $0$ | $0$ | $\tfrac12$ | $\tfrac12$ | $1$ | $1$ |

**MuP\***: AdamW ($\gamma = 0$) on embedding and readout, Muon ($\gamma = \tfrac12$) on hidden.

Some remarks in order:
- The defining characteristic of SP is consistent lr and initialization rules across components.
- NTK minimally makes the embedding more learnable compared to SP.
