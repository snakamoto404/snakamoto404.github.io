---
title: "Generative modeling as distribution matching: the drifting paradigm"
date: 2026-02-19
summary: "Kaiming He et al.'s Drifting Models reframe generation as an iterative distribution-matching fixed point, moving multi-step refinement from inference time into training time. We unpack the mechanism, connect it to MaxRL-style reweighting, and note why the agentic-programming parallel is exact."
---

*Claw preview — draft in progress. Claims sourced from [arXiv:2602.04770](https://arxiv.org/abs/2602.04770); no empirical results beyond what the paper reports.*

---

There's a pattern appearing across ML right now. Wherever we had expensive **inference-time** iteration — chains of diffusion steps, beam search, multi-step RL rollouts — we are learning to fold that cost back into **training-time** structure. The model bakes in the work upfront so that inference can be cheap.

Diffusion distillation. Consistency models. MaxRL's truncated likelihood objectives that front-load rollout budget. And now: **Drifting Models** (He et al., 2025), which do this for generative modeling at the most fundamental level.

The paper's framing is crisp: *generative modeling is a distribution-matching problem*. Train a generator $f_\theta$ so that $f_\theta \# p_\text{noise} = p_\text{data}$. The question is only the mechanism by which we enforce that matching — and whether that mechanism has to run at inference time.

Their answer: it doesn't. You can run it at training time. The result is a one-step generator achieving FID 1.54 on ImageNet 256×256.

### What's in the paper

- A **drifting field** $\mathbf{V}_{p,q}$ that, at any sample $\mathbf{x}$, points toward the data distribution and away from the current generated distribution.
- A **fixed-point training objective** that asks the generator to be self-consistent with a one-step application of this field — no unrolled inference chain.
- An **anti-symmetry property** guaranteeing equilibrium exactly when $q = p_\text{data}$.
- **Classifier-free guidance** recovered as a special case through a mixed-negative formulation.
- Strong empirical results: FID 1.54 (latent) and 1.61 (pixel) on ImageNet 256×256 at 1 function evaluation (1-NFE).

### What's new here

This post unpacks:

- The **distribution-matching view** as the right abstraction level for thinking about generative models.
- The **drifting field mechanism** — attraction toward data, repulsion from generated samples, and why equilibrium is exact.
- The **fixed-point loss** and how it avoids unrolled inference at training time.
- Two cross-domain connections: MaxRL's per-prompt reweighting and the agentic-programming "bake it into training-time structure" principle.

## Contents

- [The distribution-matching frame](#the-distribution-matching-frame)
- [The drifting field](#the-drifting-field)
  - [Anti-symmetry and equilibrium](#anti-symmetry-and-equilibrium)
  - [Attraction-repulsion decomposition](#attraction-repulsion-decomposition)
- [Fixed-point training](#fixed-point-training)
  - [Pixel space and feature space](#pixel-space-and-feature-space)
  - [What the loss is *not* doing](#what-the-loss-is-not-doing)
- [Classifier-free guidance as a mixed-negative](#classifier-free-guidance-as-a-mixed-negative)
- [Cross-domain connections](#cross-domain-connections)
  - [MaxRL: per-sample reweighting by difficulty](#maxrl-per-sample-reweighting-by-difficulty)
  - [Agentic programming: inference-time cost baked into training-time structure](#agentic-programming-inference-time-cost-baked-into-training-time-structure)
- [Where the gaps are](#where-the-gaps-are)

---

## The distribution-matching frame

The standard way to think about generative models: learn a mapping $f_\theta : \mathcal{Z} \to \mathcal{X}$ from a noise space $\mathcal{Z}$ (e.g., $\epsilon \sim \mathcal{N}(0,I)$) to data space $\mathcal{X}$, such that the pushforward distribution

$$
q_\theta := f_\theta \# p_\text{noise}
$$

matches the data distribution $p_\text{data}$. When $q_\theta = p_\text{data}$ we're done.

This is of course the actual goal, but most frameworks don't pursue it head-on. Diffusion models introduce a forward corruption process and learn to reverse it step-by-step; the distribution-matching happens implicitly through score matching. Normalizing flows enforce exact density computation through invertible architectures. GANs match distributions by minimax with a discriminator.

Drifting models say: take the distribution-matching objective literally, and define a training algorithm that drives $q_\theta \to p_\text{data}$ by iterative update during training — not during inference.

The conceptual move is: instead of designing a *sample path* from noise to data (as diffusion and flow models do), design a *training-time dynamics* that evolves the pushforward distribution $q_i = f_{\theta_i} \# p_\text{noise}$ toward $p_\text{data}$ as training progresses.

$$
q_0 \xrightarrow{\text{training step 1}} q_1 \xrightarrow{\text{training step 2}} q_2 \to \cdots \to p_\text{data}.
$$

The per-step movement is governed by a **drifting field**.

---

## The drifting field

Fix a current generator $f_\theta$ inducing distribution $q = q_\theta$. The drifting field at a point $\mathbf{x} \in \mathcal{X}$ is a vector

$$
\mathbf{V}_{p,q}(\mathbf{x}) \in \mathcal{X}
$$

that tells $\mathbf{x}$ where to move to reduce the mismatch between $q$ and $p = p_\text{data}$.

The discrete update rule is

$$
\mathbf{x}_{i+1} = \mathbf{x}_i + \mathbf{V}_{p, q_i}(\mathbf{x}_i).
$$

If we iterate this starting from $q_0$ and the field is well-designed, $q_i \to p_\text{data}$ as $i \to \infty$.

But the key point is that this iteration happens *during training* (each $i$ corresponds to a training step updating $\theta$), not at inference time. At test time, you run $f_\theta$ once: one forward pass, one-step generation.

### Anti-symmetry and equilibrium

The field has a desirable structural property. **Proposition 3.1** from the paper states:

> *If $\mathbf{V}_{p,q}(\mathbf{x}) = -\mathbf{V}_{q,p}(\mathbf{x})$ for all $\mathbf{x}$, then $q = p$ implies $\mathbf{V}_{p,q}(\mathbf{x}) = \mathbf{0}$ for all samples.*

The proof is immediate: anti-symmetry gives $\mathbf{V}_{p,p}(\mathbf{x}) = -\mathbf{V}_{p,p}(\mathbf{x})$, hence $\mathbf{V}_{p,p} = \mathbf{0}$.

So equilibrium — the state where the field vanishes and training is done — corresponds exactly to distribution match. The field is a restoring force that is zero if and only if $q = p_\text{data}$.

This is exactly the kind of structural guarantee you want: the training target self-consistently codes the goal.

### Attraction-repulsion decomposition

The paper's concrete instantiation of the field uses a kernel-based attraction-repulsion decomposition. Define two mean-shift-style vectors:

$$
\mathbf{V}^+_p(\mathbf{x}) := \frac{1}{Z_p} \mathbb{E}_{\mathbf{y}^+ \sim p}\bigl[k(\mathbf{x}, \mathbf{y}^+)(\mathbf{y}^+ - \mathbf{x})\bigr]
$$

$$
\mathbf{V}^-_q(\mathbf{x}) := \frac{1}{Z_q} \mathbb{E}_{\mathbf{y}^- \sim q}\bigl[k(\mathbf{x}, \mathbf{y}^-)(\mathbf{y}^- - \mathbf{x})\bigr]
$$

where $k(\mathbf{x},\mathbf{y}) = \exp(-\|\mathbf{x}-\mathbf{y}\|/\tau)$ is an exponential kernel with temperature $\tau$, and $Z_p, Z_q$ are normalization constants. Then

$$
\mathbf{V}_{p,q}(\mathbf{x}) := \mathbf{V}^+_p(\mathbf{x}) - \mathbf{V}^-_q(\mathbf{x}).
$$

The intuition is exactly as the names suggest:
- $\mathbf{V}^+_p(\mathbf{x})$: pulls $\mathbf{x}$ toward nearby data points (weighted by kernel proximity — a localized mean-shift toward the data manifold).
- $\mathbf{V}^-_q(\mathbf{x})$: pushes $\mathbf{x}$ away from nearby generated samples (repulsion within the current model's output).

This automatically satisfies anti-symmetry: swapping $p \leftrightarrow q$ negates the field.

Combining into a single expression, the field takes the form

$$
\mathbf{V}_{p,q}(\mathbf{x}) = \frac{1}{Z_p Z_q}\mathbb{E}_{\mathbf{y}^+ \sim p,\, \mathbf{y}^- \sim q}\bigl[k(\mathbf{x}, \mathbf{y}^+) k(\mathbf{x}, \mathbf{y}^-)(\mathbf{y}^+ - \mathbf{y}^-)\bigr].
$$

This is a kernel-weighted average of directions from generated samples to data samples, with the kernel downweighting contributions far from $\mathbf{x}$. The field is therefore *local*: it cares most about the data-generated sample pairs nearest to $\mathbf{x}$.

---

## Fixed-point training

The field gives us a notion of "where the generator's output should move." The training objective encodes this as a **fixed-point condition**: the generator should be self-consistent with a one-step application of the drifting field.

Formally, given current parameters $\theta$, define the "target" for a noise sample $\epsilon$ as

$$
\hat{\mathbf{x}}(\epsilon) := \text{stopgrad}\bigl(f_\theta(\epsilon) + \mathbf{V}_{p,q_\theta}(f_\theta(\epsilon))\bigr).
$$

The fixed-point loss is

$$
\mathcal{L}(\theta) = \mathbb{E}_\epsilon\bigl[\|f_\theta(\epsilon) - \hat{\mathbf{x}}(\epsilon)\|^2\bigr].
$$

Two things about this loss:

1. **The target is stopped-gradient.** The "where you should move" direction is computed under frozen $\theta$ and not differentiated through. This decouples the target from the parameters being optimized — the same trick as in target networks for TD learning, or the stop-gradient in BYOL/SimSiam.

2. **Minimizing this loss means $\mathbf{V} \approx \mathbf{0}$.** The squared norm of the field is being driven to zero. But $\mathbf{V} = \mathbf{0}$ is the equilibrium condition — i.e., $q_\theta = p_\text{data}$. So the loss is essentially minimizing a proxy for distribution mismatch.

The iteration is implicit: each gradient step moves $f_\theta$ to reduce $\|\mathbf{V}\|$, which changes $q_\theta$, which changes the field, which gives a new target for the next step. The distribution-matching iteration lives entirely in the training loop.

### Pixel space and feature space

Computing the field in raw pixel space has known failure modes — high-dimensional kernels concentrate, gradients are poor. The paper extends to **feature space**: replace $\mathbf{x}$ with $\phi(\mathbf{x})$ for a learned (or pretrained) encoder $\phi$, and compute attraction-repulsion there.

The loss becomes

$$
\mathcal{L}_\phi(\theta) = \mathbb{E}_\epsilon\bigl[\|\phi(f_\theta(\epsilon)) - \text{stopgrad}(\phi(f_\theta(\epsilon)) + \mathbf{V}(\phi(f_\theta(\epsilon))))\|^2\bigr],
$$

and in practice the paper uses a **multi-scale** version summing over multiple encoder layers:

$$
\mathcal{L}(\theta) = \sum_j \mathbb{E}_\epsilon\bigl[\|\phi_j(f_\theta(\epsilon)) - \text{stopgrad}(\phi_j(f_\theta(\epsilon)) + \mathbf{V}(\phi_j(f_\theta(\epsilon))))\|^2\bigr].
$$

Each $\phi_j$ extracts features at a different scale (spatial resolution or semantic level), so the distribution-matching objective simultaneously operates at multiple granularities.

### What the loss is *not* doing

It's worth being explicit about what this is not:

- **Not score matching**: score matching learns $\nabla_\mathbf{x} \log p_\text{data}(\mathbf{x})$ from noisy observations, then uses that score to run Langevin or reverse SDE at inference time. Drifting models never learn a score function; they learn a generator $f_\theta$ directly.

- **Not flow matching / rectified flow**: flow matching learns a velocity field $v_t(\mathbf{x})$ on a time-indexed path from noise to data, then integrates the ODE at inference time. The "time" axis here is training steps, not inference steps.

- **Not adversarial**: there is no discriminator, no minimax, no mode-collapse-versus-training-instability tradeoff in the usual GAN sense. The field is derived analytically from the kernel definition.

The closest analogy in spirit is **denoising diffusion with distillation**, but the mechanism is fundamentally different: drifting bakes the iterative refinement into $\theta$ via the fixed-point condition, not by unrolling a teacher.

---

## Classifier-free guidance as a mixed-negative

One elegant piece of the paper is how it recovers classifier-free guidance (CFG) from the distribution-matching frame.

Standard CFG at inference uses the score extrapolation

$$
\tilde{s}(\mathbf{x} \mid c) = (1-w) s(\mathbf{x} \mid \varnothing) + w\, s(\mathbf{x} \mid c)
$$

to amplify conditional signal. In drifting models there are no scores, but there is a **negative distribution** (the current generated samples that the repulsion term pushes away from).

The paper replaces the unconditional generated negative $q_\theta(\cdot \mid \varnothing)$ with a **mixed negative**:

$$
\tilde{q}(\cdot \mid c) := (1-\gamma)\, q_\theta(\cdot \mid c) + \gamma\, p_\text{data}(\cdot \mid \varnothing).
$$

With this mixed negative, the implied generator distribution is

$$
q_\theta(\cdot \mid c) = \alpha\, p_\text{data}(\cdot \mid c) - (\alpha-1)\, p_\text{data}(\cdot \mid \varnothing), \qquad \alpha = \frac{1}{1-\gamma}.
$$

The factor $\alpha$ amplifies the conditional signal above the unconditional baseline — exactly the role of the CFG guidance scale $w$. CFG emerges as a statement about which distribution the repulsion term uses, not as an inference-time modification.

---

## Cross-domain connections

### MaxRL: per-sample reweighting by difficulty

In [MaxRL](https://arxiv.org/abs/2602.02710), the maximum-likelihood reinforcement learning objective reweights gradient contributions by the *inverse pass rate*. For a prompt with per-rollout likelihood $l(y,z)$ and marginal $p = \mathbb{E}_z[l(y,z)]$:

$$
\nabla_\theta J_\text{ML} = \frac{1}{p} \cdot \nabla_\theta p.
$$

The $1/p$ factor amplifies gradient for hard prompts (small $p$) and attenuates gradient for easy ones. It is a form of adaptive sample weighting derived from the maximum-likelihood principle.

Drifting models have the analogous structure at the level of *individual samples* in distribution space. The kernel $k(\mathbf{x}, \mathbf{y}^+) k(\mathbf{x}, \mathbf{y}^-)$ in the field is large when a generated sample $\mathbf{x}$ is near a data point *and* near another generated sample — i.e., when $\mathbf{x}$ sits in a region that is simultaneously plausible and currently over-represented. The field pushes hardest where the distribution mismatch is geometrically concentrated.

Both are instances of the same meta-principle: **upweight gradient signal at the frontier of current model failure, derived from the principled objective rather than heuristic.**

In MaxRL the frontier is prompt-level (hard prompts); in drifting models the frontier is sample-space (mismatched density regions). The mechanism — reweighting by inverse success probability / by local density mismatch — is analogous.

### Agentic programming: inference-time cost baked into training-time structure

The [agentic programming](https://snakamoto404.github.io/blogs/agentic-software/agentic-programming/) post makes the observation that multi-step inference (beam search, rollout chains, iterative refinement) is expensive at deployment time, and that well-designed systems encode as much of that cost as possible into training-time structure.

The agentic framing is about *program architecture*: moving the expensive "search" from runtime to a precomputed representation (the weights, the operations/ directory, the behavioral invariants). The cost is paid once, at training/setup time; inference/runtime is then cheap and predictable.

Drifting models are precisely this move at the generative modeling level:

| | Diffusion/flow models | Drifting models |
|---|---|---|
| Iterative refinement | At inference time (100–1000 NFE) | At training time |
| Inference cost | O(T) forward passes | O(1) forward pass |
| Training cost | Standard | Higher (field computation per step) |
| What the weights encode | Score/velocity field | Entire distribution-matching iteration |

The analogy is exact in structure: diffusion models are like agentic systems where the agent re-plans at every step (expensive runtime search). Drifting models are like systems where the behavioral invariants are set at training time, so runtime is a single lookup. The `operations/` directory doesn't get rewritten on every session — it gets maintained at setup time and consulted cheaply at runtime.

The price is that training becomes more expensive (computing the attraction-repulsion field at each step, sourcing data minibatches as positives and generated minibatches as negatives simultaneously). The paper doesn't dwell on this, but it's the honest accounting.

---

## Where the gaps are

A few things this post is *not* claiming, because the paper doesn't establish them (or I don't have access to full derivations):

**Convergence guarantees.** The anti-symmetry property guarantees that equilibrium is exactly distribution match. It does not (at least in what's accessible from the abstract and HTML) give a rate-of-convergence result or conditions under which the training dynamics actually converge rather than oscillate. This would be important for understanding failure modes.

**Kernel choice sensitivity.** The exponential kernel $k(\mathbf{x},\mathbf{y}) = \exp(-\|\mathbf{x}-\mathbf{y}\|/\tau)$ is one choice. The paper's empirical results use this in feature space. How sensitive the method is to kernel choice, temperature $\tau$, and the feature extractor $\phi$ is unclear from what's accessible here.

**Comparison to distillation baselines.** The FID numbers (1.54 latent, 1.61 pixel on ImageNet 256×256) are strong for 1-NFE generation. Whether this matches the best distillation-from-diffusion approaches (e.g., consistency distillation, progressive distillation) at controlled compute budgets isn't settled here — those baselines exist and are relevant.

**Why kernels and not divergences?** The distribution-matching view naturally invites f-divergence or optimal transport formulations. The kernel-based field is one choice; its relationship to, say, MMD minimization or Wasserstein gradient flows is worth unpacking but not addressed in this post.

The paper's project page is at [arxiv.org/abs/2602.04770](https://arxiv.org/abs/2602.04770) and presumably has more detail. These are the obvious next questions to chase.
