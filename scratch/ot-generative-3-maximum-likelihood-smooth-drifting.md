---
title: "OT for generative modeling 3 — Maximum likelihood smooth drifting"
date: 2026-02-25
summary: "Research plan for a generalization of the recent drifting paper."
---

Previous parts of the blog post have identified the **statistical principles** behind drifting, which executes the pullback of the Wasserstein gradient in parameter space.

Here, we propose a modeling variant where main changes consist of:

- Explicitly walking the Wasserstein gradient of the **maximum likelihood objective**, instead of the mode-seeking, reverse-KL objective.
- Side-stepping the difficulty of disjoint high-dimensional kernel-density supports by training to a joint KL objective with latent
- Using a parametric critic to identify the likelihood ratio  [maybe elaborate here].

This document begins with the the potential weaknesses of the existing modeling method, proposals for concrete changes with corresponding experimental ablations, and finally contextualize the resulting model, which we call **maximum likelihood smooth drifting**, with existing generative modeling paradigms such as GAN and flow matching.

## Contents

- [Existing challenges]()
- [Proposed change 1: Maximum likelihood training]()
- [Proposed change 2: ...(proper name for "smoothing")]
- [Proposed change 3: parametric likelihood ratio estimation using a critic]()
- [Comparison with existing models]()
    - [Diffusion]
    - [GANs]

## Existing challenges

Existing drifting (with Gaussian kernel) implements Wasserstein gradient descent of reverse KL using kernel density likelihood ratio estimates. There're two catches here:

- Reverse KL is mode-seeking.
- KDE are starved in high dimensions.

We can do experiments to verify each: The proposed experiments below should ideally be extremly easy-to-do (ideally minimal wind-tunnels). They should result in sharp (meaning to-the-point) and salient visuals (graphs / plots) as well as compelling quantitative metrics.

- [Experiment 1: can we have a minimal, sharp experiments to do here?]
    - Purpose: highlight the difference between the existing objective and our proposed, proper forward-KL objective.
- [Experiment 2: demonstrate that KDE are starved in high dimensions; consider jointly designing with smoothing]


## Change 1: Maximum likelihood training

Recap the theory here very succinctly, recap the exact Wasserstein gradients for reverse v. forward KLs.

We can rewrite the maximum likelihood Wasserstein gradient as
...

:::remark[knob]
Clipping the likelihood ratio
:::

## Change 2: Smoothing (good name)

For both reverse and forward KL, when p_data and q_theta have non-overlapping supports, KL can become ill-behaved [elaborate]. The dominant diffusion / flow-matching paradigm side-steps this issue by explicitly convolving the data with noise to achieve overlapping supports [..elaborate..], but [..at the cost of having to do multiple steps of sampling; make sure to extremely correctly / sharply convey the right things here..]. Emphasize that this is a mathematical difficulty that's independent of our choice of estimators. This appears to be empirically observed in Deng et al's paper, where high-dimensional generation is challenging without pretrained feature encoders.

The same principle applies here. We propose to split training into two stages:

### [proper name, maybe pretraining?]

stage (1) to optimize $\Exp_{\epsilon \sim \mathrm{Unif}(0, 1)} \mathrm{KL}(p_{\mathrm{data}, \epsilon} \|  q_{\mathrm{data}, \epsilon})$ where $\epsilon$ is defined as the $\epsilon$-mixture of i.i.d. Gaussian (with coordinate-wise data empirical covariance and mean) and the source distribution. A major strength of this is that (1) disjoint support contribute measure 0 in the expectation, (2) principled joint-KL minimization interpretation [rewrite the expectation as KL over joint distributions].
- Derive the concrete training protocol for this, using general likelihood ratio estimate as a subroutine first, then specialize to Gaussian-kernel KDE later.

Propose experiments.

Knobs: no knobs to turn here, maybe at most the noise limit being standard Gaussian or Gaussian with empirical means.

### [proper name, maybe finetuning? but I'm slightly against the pretrian-finetune naming because they're not totally accurate]

Stage (2) optimizes KL on a noise schedule much more concentrated about $0$. What would be a canonical distribution? (exponential?). The idea here is that pretraining (replace with name) provided sufficient support overlap such that

Knobs: maybe at most the parameters of the tight noise schedule. Note that

Propose experiments & things to track here.

## Change 3 (optional): parametric likelihood ratio estimate using a critic

Tldr: use a critic to explicitly help get the likelihood ratio, as well as the corresponding score.

[Derive and write out the critic objective]

This is a decoupled change.

Knobs:
- Nonstationarity 1: Weight EWM for the critic
- Nonstationarity 2: Replay buffers
- Smoothness 1: Applying spectral norm to ensure the smoothness of the critic
- Smoothness 2 (remind you of what was in MLE): clipping.

Propose experiments & things to track here.

## Comparison with existing models



### Diffusion

Diffusion addresses by learning the score on the joint distribution [....]. While the initial data and model distributions might be disjoint, the joint distribution has disjoint support of measure 0. We try to take the same idea from there.

### GANs

GANs rely on variational definitions of statistical divergences. Minimax Training is highly coupled both ways. In a sense, the critic in our method is much less adversarial, since its purpose is to **guide** the generator and offer the best estimate of the likelihood ratio. Critic data non-stationarity is still a problem, which we try to address by RL-inspired methods.

# *** Instructions for planning agents ***
... elaborate here as well, including:
1. Experimental design: I would like you to propose experiments which are high-leverage, easy-to-do (where they can be), and crystal clear. For example, use wind tunnels where we can, and make sure to do experiments which sharply prove the point we wish to demonstrate, together with proposals for clear, intuitive metrics to monitor; carefully define the knobs & ablations of the experiments though, what we expect, and what questions they're supposed to answer / points to prove. We also need integration experiments (benchmark-runs) at the end.
2. Exposition: be sharp and precise about all the mathematical statements with proper ties to context.