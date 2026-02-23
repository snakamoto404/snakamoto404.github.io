---
title: "OT for generative modeling 3 — drifting models"
date: 2026-02-23
summary: "Part 3 of a series on optimal transport in generative models. We formulate Kaiming He et al.'s Drifting Models, interpret the antisymmetric drifting field as Wasserstein gradient flow on a kernelized discrepancy, and draw the connection to maximum likelihood estimation."
---

Diffusion and flow matching both split the generative problem into two phases: at *training* time, learn a vector field; at *inference* time, integrate an ODE or SDE through that field to produce a sample. The integration is expensive — the reason we care about "number of function evaluations" (NFE) at all — and a great deal of recent work has gone into compressing it: distillation, consistency models, progressive reduction.

But there is a more radical question: *what if the integration happens during training instead?* If the SGD optimizer is already iteratively updating the generator, and each update implicitly moves the pushforward distribution $q = f_{\#} p_\epsilon$ through the Wasserstein manifold, then maybe we can design a training objective that makes that movement *purposeful* — steering $q$ toward $p_{\text{data}}$ in a geometrically principled way — and skip the ODE solver at inference entirely.

This is precisely the idea behind **Drifting Models** ([He et al., 2026](https://arxiv.org/abs/2602.04770)). The construction is strikingly clean: define a "drifting field" $\mathbf{V}_{p,q}(\mathbf{x})$ that tells each generated sample which direction to move, require antisymmetry ($\mathbf{V}_{p,q} = -\mathbf{V}_{q,p}$) so that the field vanishes at equilibrium ($p = q$), and train the generator to chase its own drifted targets. The result is a single-pass, 1-NFE generator that achieves state-of-the-art FID on ImageNet 256×256.

The paper presents this mechanistically, and it works. But equipped with the Wasserstein machinery we built in [Parts 0](/blogs/machine-learning/ot-generative-0-static/)–[2](/blogs/machine-learning/ot-generative-2-wasserstein-gradients/), we can say something much sharper. The antisymmetric drifting field is the Wasserstein gradient of a distribution discrepancy; the training dynamics execute gradient descent on the manifold $\mathcal{W}_2$; and the $L^2$ regression loss against drifted targets — a simple MSE in PyTorch — is, through the Wasserstein bridge, a gradient step on a proper statistical divergence. The whole thing admits a precise (if indirect) connection to maximum likelihood.

## Contents

- [Formulation](#formulation)
- [Interpretation as Wasserstein gradient flow](#interpretation-as-wasserstein-gradient-flow)
- [Drifting as MLE](#drifting-as-mle)

## Formulation

*Coming soon.*

## Interpretation as Wasserstein gradient flow

*Coming soon.*

## Drifting as MLE

*Coming soon.*
