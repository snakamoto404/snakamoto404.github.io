---
title: "Optimizer Intuitions"
date: 2026-10-11
---

## Optimizers overview
### Adam as SignSGD

Let $W$ denote weights, $\eta$ learning rate, $\lambda$ weight decay, and $U$ the computed update, the AdamW update reads:
$$
    W \leftarrow (1 - \eta \lambda)\, W - \eta\, U
$$
Given gradients $G$, the update is simply the **<u>coordinate-wise</u> normalized gradient**
$$
    U = \dfrac{\hat G_{\mathrm{ewm}}}{\sqrt{G^2_{\mathrm{ewm}}} + \epsilon}
$$
There're some bells and whistles as to the exponential moving average calculation (also interpretable as momentum), which is used as a bias-variance tradeoff against noisy statistics, and de-biasing.

:::proposition[adam as SignSGD]{#adam-signsgd}
Abstracting out the EWM calculation, Adam updates are ideally $\pm 1$ based on the sign of the gradient. More precisely, the typical entries of Adam updates are constant-sized:
$$
    \|U_{\mathrm{Adam}}\|_{\mathrm{RMS}} = \Theta(1)
$$
:::
This SignSGD interpretation is the intuition behind counterexamples to Adam convergence.

### SGD optimizes $L_2$-proximal linearization

### Muon optimizes spectral-constrained linearization
