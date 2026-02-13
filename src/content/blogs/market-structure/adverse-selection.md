---
title: Adverse Selection
date: 2026-02-13
summary: Placeholder post on adverse selection in market structure.
---

**Placeholder content:** This draft is intentionally temporary and will be replaced with full research-backed analysis.

Adverse selection appears when one side of a trade has better information than the other, often worsening execution quality.

```ts
const expectedValue = (winProb: number, payoff: number, loss: number) =>
  winProb * payoff - (1 - winProb) * loss;
```

Inline math placeholder: $\text{edge} = p\cdot q - c$.

$$
\mathbb{E}[\Pi] = p\cdot \Delta - (1-p)\cdot \lambda
$$
