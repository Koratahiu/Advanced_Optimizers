# Optimizers

This page lists every optimizer available in `adv_optm` and what makes each one distinct.

---

## AdamW_adv

The workhorse of the library. An advanced AdamW with a rich set of optional features. Best suited for general-purpose training where you want adaptive gradients with full control over memory, momentum dynamics, and weight decay behavior.

**Key features:**
- All precision and factorization modes
- Atan2 update rule
- OrthoGrad support
- Spectral normalization
- Kourkoutas-β dynamic scheduling (sunspike-resistant β₂)
- Fisher weight decay (FAdam)
- Variance-normalized momentum (NtM)

* Paper: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

---

## Prodigy_adv

A self-adaptive optimizer that estimates its own learning rate on the fly via the `d`-adaptation mechanism. Ideal when you want to avoid manual learning-rate tuning.

**Key features:**
- Automatic step-size estimation (`d_hat`)
- D-limiter to prevent runaway growth
- Slice-p for memory-efficient adaptation stats
- Prodigy step cap (`prodigy_steps`) to switch to fixed LR after warmup
- Full support for all shared features (Kourkoutas-β, FAdam, OrthoGrad, etc.)

* Paper: [Prodigy: An Expeditiously Adaptive Parameter-Free Learner](https://arxiv.org/abs/2306.06101)

---

## Adopt_adv

ADOPT modifies Adam by initializing the second moment as `v₀ = g₀²` and normalizing the gradient using the *previous* step's second moment, before updating momentum. This decorrelation step often leads to faster convergence in noisy training.

**Key features:**
- Previous-step normalization (decorrelation)
- Atan2 update rule option
- Clip-lambda for gradient clipping
- Kourkoutas-β dynamic scheduling
- All shared precision and memory features

* Original Paper: [ADOPT: Modified Adam Can Converge with Any β<sub>2</sub> with the Optimal Rate](https://arxiv.org/abs/2411.02853)

---

## Lion_adv

Lion updates parameters using the sign of an exponential moving average of gradients. It is simple, memory-efficient, and works well across a range of architectures. This implementation adds SMMF low-rank compression on top.

**Key features:**
- Lion-K update rule with tunable p-norm (`kappa_p`)
- Auto kappa-p (p=2 for Conv2D, p=1 for Linear/Embedding)
- Stochastic sign operator
- OrthoGrad support
- Spectral normalization

* Original Paper: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)

---

## Muon_adv

Muon orthogonalizes the momentum update matrix using Newton-Schulz iteration, making it especially effective for hidden layers of neural networks. It also includes an auxiliary AdamW for bias/norm parameters.

**Key features:**
- Newton-Schulz orthogonalization (configurable iterations and coefficients)
- NorMuon (neuron-wise normalization)
- Low-rank orthogonalization
- Chebyshev-accelerated NS (CANS) with dynamic singular-value bounds
- Approx MARS-M variance reduction
- Auxiliary AdamW for non-matrix parameter groups
- Kourkoutas-β support

* Original Post: [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)

---

## AdaMuon_adv

Combines Muon's geometry-aware orthogonal updates with Adam-style element-wise adaptive scaling. Designed primarily for 2D parameters (linear layers).

**Key features:**
- Sign-stabilized orthogonal update
- Element-wise second-moment estimation on orthogonalized directions
- RMS-aligned rescaling for LR schedule compatibility
- Auto-projection (p=2 for Conv2D, p=1 otherwise)
- All Muon-side features (CANS, NorMuon, MARS-M, etc.)
- Auxiliary AdamW for non-matrix groups

* Paper: [AdaMuon: Adaptive Muon Optimizer](https://arxiv.org/abs/2507.11005)

---

## SignSGD_adv

Sign-based SGD with several advanced variants. Sign operators are inherently memory-efficient and robust to gradient magnitude outliers.

**Key features:**
- Variance/Confidence preconditioning (`snr_cond`, requires `normed_momentum`)
- Geometric weight decay
- Normed momentum (NtM)
- OrthoGrad support
- Spectral normalization
- Stochastic sign operator with $L_\infty$ preconditioning

---

## SinkSGD_adv

Uses Sinkhorn iterative normalization to produce well-conditioned updates. Particularly effective when combined with variance-normalized momentum and preconditioning.

**Key features:**
- Configurable Sinkhorn iterations
- Orthogonal Sinkhorn variant
- Variance/Confidence preconditioning (`snr_cond`)
- Geometric weight decay
- Normed momentum (NtM)
- Nesterov momentum support
- Spectral normalization


* Paper: [Gradient Multi-Normalization for Stateless and Scalable LLM Training](https://arxiv.org/abs/2502.06742)

---

## Choosing an Optimizer

| Use case | Recommended optimizer |
|---|---|
| General purpose. | `AdamW_adv` |
| No manual LR tuning | `Prodigy_adv` |
| Fast decorrelated updates, small batch size, noisy training | `Adopt_adv` |
| Memory-efficient sign-based training | `Lion_adv` or `SignSGD_adv` |
| Hidden layers / matrix parameters | `SinkSGD_adv`, `Muon_adv` or `AdaMuon_adv` |
