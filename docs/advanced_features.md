# Advanced Features

This page covers advanced features that go beyond the standard optimizer settings.

---

## OrthoGrad (`orthogonal_gradient`)

OrthoGrad orthogonalizes the gradient before applying the update, improving training stability and generalization by removing redundant directional components.

| Value | Description |
|---|---|
| `'disabled'` | Off. |
| `'flattened'` | Standard vectorized OrthoGrad. Projects each flattened parameter vector onto the orthogonal complement of the gradient. |
| `'iterative'` | Matrix-wise rank-2 OrthoGrad. Operates on the 2D parameter matrix directly, useful for layer weights. |

* Available on all optimizers.
* Reference Paper: [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697)


**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    orthogonal_gradient='flattened',
)
```

---

## Spectral Normalization (`spectral_normalization`)

Applies explicit spectral normalization using power iteration to the update matrix. This achieves width/rank invariant updates, making learning rates more transferable across architectures.

| Parameter | Default | Description |
|---|---|---|
| `spectral_normalization` | `False` | Enables spectral normalization via power iteration. |

* Available on all optimizers.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    spectral_normalization=True,
)
```

---

## Stochastic Sign (`stochastic_sign`)

Replaces the deterministic sign operator with a stochastic variant preconditioned by the $L_\infty$ norm. This adds controlled randomness that can help escape sharp minima.

| Parameter | Default | Description |
|---|---|---|
| `stochastic_sign` | `False` | Enables the stochastic sign operator. |

Available on: `SignSGD_adv`, `Lion_adv`.

**Example:**
```python
optimizer = SignSGD_adv(
    model.parameters(),
    stochastic_sign=True,
)
```

* EXPERIMENTAL

---

## SNR Conditioning (`snr_cond`)

Variance/Confidence preconditioning that scales updates by the signal-to-noise ratio of the gradient.

| Parameter | Default | Description |
|---|---|---|
| `snr_cond` | `False` | Enables SNR conditioning. |

* EXPERIMENTAL
* Available on: `SignSGD_adv`, `SinkSGD_adv`.
* Requires `normed_momentum=True`
* Read more: [AASS](https://koratahiu.github.io/aass/) & [sink-v](https://koratahiu.github.io/sink-v/)

**Example:**
```python
optimizer = SignSGD_adv(
    model.parameters(),
    momentum=0.9,
    normed_momentum=True,
    snr_cond=True,
)
```

---

## Newton-Schulz Iteration (Muon variants)

Controls the orthogonalization process in Muon and AdaMuon optimizers.

| Parameter | Default | Description |
|---|---|---|
| `ns_steps` | 5 | Number of Newton-Schulz iterations. |
| `ns_eps` | 1e-7 | Epsilon for normalization stability. `None` uses scale-invariant rule. |
| `ns_coeffs` | `(3.4445, -4.7750, 2.0315)` | (a, b, c) coefficients for the quintic polynomial. |

**Example:**
```python
optimizer = Muon_adv(
    model.parameters(),
    ns_steps=5,
    ns_eps=None,  # scale-invariant
)
```

---

## Chebyshev-Accelerated NS (`accelerated_ns`)

Dynamically calculates optimal 3rd-order polynomial coefficients for the Newton-Schulz iteration using Chebyshev polynomials. Faster convergence than fixed coefficients.

| Parameter | Default | Description |
|---|---|---|
| `accelerated_ns` | `False` | Enables CANS. |
| `cns_a_bound` | `None` | Initial lower bound for singular values. `None` uses scale-invariant rule. |

* Available on: `Muon_adv`, `AdaMuon_adv`.

---

## NorMuon (`normuon_variant`)

Adds neuron-wise normalization to Muon, computing per-neuron second moments and normalizing accordingly.

| Parameter | Default | Description |
|---|---|---|
| `normuon_variant` | `False` | Enables NorMuon. |
| `beta2_normuon` | 0.95 | Decay rate for NorMuon second moments. |
| `normuon_eps` | 1e-8 | Stability epsilon for NorMuon. |

Available on: `Muon_adv`.

---

## Low-Rank Orthogonalization (`low_rank_ortho`)

Projects the update to a lower rank before orthogonalization, reducing computational cost for large matrices.

| Parameter | Default | Description |
|---|---|---|
| `low_rank_ortho` | `False` | Enables low-rank orthogonalization. |
| `ortho_rank` | 128 | Rank for the projection. |

Available on: `Muon_adv`, `AdaMuon_adv`.

---

## Approx MARS-M (`approx_mars`)

Variance reduction based on the MARS-M paper ("MARS-M: When Variance Reduction Meets Matrices"). Reduces gradient noise in Muon-style updates.

| Parameter | Default | Description |
|---|---|---|
| `approx_mars` | `False` | Enables MARS-M variance reduction. |
| `mars_gamma` | 0.025 | Scaling coefficient for gradient correction. |

Available on: `Muon_adv`, `AdaMuon_adv`.

---

## RMS Rescaling (`rms_rescaling`)

Uses Root-Mean-Square for the final update vector, aligning update magnitudes with Adam-style optimizers. This allows reuse of existing LR schedules.

| Parameter | Default | Description |
|---|---|---|
| `rms_rescaling` | `True` | Enables RMS-aligned rescaling. |

* Available on: `Muon_adv`, `AdaMuon_adv`.
* Not compatible with `spectral_normalization`

---

## Sinkhorn Settings (SinkSGD)

| Parameter | Default | Description |
|---|---|---|
| `sinkhorn_iterations` | 5 | Number of Sinkhorn iterations. |
| `orthogonal_sinkhorn` | `False` | Uses orthogonal Sinkhorn variant. This reaches a state where all rows and cols are orthogonal to the weights (rank-2 Orthogonal projection) |

---

## Prodigy-Specific Settings

| Parameter | Default | Description |
|---|---|---|
| `d0` | 1e-6 | Initial D estimate. Rarely needs changing. |
| `d_coef` | 1.0 | Coefficient in the d-estimate expression. Tune to adjust adaptation speed. |
| `growth_rate` | inf | Max multiplicative rate for D estimate growth. ~1.02 gives warmup effect. |
| `slice_p` | 11 | Calculate adaptation stats on every pth entry to save memory. |
| `prodigy_steps` | 0 | Disable adaptation after this many steps; release state memory. |
| `d_limiter` | `False` | Clamp d_hat to prevent volatile step-size increases. |
| `fsdp_in_use` | `False` | Set True when using FSDP sharded parameters. |

---

## Feature Availability Matrix

| Feature | AdamW | Prodigy | Adopt | Lion | Muon | AdaMuon | SignSGD | SinkSGD |
|---|---|---|---|---|---|---|---|---|
| OrthoGrad | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Spectral Norm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Stochastic Sign | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| SNR Conditioning | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| NtM | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Kourkoutas-β | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| FAdam | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Centered WD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cautious WD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Geometric WD | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| SMMF / Factored | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Atan2 | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Compiled | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
