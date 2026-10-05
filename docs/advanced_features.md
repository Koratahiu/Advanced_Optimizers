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

**Parameter tagging for PEFT/LoRA:**
Spectral normalization reads special attributes from parameter tensors to apply the correct scaling. For LoRA, DoRA, and OFT layers, you must tag parameters before creating the optimizer:

```python
p._is_lora_B = True
```

Set the following attributes based on parameter names:

| Attribute | Set On | Purpose |
|---|---|---|
| `_is_lora_A` | `lora_down.weight`, `lokr_w1_b`, `lokr_w2_b` | Down projection: spectral target of `1` |
| `_is_lora_B` | `lora_up.weight`, `lokr_w1_a`, `lokr_w2_a` | Up projection: spectral target of `sqrt(d_out / d_in)` |
| `_is_dora_scale` | `dora_scale`, `dora_log_multiplier` | DoRA scale vector: max-abs normalized |
| `_is_oft` | `oft_R.weight` | OFT rotation matrix: Decomposition to skew-symmetric matrix and spectral target of `0.5` |

* This ensures scale-invariant effective LR O(1) across all parameters.

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

## Sinkhorn Settings (SinkSGD)

| Parameter | Default | Description |
|---|---|---|
| `sinkhorn_iterations` | 5 | Number of Sinkhorn iterations. |
| `orthogonal_sinkhorn` | `False` | Uses orthogonal Sinkhorn variant. This reaches a state where all rows and cols are orthogonal to the weights (rank-2 Orthogonal projection) |

* For Muon-specific settings (Newton-Schulz, CANS, NorMuon, MARS-M), see [Muon-Related Settings](muon_settings.md).
* For Adam/Prodigy-specific settings (Kourkoutas-β, FAdam, Atan2, Prodigy adaptation), see [Adam-Related Settings](adam_settings.md).

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
