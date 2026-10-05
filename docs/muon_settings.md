# Muon-Related Settings

This page covers settings specific to `Muon_adv` and `AdaMuon_adv`.

---

## Newton-Schulz Iteration

Controls the orthogonalization process. Muon applies SGD with momentum, then orthogonalizes the update matrix using a Newton-Schulz polynomial iteration.

| Parameter | Default | Description |
|---|---|---|
| `ns_steps` | 5 | Number of Newton-Schulz iterations. More steps = more accurate orthogonalization but slower. |
| `ns_eps` | 1e-7 | Epsilon for normalization stability. Set to `None` to use the scale-invariant rule. |
| `ns_coeffs` | `(3.4445, -4.7750, 2.0315)` | (a, b, c) coefficients for the quintic polynomial used in the iteration. |

**Example:**
```python
optimizer = Muon_adv(
    model.parameters(),
    ns_steps=5,
    ns_eps=None,  # scale-invariant rule
)
```

---

## Chebyshev-Accelerated NS (`accelerated_ns`)

Dynamically calculates optimal 3rd-order polynomial coefficients for the Newton-Schulz iteration using Chebyshev polynomials. Faster convergence than fixed coefficients, especially for ill-conditioned matrices.

| Parameter | Default | Description |
|---|---|---|
| `accelerated_ns` | `False` | Enables CANS. |
| `cns_a_bound` | `None` | Initial lower bound for singular values. Set to `None` to use the scale-invariant rule. |

**Example:**
```python
optimizer = Muon_adv(
    model.parameters(),
    accelerated_ns=True,
    cns_a_bound=None,
)
```

---

## NorMuon (`normuon_variant`)

Adds neuron-wise normalization to Muon, computing per-neuron second moments and normalizing accordingly. Improves training stability for deep networks.

| Parameter | Default | Description |
|---|---|---|
| `normuon_variant` | `False` | Enables NorMuon. |
| `beta2_normuon` | 0.95 | Decay rate for the second-moment EMA used in NorMuon. |
| `normuon_eps` | 1e-8 | Stability epsilon for NorMuon normalization. |

**Example:**
```python
optimizer = Muon_adv(
    model.parameters(),
    normuon_variant=True,
    beta2_normuon=0.95,
)
```

* Paper: NorMuon: [Making Muon more efficient and scalable](https://arxiv.org/abs/2510.05491)

---

## Low-Rank Orthogonalization (`low_rank_ortho`)

Projects the momentum update to a lower rank before orthogonalization, reducing computational cost for large matrices. Useful when full-rank Newton-Schulz is too expensive.

| Parameter | Default | Description |
|---|---|---|
| `low_rank_ortho` | `False` | Enables low-rank orthogonalization. |
| `ortho_rank` | 128 | Rank for the projection. |

**Example:**
```python
optimizer = AdaMuon_adv(
    model.parameters(),
    low_rank_ortho=True,
    ortho_rank=64,
)
```

---

## Approx MARS-M (`approx_mars`)

Variance reduction based on the MARS-M paper ("MARS-M: When Variance Reduction Meets Matrices"). Reduces gradient noise in Muon-style updates by maintaining a running average of past gradients.

| Parameter | Default | Description |
|---|---|---|
| `approx_mars` | `False` | Enables MARS-M variance reduction. |
| `mars_gamma` | 0.025 | Scaling coefficient for the gradient correction term. |

**Example:**
```python
optimizer = Muon_adv(
    model.parameters(),
    approx_mars=True,
    mars_gamma=0.025,
)
```

---

## RMS Rescaling (`rms_rescaling`)

Uses Root-Mean-Square for the final update vector, aligning update magnitudes with Adam-style optimizers. This allows reuse of existing Adam learning-rate schedules with Muon.

| Parameter | Default | Description |
|---|---|---|
| `rms_rescaling` | `True` | Enables RMS-aligned rescaling. |

> **Note:** Not compatible with `spectral_normalization`.

---

## Auxiliary AdamW (Muon + Adam parameter groups)

Both `Muon_adv` and `AdaMuon_adv` can handle mixed parameter groups — Muon for matrix weights and an auxiliary AdamW for biases, norms, and other 1D parameters.

Specify the optimizer type per parameter group using the `optim_type` key:

```python
muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'bias' not in n]
adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or 'bias' in n]

optimizer = Muon_adv(
    [
        {'params': muon_params, 'optim_type': 'muon'},
        {'params': adam_params, 'optim_type': 'adam'},
    ],
    lr=1e-3,
)
```

When `use_muon` is not provided in a group, the optimizer auto-detects based on parameter shape (2D+ → Muon, otherwise → AdamW).
