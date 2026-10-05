# Momentum Settings

This page covers all momentum-related configurations available across the library.

---

## Standard Momentum

Most optimizers accept a momentum or beta parameter:

| Optimizer | Parameter | Default | Range |
|---|---|---|---|
| `AdamW_adv` | `betas[0]` | 0.9 | [0, 1) |
| `Prodigy_adv` | `betas[0]` | 0.9 | [0, 1) |
| `Adopt_adv` | `betas[0]` | 0.9 | [0, 1) |
| `Lion_adv` | `betas[0]` | 0.9 | [0, 1) |
| `Muon_adv` | `beta1` | 0.9 | [0, 1) |
| `AdaMuon_adv` | `betas[0]` | 0.95 | [0, 1) |
| `SignSGD_adv` | `momentum` | 0.9 | [0, 1] |
| `SinkSGD_adv` | `momentum` | 0.0 | [0, ∞) |

Higher values mean smoother but more lagging updates. Lower values respond faster to gradient changes.

---

## Nesterov Momentum

Nesterov momentum looks ahead before applying the update, often improving convergence.

| Parameter | Default | Description |
|---|---|---|
| `nesterov` | `False` | Enables Nesterov momentum. Must be used with non-zero momentum. |
| `nesterov_coef` | `None` | Optional coefficient for the Nesterov look-ahead. Defaults to a sensible value (=`momentum value`) when None. |


**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    lr=1e-3,
    betas=(0.99, 0.999),
    nesterov=True,
    nesterov_coef=0.8,
)
```

Or:

```python
optimizer = AdamW_adv(
    model.parameters(),
    lr=1e-3,
    betas=(0.95, 0.999),
    nesterov=True,
    nesterov_coef=None,
)
```

---

## Variance-Normalized Momentum (NtM)

Normalization-then-Momentum applies the adaptive normalization *before* computing momentum. This can improve stability, especially with sign-based or sinkhorn-normalized updates.

| Parameter | Default | Description |
|---|---|---|
| `normed_momentum` | `False` | Enables NtM. |

* Available on `AdamW_adv`, `SignSGD_adv`, and `SinkSGD_adv`
* ADOPT is already bundled with it.

**Example:**
```python
optimizer = SignSGD_adv(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    normed_momentum=True,
)
```

> **Note:** `snr_cond` (see [Advanced Features](advanced_features.md)) requires `normed_momentum=True`.

---

## Kourkoutas-β (Dynamic β₂)

Replaces the fixed β₂ with a layer-wise dynamic schedule that lowers β₂ during periods of high gradient variance ("sunspikes"), then recovers it. This improves both stability and convergence speed.

| Parameter | Default | Description |
|---|---|---|
| `kourkoutas_beta` | `False` | Enables dynamic β₂ scheduling. |
| `beta2_min` | 0.88 | Minimum β₂ during sunspikes. Must be less than `betas[1]`. |
| `ema_alpha` | 0.93 | Decay rate for the EMA of pooled gradient norms. |
| `tiny_spike` | 1e-9 | Constant to prevent division by zero in sunspike detection. |
| `k_warmup_steps` | 0 | Steps to hold β₂ at its fixed value before dynamic logic activates. |
| `k_logging` | 0 | If > 0, logs β₂ statistics (min/max/mean) every `k_logging` steps. |
| `layer_key_fn` | `None` | Custom function to bucket parameters into layers for shared β₂ scheduling. |

* Available on: `AdamW_adv`, `Prodigy_adv`, `Adopt_adv`, `Muon_AuxAdam`.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    betas=(0.9, 0.999),
    kourkoutas_beta=True,
    beta2_min=0.88,
    ema_alpha=0.93,
)
```

* Paper: [Kourkoutas-Beta: A Sunspike-Driven Adam Optimizer with Desert Flair](https://arxiv.org/abs/2508.12996)

---

## Lion-K (p-Norm Geometry)

Lion-K generalizes the sign update to an Lp-norm, letting you interpolate between sign updates (p=1) and spherical normalization (p=2).

| Parameter | Default | Description |
|---|---|---|
| `kappa_p` | 1.0 | The p-value for the Lp-norm (domain [1.0, 2.0]). |
| `auto_kappa_p` | `False` | Auto-sets p=2.0 for 4D tensors (Conv2D) and p=1.0 otherwise. |

* Available on: `Lion_adv`, `AdaMuon_adv`.
* Paper: [Lion Secretly Solves Constrained Optimization, As Lyapunov Predicts](https://arxiv.org/abs/2310.05898v7)

**Example:**
```python
# Standard Lion (sign update)
optimizer = Lion_adv(model.parameters(), kappa_p=1.0)

# Spherical Lion (L2-normalized update)
optimizer = Lion_adv(model.parameters(), kappa_p=2.0)

# Between sign and spherical
optimizer = Lion_adv(model.parameters(), kappa_p=1.5)

# Auto mode
optimizer = Lion_adv(model.parameters(), auto_kappa_p=True)
```

---

## Atan2 Update Rule

Replaces the standard adaptive scaling with an atan2-based normalization, which can improve stability by removing the need for `eps`.

| Parameter | Default | Description |
|---|---|---|
| `use_atan2` | `False` | Enables the atan2 update rule. |

Available on: `AdamW_adv`, `Prodigy_adv`, `Adopt_adv`, `AdaMuon_adv`.
