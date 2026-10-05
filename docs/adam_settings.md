# Adam-Related Settings

This page covers settings specific to `AdamW_adv`, `Adopt_adv`, and `Prodigy_adv`. All three are Adam-family optimizers and share many of the same advanced features.

---

## Atan2 Update Rule

Replaces the standard adaptive scaling with an atan2-based normalization, which can improve stability by removing the need for `eps`.

| Parameter | Default | Description |
|---|---|---|
| `use_atan2` | `False` | Enables the atan2 update rule. |

* Available on: `AdamW_adv`, `Prodigy_adv`, `Adopt_adv`, `AdaMuon_adv`.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    lr=1e-3,
    use_atan2=True,
)
```

---

## Variance-Normalized Momentum (NtM)

Normalization-then-Momentum applies the adaptive normalization *before* computing momentum. ADOPT is already bundled with this behavior by design.

| Parameter | Default | Description |
|---|---|---|
| `normed_momentum` | `False` | Enables NtM. |

* Available on: `AdamW_adv`, `SignSGD_adv`, and `SinkSGD_adv`.
* ADOPT applies normalization before momentum inherently.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    lr=1e-3,
    normed_momentum=True,
)
```

---

## Factored Second Moment (`factored_2nd`)

Keeps the first moment in full precision while compressing only the second moment into a rank-2 factored form. Works alongside any `state_precision` setting.

| Parameter | Default | Description |
|---|---|---|
| `factored_2nd` | varies | Compresses the second moment with SMMF. Default is `True` for Prodigy, `False` for others. |

* Available on: `AdamW_adv`, `Prodigy_adv`, `Adopt_adv`.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    state_precision='bf16_sr',
    factored_2nd=True,
)
```

---

## Fisher Weight Decay (`fisher_wd`)

Maps the decay direction through the empirical Fisher information matrix and clips its RMS, as described in the [FAdam paper](https://arxiv.org/abs/2405.12807). Provides geometry-aware regularization.

| Parameter | Default | Description |
|---|---|---|
| `fisher_wd` | `False` | Enables Fisher weight decay. |

* Available on: `AdamW_adv`, `Prodigy_adv`, `Adopt_adv`.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    weight_decay=0.01,
    fisher_wd=True,
)
```

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

* Available on: `AdamW_adv`, `Prodigy_adv`, `Adopt_adv`, and the auxiliary AdamW in `Muon_adv` / `AdaMuon_adv`.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    betas=(0.9, 0.999),
    kourkoutas_beta=True,
    beta2_min=0.88,
    ema_alpha=0.93,
    k_logging=100,
)
```

* Paper: [Kourkoutas-Beta: A Sunspike-Driven Adam Optimizer with Desert Flair](https://arxiv.org/abs/2508.12996)

---

## Prodigy-Specific Settings

Prodigy estimates its own learning rate on the fly via the `d`-adaptation mechanism. These settings control that adaptation.

| Parameter | Default | Description |
|---|---|---|
| `d0` | 1e-6 | Initial D estimate. Rarely needs changing. |
| `d_coef` | 1.0 | Coefficient in the d-estimate expression. Tune to adjust adaptation speed (0.5–2.0 typically work). |
| `growth_rate` | inf | Max multiplicative rate for D estimate growth. ~1.02 gives a learning-rate warmup effect. |
| `slice_p` | 11 | Calculate adaptation stats on every pth entry to save memory. Values ~11 are reasonable. |
| `prodigy_steps` | 0 | Disable adaptation after this many steps; release all Prodigy state memory. |
| `d_limiter` | `False` | Clamp d_hat to prevent volatile step-size increases. |
| `fsdp_in_use` | `False` | Set True when using FSDP sharded parameters (auto-detected otherwise). |

**Example:**
```python
optimizer = Prodigy_adv(
    model.parameters(),
    lr=1.0,  # Prodigy self-adapts; this is the cap
    d_coef=0.5,
    growth_rate=1.02,
    d_limiter=True,
)
```

---

## ADOPT-Specific Settings

ADOPT modifies Adam by initializing the second moment as `v₀ = g₀²` and normalizing using the *previous* step's second moment before updating momentum.

| Parameter | Default | Description |
|---|---|---|
| `clip_lambda` | `lambda step: step**0.25` | Function that clips the normalized gradient. Only used when `use_atan2` is False. |

**Example:**
```python
optimizer = Adopt_adv(
    model.parameters(),
    lr=1e-4,
    clip_lambda=lambda step: step ** 0.25,
)
```

---


## Quick Comparison

| Feature | AdamW_adv | Adopt_adv | Prodigy_adv |
|---|---|---|---|
| Kourkoutas-β | ✅ | ✅ | ✅ |
| Fisher WD | ✅ | ✅ | ✅ |
| Atan2 | ✅ | ✅ | ✅ |
| NtM | ✅ | Bundled | ❌ |
| Self-adaptive LR | ❌ | ❌ | ✅ |
| Factored 2nd | ✅ | ✅ | ✅ (default) |
| Clip lambda | ❌ | ✅ | ❌ |
