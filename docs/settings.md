# Settings & Features

This page covers the settings and features that are shared across most (or all) optimizers in the library.

---

## Learning Rate

| Parameter | Default | Description |
|---|---|---|
| `lr` | varies by optimizer | Base learning rate. Prodigy uses 1.0 as default since it self-adapts. |

---

## Basic Optimizer Parameters

| Parameter | Default | Description |
|---|---|---|
| `betas` / `beta1` / `momentum` | varies | Momentum or first-moment coefficients. Ranges from 0 (no momentum) to 1 (heavy smoothing). |
| `eps` | varies | Numerical stability term added to denominators. Set to `None` for scale-invariant eps. |
| `weight_decay` | 0.0 | Decoupled weight decay coefficient. |

---


## Stochastic Rounding

| Parameter | Default | Description |
|---|---|---|
| `stochastic_rounding` | `True` | Accumulates parameter updates and weight decay in float32, then rounds once at the end. Reduces quantization noise in BF16 training. |

---

## Compiled Optimizer

| Parameter | Default | Description |
|---|---|---|
| `compiled_optimizer` | `False` | Compiles the core step function with `torch.compile` for fused, optimized execution. Requires PyTorch 2.3+. |

