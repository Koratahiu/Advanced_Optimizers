# Settings & Features

This page covers the settings and features that are shared across most (or all) optimizers in the library.

---

## Learning Rate

| Parameter | Default | Description |
|---|---|---|
| `lr` | varies by optimizer | Base learning rate. |

*  Prodigy uses 1.0 as default since it self-adapts.

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

---

## Fused Back Pass

All optimizers support Fused back pass hooks.
Fused back pass hooks each parameter individually after its gradient is computed. As soon as a parameter's gradient is ready, the hook calls `optimizer.step_parameter()`, then immediately sets `tensor.grad = None`, freeing that parameter's gradient memory right away.

This means you never hold the full set of model gradients in memory simultaneously. Each parameter's gradient is consumed and freed one at a time during backward, saving roughly one full model size of VRAM.

**Example (inside a trainer):**

The trainer registers a `post_accumulate_grad_hook` on each parameter. The hook flow is:
1. Gradient for that parameter has finished accumulating.
2. (Multi-GPU) Reduce the gradient across devices if enabled.
3. Apply gradient clipping.
4. Call `optimizer.step_parameter(tensor, param_group, index)`: stepping *this* parameter only.
5. Set `tensor.grad = None`: freeing the gradient immediately.


