# Memory & Precision

This page covers settings that control memory usage and state precision in optimizer states.

---

## State Precision (`state_precision`)

Controls the dtype used to store optimizer states (first moment, second moment, etc.). Lower precision saves VRAM at the cost of some numerical accuracy.

| Value | Description |
|---|---|
| `'auto'` | Uses the parameter's native dtype. Best compatibility. |
| `'fp32'` | Full float32 precision. Highest memory usage. |
| `'bf16_sr'` | BF16 with stochastic rounding. Good balance of speed and accuracy. |
| `'fp16'` | Float16 precision. |
| `'int8_sr'` | 8-bit integer with stochastic rounding and per-block scales. |
| `'factored'` | Rank-2 factored mode (SMMF). Stores states as two low-rank matrices. Lowest memory. |

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    lr=1e-3,
    state_precision='bf16_sr',
)
```

> **Note:** Not supported in Lion_adv (WIP).

---

## Factored Second Moment (`factored_2nd`)

Keeps the first moment in `state_precision` precision while compressing only the second moment into a rank-2 factored form.

| Parameter | Default | Description |
|---|---|---|
| `factored_2nd` | `False` | Compresses the second moment with SMMF. Available on all Adam variants. |

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    lr=1e-3,
    state_precision='bf16_sr',
    factored_2nd=True,
)
```

This combination gives you both precision control on the first moment and memory savings on the second moment.

---

## Low-Rank Factorization (SMMF)

The Structured Memory-Minimizing Factorization (SMMF) technique compresses optimizer states into low-rank factors, dramatically reducing VRAM usage.

| Parameter | Default | Description |
|---|---|---|
| `nnmf_factor` or `state_precision="factored"` | `False` | Enables SMMF factorization. |
| `vector_reshape` | `False` | Reshapes 1D states into 2D matrices before factorization, enabling rank-2 compression. |

**Memory savings:** Factorized states can use as little as ~1-bit the parameter memory.
- **`vector_reshape=False`** is not recommended when using factorization, but it can be used for large vectors as it enables rank-2 compression on vector states.

* Paper: [SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization](https://arxiv.org/abs/2412.08894)

**Example:**
```python
optimizer = Lion_adv(
    model.parameters(),
    lr=1e-4,
    nnmf_factor=True,
    vector_reshape=True,
)
```

---

## Stochastic Rounding (`stochastic_rounding`)

When enabled (default: `True`), parameter updates and weight decay are accumulated in float32 and rounded once at the end. This reduces quantization noise when training in BF16.

```python
optimizer = AdamW_adv(
    model.parameters(),
    lr=1e-3,
    stochastic_rounding=True,  # default
)
```

---

## Quick Reference: Memory Modes

| Mode | Approx. State Memory (per parameter) | Best For |
|---|---|---|
| `fp32` | 2× param size (Adam-style) | Maximum accuracy |
| `bf16_sr` | ~1× param size | Balanced speed/memory |
| `int8_sr` | ~0.25× param size | Extreme memory constraints |
| `factored` | ~2× rank × dim | Large models, limited VRAM |
| `factored_2nd` + `bf16_sr` | ~1× param size (1st) + compressed (2nd) | Best of both worlds |
