# Weight Decay

This page covers all weight decay variants available across the library.

---

## Standard Weight Decay

The classic L2 regularization, applied as a simple penalty on parameter magnitudes.

| Parameter | Default | Description |
|---|---|---|
| `weight_decay` | 0.0 | L2 penalty coefficient. |

```python
optimizer = AdamW_adv(model.parameters(), weight_decay=0.01)
```

---

## Cautious Weight Decay (`cautious_wd`)

Applies weight decay **only** to parameter coordinates where the sign of the parameter and the sign of the optimizer update align. This prevents decay from fighting against the optimizer directions.

| Parameter | Default | Description |
|---|---|---|
| `cautious_wd` | `False` | Enables cautious weight decay. |

* Available on all optimizers.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    weight_decay=0.01,
    cautious_wd=True,
)
```

> Based on the [Cautious Weight Decay](https://arxiv.org/abs/2510.12402).

---

## Centered Weight Decay (`centered_wd`)

Instead of decaying weights toward zero, centered weight decay pulls them toward their initial values (anchors). This preserves the relative structure of pretrained weights while still regularizing.

| Parameter | Default | Description |
|---|---|---|
| `centered_wd` | 0.0 | Centered weight decay coefficient. |
| `centered_wd_mode` | `'float8'` | Quantization format for storing anchors to save VRAM. |

**Anchor precision options:**

| Value | Description | Memory |
|---|---|---|
| `'full'` | Stores anchors in the original parameter dtype. | Highest |
| `'float8'` | Uses `torch.float8_e4m3fn`. | Low |
| `'int8'` | 8-bit block-wise quantization (block size 128). | Low |
| `'int4'` | 4-bit block-wise quantization (block size 32). | Lowest |

* Available on all optimizers.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    weight_decay=0.01,
    centered_wd=0.1,
    centered_wd_mode='float8',
)
```

---

## Fisher Weight Decay (`fisher_wd`)

Maps the decay direction through the empirical Fisher information matrix and clips its RMS, as described in the [FAdam paper](https://arxiv.org/abs/2405.12807). This provides geometry-aware regularization.

| Parameter | Default | Description |
|---|---|---|
| `fisher_wd` | `False` | Enables Fisher weight decay. |

Available on: `AdamW_adv`, `Prodigy_adv`, `Adopt_adv`.

**Example:**
```python
optimizer = AdamW_adv(
    model.parameters(),
    weight_decay=0.01,
    fisher_wd=True,
)
```

* Compatible with `centered_wd` and `cautious_wd`

---

## Geometric Weight Decay (`geometric_wd`)

A weight decay variant designed for sign-based and Sinkhorn-normalized optimizers. It operates on the geometric properties of the update direction:
1. Signed-L1 penalty for `SignSGD_adv`
2. Row and column distribution penalty for `SinkSGD_adv`

| Parameter | Default | Description |
|---|---|---|
| `geometric_wd` | `False` | Enables geometric weight decay. |

Available on: `SignSGD_adv`, `SinkSGD_adv`.

**Example:**
```python
optimizer = SignSGD_adv(
    model.parameters(),
    weight_decay=0.01,
    geometric_wd=True,
)
```

* Compatible with `centered_wd` and `cautious_wd`

---

## Combining Weight Decay Variants

Standard and centered weight decay can be used together:

```python
optimizer = AdamW_adv(
    model.parameters(),
    weight_decay=0.01,        # standard L2 decay
    centered_wd=0.1,          # pull toward anchors
    cautious_wd=True,         # only decay aligned coordinates
)
```

Fisher weight decay is an alternative to (not additive with) standard weight decay, it replaces the simple L2 penalty with a geometry-aware version.

---

## Quick Reference

| Variant | Parameters | Available On |
|---|---|---|
| Standard | `weight_decay` | All |
| Cautious | `cautious_wd` | All |
| Centered | `centered_wd`, `centered_wd_mode` | All |
| Fisher | `fisher_wd` | AdamW, Prodigy, Adopt |
| Geometric | `geometric_wd` | SignSGD, SinkSGD |
