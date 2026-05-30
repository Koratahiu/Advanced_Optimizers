# Advanced Optimizers (AIO)

A comprehensive, all-in-one collection of optimization algorithms for deep learning, designed for **maximum efficiency**, **minimal memory footprint**, and **superior performance** across diverse model architectures and training scenarios.

[![PyPI](https://img.shields.io/pypi/v/adv_optm)](https://pypi.org/project/adv_optm/)

## 🔥 What's New

### in 2.1.x

- Added Signum (SignSGD with momentum): A new optimizer in the family (SignSGD_adv)
- More info coming soon.

### in 2.0.x

* Implemented torch.compile for all advanced optimizers. Enabled via (compiled_optimizer=True) to fuse and optimize the optimizer step path.
* Better and improved 1-bit factored mode via (nnmf_factor=True).
* Various improvements across the optimizers.

### in 1.2.x
* Added **advanced variants** of [Muon optimizer](https://kellerjordan.github.io/posts/muon/) with **features** and **settings** from recent papers.

| Optimizer | Description |
|---|---|
| `Muon_adv` | Advanced Muon implementation with CANS, NorMuon, Low-Rank ortho, etc. features. |
| `AdaMuon_adv` | Advanced AdaMuon implementation, which combines Muon's geometry with Adam-like adaptive scaling and sign-based orthogonalization. |

> *Documentation coming soon.*

* Implemented [Cautious Weight Decay](https://arxiv.org/abs/2510.12402) for all advanced optimizers.

* Improved parameter update and weight decay for **BF16** with **stochastic rounding**. The updates are now accumulated in **float32** and rounded once at the end.

* Use fused and in-place operations whenever possible for all advanced optimizers.

* **Prodigy variants** are now **50% faster** by [avoiding CUDA syncs](https://github.com/Koratahiu/Advanced_Optimizers/pull/5). Thanks to **@dxqb**!

---

## 📦 Installation

```bash
pip install adv_optm
```

---

## 🧠 Core Innovations

This library integrates multiple state-of-the-art optimization techniques validated through extensive research and practical training, with **1-bit compression for optimizer states**:

### **Memory-Efficient Optimization (SMMF-inspired)**
- **Paper**: [SMMF: Square-Matricized Momentum Factorization](https://arxiv.org/abs/2412.08894)
- **Approach**: Uses rank-1 non-negative matrix factorization with reconstruction cycle (factor → reconstruct → update → factor)
- **Innovation**:
  - First moment split into **1-bit sign + absolute value**
  - Final storage: **four factored vectors + one 1-bit sign state**
  - Preserves Adam-like update quality with drastically reduced memory

---

## ⚡ Performance Characteristics

### Memory Efficiency (SDXL Model – 6.5GB)
| Optimizer | Memory Usage | Description |
|-----------|--------------|-------------|
| `Adopt_Factored` | 328 MB | 4 small vectors + 1-bit state |
| `Adopt_Factored + AdEMAMix` | 625 MB | 6 small vectors + two 1-bit states |

### Speed Comparison (SDXL, Batch Size 4)
| Optimizer | Speed | Notes |
|-----------|-------|-------|
| `Adafactor` | ~8.5s/it | Baseline |
| `Adopt_Factored` | ~10s/it | +18% overhead from compression |
| `Adopt_Factored + AdEMAMix` | ~12s/it | +41% overhead (3 factored states) |

---

## 🧪 Available Optimizers

### Standard Optimizers (All support `factored=True/False`)
| Optimizer | Description | Best For |
|-----------|-------------|----------|
| `Adam_Adv` | Advanced Adam implementation | General purpose |
| `Adopt_Adv` | Adam-variant with independent beta2 | Stable training for small batch size regimes |
| `Prodigy_Adv` | Prodigy with D-Adaptation | Adam with automatic LR tuning |
| `Lion_Adv` | Advanced Lion implementation | Memory-constrained environments |
| `Prodigy_Lion_Adv` | Prodigy + Lion combination | Lion with automatic LR tuning |

---

## ⚙️ Feature Matrix

| Feature | Adam_Adv | Adopt_Adv | Prodigy_Adv | Lion_Adv |
|---------|----------|-----------|-------------|----------|
| Factored | ✓ | ✓ | ✓ ✓ |
| AdEMAMix | ✓ | ✓ | ✓ | ✗ |
| OrthoGrad | ✓ | ✓ | ✓ | ✓ |
| Grams | ✓ | ✓ | ✓ | ✗ |
| Cautious | ✓ | ✓ | ✓ | ✓ |
| atan2 | ✓ | ✓ | ✓ |✗ |
| Stochastic Rounding | ✓ | ✓ | ✓ |✓ |
| Fused Backward Pass | ✓ | ✓ | ✓ | ✓ |
| **Kourkoutas-β** | ✓ | ✓ | ✓ | ✗ |

---

## 🛠️ Comprehensive Feature Guide

### A. Universal Safe Features
*These features work with all optimizers and are generally safe to enable.*

| Feature | Description | Recommended Usage | Performance Impact | Theoretical Basis | Compatibility |
|--------|-------------|-------------------|--------------------|-------------------|--------------|
| **Fused Back Pass** | Fuses backward pass; gradients used immediately and memory freed on-the-fly | Memory-constrained environments | Reduces peak memory | Memory optimization | All optimizers |
| **Stochastic Rounding** | Replaces nearest rounding with stochastic rounding to preserve small gradient updates in BF16 | BF16 training | Minimal overhead (<5%) | [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192) | All optimizers |
| **OrthoGrad** | Removes gradient component parallel to weights to reduce overfitting | Full fine-tuning without weight decay | +33% time overhead (BS=4); less at larger BS | [Grokking at Edge](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability) | All optimizers |
| **Factored** | Memory-efficient optimization via rank-1 1-bit factorization of optimizer states | Large models / memory-limited hardware | Adds compression overhead | [SMMF](https://arxiv.org/abs/2412.08894) | All optimizers |

### B. Individual Features

| Feature | Description | Recommended Usage | Performance Impact | Theoretical Basis | Compatibility |
|--------|-------------|-------------------|--------------------|-------------------|--------------|
| **Cautious** | Only applies update if gradient direction aligns with momentum direction | Accelerating convergence | No overhead | [C-Optim](https://github.com/kyleliang919/C-Optim) | Adam/Adopt/Prodigy/Lion |
| **Grams** | Update direction derived purely from current gradient | When Cautious is insufficient | No overhead | [Grams](https://github.com/Gunale0926/Grams) | Adam/Adopt/Prodigy |
| **AdEMAMix** | Dual EMA system that retains relevance of gradients over tens of thousands of steps | Long training runs, especially where model forgetting is a concern | +1 state memory | [AdEMAMix](https://arxiv.org/abs/2409.03137) | Adam/Adopt/Prodigy |
| **atan2** | Robust epsilon replacement with built-in gradient clipping | Use for stable bounded updates (or for Adopt as it needs that) | No overhead | [Adam-atan2](https://github.com/lucidrains/adam-atan2-pytorch) | Adam/Adopt/Prodigy |
| **Kourkoutas-β** | Layer-wise adaptive β₂ based on gradient “sunspike” ratio | Noisy/small/large-batch/high-LR training | No overhead | [Kourkoutas-β]() | Adam/Adopt/Prodigy |

> **Note**: If both **Cautious** and **Grams** are enabled, **Grams takes precedence** and Cautious is disabled.

---

## 🔍 Feature Deep Dives

### AdEMAMix

- Adds a **slow-decaying second EMA** (`beta3`) that retains gradient memory over tens of thousands of steps.
- Particularly effective for **small batch sizes**, where Adam’s standard first moment is nearly useless.

#### Tunable Hyperparameters
| Parameter | Default | Tuning Guide |
|-----------|---------|--------------|
| `beta3` | 0.9999 | • Runs >120k steps: **0.9999**<br>• Runs ≤120k steps: **0.999** |
| `alpha` | 5 | • Reduce to **2–3** if diverging<br>• Increase to strengthen long-term memory |

> ✅ **Pro Tip**: Set `beta1=0` in Adam/Adopt/Prodigy to skip standard EMA entirely and rely solely on AdEMAMix’s slow EMA, ideal for small-batch regimes.

---

### atan2

- Replaces `eps` in Adam-family optimizers with a **scale-invariant**, bounded update rule.
- Automatically clips updates to **[-2, 2]**, preventing destabilizing jumps.
- **Highly recommended** for `Adopt_Adv`, which is prone to instability without clipping.

> 📚 **Reference**:
> - Paper: https://arxiv.org/abs/2407.05872
> - Code: https://github.com/lucidrains/adam-atan2-pytorch

---

### **Kourkoutas-β**

**Kourkoutas-β** introduces a **sunspike-driven, layer-wise adaptive second-moment decay (β₂)** as an optional enhancement for `Adam_Adv`, `Adopt_Adv`, `Prodigy_Adv`.

Instead of using a fixed β₂ (e.g., 0.999 or 0.95), it **dynamically modulates β₂ per layer** based on a bounded *sunspike ratio*:

- **During gradient bursts** → β₂ ↓ toward `Lower β₂` → faster reaction
- **During calm phases** → β₂ ↑ toward `The Selected β₂` → stronger smoothing

This is especially effective for **noisy training, small batch sizes, and high learning rates**, where gradient norms shift abruptly due to noise or aggressive LR schedules.

#### Pros/Cons

| **Category** | **Details** |
|--------------|-------------|
| ✅ **Pros** | • **Layer-wise adaptation** blends benefits of high β₂ (strong smoothing) and low β₂ (fast reaction).<br>• **Robust to sudden loss landscape shifts**, reacts quickly during gradient bursts, smooths during calm phases.<br>• **High tolerance to aggressive learning rates**. |
| ⚠️ **Cons** | • **Potentially unstable at the start of training** due to unreliable early gradient norms; mitigated by using `K-β Warmup Steps`. |

> 💡 **Best Practice**: Set `K_warmup_steps` equal to your standard LR warmup steps. During warmup, the optimizer uses the static `beta2`; adaptation begins only after warmup ends.

> 📚 **Reference**:
> - Paper: [Kourkoutas-β: A Sunspike-Driven Adam Optimizer with Desert Flair](https://arxiv.org/abs/2508.12996)
> - Code: [kbeta](https://github.com/sck-at-ucy/kbeta)

---

## 📚 References

1. [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192)
2. [SMMF: Square-Matricized Momentum Factorization](https://arxiv.org/abs/2412.08894)
3. [The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137)
4. [Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD](https://arxiv.org/abs/2502.02431)
6. [Kourkoutas-β: A Sunspike-Driven Adam Optimizer with Desert Flair](https://arxiv.org/abs/2508.12996)
7. [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872)
