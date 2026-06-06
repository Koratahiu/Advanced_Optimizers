# Advanced Optimizers (AIO)

A comprehensive, all-in-one collection of state-of-the-art optimization algorithms for deep learning. Designed for **maximum efficiency**, **minimal memory footprint**, and **superior performance** across diverse model architectures and training scenarios.

[![PyPI version](https://img.shields.io/pypi/v/adv_optm.svg?color=blue&style=flat-square)](https://pypi.org/project/adv_optm/)
[![Python versions](https://img.shields.io/pypi/pyversions/adv_optm.svg?style=flat-square)](https://pypi.org/project/adv_optm/)
[![License](https://img.shields.io/badge/license-Apache-green?style=flat-square)](LICENSE)

---

## 📦 Installation

```bash
pip install adv_optm
```
*Requires PyTorch 2.3+ for `torch.compile` support.*

---

## What's New

### 🌟 Version 2.5.x: The Massive Refactor
This major update introduces a complete architectural refactor of the library:

**🆕 New Optimizers & Scaling**
* **`SinkSGD_adv`:** Added a powerful new optimizer to the lineup.
* **Spectral Scaling:** Now available across *all* optimizers, achieving width/rank invariant updates for highly stable training.

**💾 Memory & State Precision Control**
* **Granular State Precision (`state_precision`):** Drastically reduce memory overhead with new optimizer state modes: 
  * `factored` (Rank-2 factored mode)
  * `fp32` (Full precision)
  * `bf16_sr` & `int8_sr` (BF16/Int8 with Stochastic Rounding)
* **Factored Second Moment (`factored_2nd`):** Available for all Adam variants. Works seamlessly alongside any `state_precision` setting to further slash memory usage.

**⚙️ Advanced Dynamics & Momentum**
* **Variance Normalized Momentum (`normed_momentum`):** Applies optimizer normalization *before* momentum (Normalization then Momentum/NtM). Available for `AdamW_adv`, `SignSGD_adv`, and `SinkSGD_adv`.
* **Universal Nesterov Momentum:** Replaced the hard-to-tune Simplified_AdEMAMix with Nesterov momentum (`nesterov`) and a dedicated coefficient (`nesterov_coef`) across all optimizers.
* **Preconditioning & Signs:** 
  * Added **Variance/Confidence Preconditioning (`snr_cond`)** for `SignSGD_adv` and `SinkSGD_adv` (requires `normed_momentum`). Read the technical reports: [AASS](https://koratahiu.github.io/aass/) & [sink-v](https://koratahiu.github.io/sink-v/).
  * Added **Adaptive Stochastic Sign** with $L_\infty$ preconditioning (`stochastic_sign`) for `SignSGD_Adv` and `Lion_adv`.
* **Improved CANS (`accelerated_ns`):** Enhanced for Muon variants by integrating a dynamic lower bound.
* **New OrthoGrad modes (`orthogonal_gradient`):** Standard OrthoGrad `flattened` and a new matrix-wise mode `iterative`.

**⚓ Weight Decay Innovations**
* **Centered Weight Decay (`centered_wd`):** Pulls weights toward their pre-train state (anchor). To save memory, anchor precision (`centered_wd_mode`) can be set to full, float8, int8, or int4.
* **Fisher Weight Decay (`fisher_wd`):** Now available for Adam variants based on the [FAdam paper](https://arxiv.org/abs/2405.12807).
* **Geometric Weight Decay:** Added specifically for `SinkSGD_adv` and `SignSGD_adv`.

*(Note: `Lion_Prodigy_adv`, `Simplified_AdEMAMix`, and heuristic cautious/grams modes have been deprecated in favor of these superior, theoretically-grounded features).*

<details>
<summary><b>Click to see older release notes (v1.2.x - v2.1.x)</b></summary>

### Version 2.1.x
* **New Optimizer:** Added **Signum** (SignSGD with momentum) to the `SignSGD_adv` family.

### Version 2.0.x
* ⚡ **`torch.compile` Support:** Fully implemented for all advanced optimizers. Enable via `compiled_optimizer=True` to heavily fuse and optimize the optimizer step path.
* 📉 **1-Bit Factored Mode:** Vastly improved implementation via `nnmf_factor=True`.
* 🛠️ Broad performance and stability improvements across all optimizers.

### Version 1.2.x
* **Advanced Muon Variants:** Brought the groundbreaking [Muon optimizer](https://kellerjordan.github.io/posts/muon/) into the fold, enriched with features from recent literature.

| Optimizer | Description |
|---|---|
| `Muon_adv` | Advanced Muon implementation featuring CANS, NorMuon, Low-Rank Orthogonalization, and more. |
| `AdaMuon_adv` | Combines Muon's geometry with Adam-like adaptive scaling and sign-based orthogonalization. |

* **Prodigy Speedup:** Prodigy variants are now **50% faster** by eliminating unnecessary CUDA syncs (Shoutout to **@dxqb**!).
* **Stochastic Rounding for BF16:** Parameter updates and weight decay now accumulate in float32 and round once at the end.
* **Cautious Weight Decay:** Implemented for all advanced optimizers ([Paper](https://arxiv.org/abs/2510.12402)).
* **Fused Operations:** Transitioned to fused and in-place operations wherever possible.

</details>

---

## 💡 Core Innovations

*(Documentation expanding on the theory and usage of these features is coming soon!)*