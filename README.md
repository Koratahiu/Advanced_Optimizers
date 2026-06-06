# Advanced Optimizers (AIO)

A comprehensive, all-in-one collection of optimization algorithms for deep learning, designed for **maximum efficiency**, **minimal memory footprint**, and **superior performance** across diverse model architectures and training scenarios.

[![PyPI](https://img.shields.io/pypi/v/adv_optm)](https://pypi.org/project/adv_optm/)

## 🔥 What's New

### In 2.4.x:

This update introduces a whole refactor of the library with many new features and changes:

- New optimizers state mode option (`state_precision`) with many precision settings for the optimizer states: rank-2 factored mode (`factored`), full FP32 (`fp32`), BF16 with Stochastic Rounding (`bf16_sr`), int8/uint8 with Stochastic Rounding (`int8_sr`), FP16 (`fp16`)
- Added new powerful optimizer: SinkSGD_adv.
- Added spectral scaling option to all optimizers, achieving width/rank invariant updates.
- Added Nesterov momentum (`nesterov`) and its coef (`nesterov_coef`) to all optimizers.
- Added centered weight decay (`centered_wd`), to pull the weights toward their pre-train state (anchor)
    - anchor precision can be changed to save memory (`centered_wd_mode`): full, float8, int8, int4
- Added Fisher Weight Decay option for Adam variants (`fisher_wd`).
    - Paper: [FAdam...](https://arxiv.org/abs/2405.12807)
- Added Factored Second Moment option for Adam variants (`factored_2nd`). This works alongside any `state_precision` setting.
- Added Geometric Weight Decay for SinkSGD_adv and SignSGD_adv.
- Added new powerful mode: variance normalized momentum (`normed_momentum`). Which applies the optimizer normalization before the momentum (also called as Normalization then momentum NtM)
    - For: AdamW_adv, SignSGD_adv, SinkSGD_adv.
- Added Variance/Confidence Preconditioning (`snr_cond`) for SignSGD_adv, SinkSGD_adv.
    - Only works with `normed_momentum`.
    - Technical reports: [AASS](https://koratahiu.github.io/aass/), and [sink-v](https://koratahiu.github.io/sink-v/).
- Added Adaptive Stochastic Sign with L_inf preconditioning (`stochastic_sign`) for SignSGD_Adv and Lion_adv.
- Improved CANS (`accelerated_ns`) for Muon variants, by integrating dynamic lower bound.
- Removed Simplified_AdEMAMix optimizer and its settings in other optimizers, they are now replaced by Nesterov momentum and its coef. Which is better and less hard to tune.
- Removed cautious and grams modes, as they were heuristic and not working well.
- Removed optimizers: Lion_Prodigy_adv, and Simplified_AdEMAMix.

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

This library integrates multiple state-of-the-art optimization techniques validated through extensive research and practical training.

---

