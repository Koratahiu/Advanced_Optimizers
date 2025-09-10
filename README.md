# Advanced Optimizers

This repo introduces a new family of highly efficient, lightweight yet powerful optimizers, born from extensive research into recent academic literature and validated through practical training runs across diverse models.

---

### Install

`pip install adv_optm`

---

### Theory (Inspired by SMMF)

Based primarily on:  
**[SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization](https://arxiv.org/abs/2412.08894)**

The core innovation:
- Uses fast, non-negative matrix factorization (rank 1, à la Adafactor), but **reconstructs the full state before each update** to preserve momentum accuracy, then re-factors afterward (factor → reconstruct → update → factor cycle).
- For the *signed first moment*, we split into **sign + absolute value**:
  - Sign is stored as **1-bit state** via bitwise ops (SMMF originally used 8-bit with 7 bits wasted).
  - Absolute value goes through the factor/reconstruct cycle using two factored vectors + the signed state.
- Final storage: **four factored vectors + one 1-bit sign**.
- Updates behave like full-state Adam but with drastically reduced memory.

> ✅ **TL;DR**: Lightweight, strong, memory-efficient optimizer.

---

### Memory Cost

- **Adopt_Factored** for full SDXL finetune: **328 MB** (4 small vectors + 1-bit state)
- **Adopt_Factored with AdEMAMix** for full SDXL finetune: **625 MB** (6 small vectors + two 1-bit states)
> SDXL is 6.5GB model.

---

### ⏱️ Speed (my tests in SDXL - BS 4)

- **Adopt_Factored**: ~10s/it
- **Adopt_Factored with AdEMAMix**: ~12s/it
- **Adafactor**: ~8.5s/it  
→ Overhead from compression/reconstruction cycles.
→ It's faster than [MLorc](https://arxiv.org/abs/2506.01897) (~12s/it), which uses RSVD compression, and should be the fastest momentum compression (AFAIK).

---

### 📈 Performance

- **Better than Adafactor, and CAME factorzation methods**
- **Comparable or identical to Adam** (see SMMF paper results)

---

### Available Optimizers (all support `Factored` toggle)

Set `Factored=False` to disable factorization and run as a full uncompressed optimizer (like vanilla Adam).

1. **Adam**
2. **Prodigy**
3. **Adopt**

---

### Bonus Features (Built-in)

- **Fused Backward Pass**

- **Stochastic Rounding (SR)**: Improves quality and convergence for **BF16 training**.

- **[AdEMAMix](https://arxiv.org/abs/2409.03137)**  
  → This adds a second, slow-moving EMA, which is combined with the primary momentum to stabilize updates, especially during long runs of full finetuning.
  → A higher value of beta3 (e.g., 0.9999) gives the EMA a longer memory, making it more stable but slower to adapt. A lower value (e.g., 0.999) is often better for shorter training runs (2k-4k steps).
  → When `factored` is true, it compresses the new momentum in the same way as the first moment (1-bit state + 2 vectors). However, this introduces noticeable overhead as we are compressing/reconstructing a third state each step.
  ⚠️ **Note**: AdEMAMix updates are more aggressive than normal Adam/Adopt, so use a smaller LR than usual (2x-5x smaller LR exploded in my tests).
  ⚠️ **Note**: The factored AdEMAMix is **Experimental** (as it needs more tests and validation, but it should work). Also, Adopt with AdEMAMix is **Experimental** (as Adopt normalizes the gradients for the momentum).

- **[`atan2` smoothing & scaling](https://github.com/lucidrains/adam-atan2-pytorch)**  
  → Robust `eps` replacement (no tuning!) + built-in gradient clipping  
  → *Ideal for ADOPT* (which normally needs higher `eps` and clipping), so `use_atan2` is all-in-one for it.

- **[OrthoGrad](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability)**  
  → Removes gradient component parallel to weights → prevents "naïve loss minimization" (NLM) → reduces natural overfitting  
  → Perfect for fine-tuning the direction of existing features (e.g., full finetune or training a trained LoRA) without weight decay erasing prior knowledge.
  ⚠️ **Note**: OrthoGrad introduces **~33% time overhead**, so take this into account.

- **[Grams: Gradient Descent with Adaptive Momentum Scaling](https://github.com/Gunale0926/Grams)**  
  → Eliminates the need for 1-bit momentum sign storage by using the **sign of gradients** for the first moment.
  ⚠️ **Not recommended for small batch sizes**: gradients are too noisy, which can destabilize momentum (tested for Prodigy and it made the optimizer slower to find the LR or converge in BS 4).

### Other Notes

- **Adopt** skips the first step (only initializes the states) and has built-in clipping (sticking to the original optimizer), but we skip both of these when you enable `use_atan2`; as the optimizer becomes scale-invariant and the values of the states won't cause any issues or instability.

- When `use_atan2` is True, `eps` will be ignored and you should also disable any gradient clipping.

- I don't recommend using **OrthoGrad** for training LoRA or embeddings, as their weights are zero-initialized and using weight decay for them should be safe and also beneficial (OrthoGrad is intended for fine-tuning pretrained models with no weight decay).

---
