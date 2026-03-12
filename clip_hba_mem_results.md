---
title: "CLIP-HBA-Mem: Preliminary Memorability Results"
author: "Marren Jenkins & David Tovar — Vanderbilt University"
date: "March 2026"
geometry: margin=0.75in
pdf-engine: xelatex
fontsize: 11pt
linestretch: 1.3
colorlinks: true
linkcolor: blue
---

## Overview

We present preliminary results from **CLIP-HBA-Mem**, a memorability prediction model built on top of CLIP-HBA-Behavior (Zhao et al., 2025; [arXiv:2502.04658](https://arxiv.org/abs/2502.04658)). CLIP-HBA-Behavior fine-tunes CLIP ViT-L/14 on SPoSE embeddings from the THINGS dataset to align the model's internal representations with the representations derived from human behavioral similarity judgments. Our central question: **does behavioral alignment training on THINGS confer advantages on downstream memorability prediction?**

To test this, we freeze the CLIP-HBA-Behavior backbone and train a lightweight MLP regression head to predict memorability scores. We compare against an identical MLP head trained on a frozen vanilla CLIP ViT-L/14 backbone, isolating the contribution of behavioral alignment specifically. Notably, a frozen vanilla CLIP + MLP baseline on LaMem does not appear to have been reported in the published literature, making both conditions novel contributions.

---

## Experimental Setup

**Backbone models**

- *CLIP ViT-L/14 (frozen)* — OpenAI pretrained CLIP ViT-L/14, no further fine-tuning; serves as the generic baseline
- *CLIP-HBA-Behavior (frozen)* — OpenAI pretrained CLIP ViT-L/14, additionally fine-tuned on SPoSE behavioral embeddings from THINGS 

**MLP head architecture**

We use the PerceptCLIP MLP head architecture for both conditions. The MLP maps the image encoder's CLS token embedding to a scalar memorability score. Training uses MSE loss on memorability scores.

| Hyperparameter | Search Range | Winning Value |
|---|---|---|
| Learning rate | `1e-4` `5e-5` `1e-5` | `1e-5` |
| Hidden dimensions | `(512, 256)` `(256, 128)` `(512, 256, 128)` `(256)` | `(256, 128)` |
| Dropout | `.3` `.5` | `.5` |
| Batch size | `32` `64` | `32` |
| Max epochs | `300` | — |
| Early stopping patience | `12 epochs` | — |

A **hyperparameter search** was conducted for both vanilla CLIP and CLIP-HBA-Behvior via grid search over 48 configurations on fold 1 of LaMem, using validation SRCC as the selection criterion. The winning configuration for each model was then applied to for all subsequent runs.

**Training data:** LaMem (Khosla et al., 2015) — ~58,000 images with memorability scores, 5-fold cross-validation with 10,000-image held-out test sets per fold. Full training had a maximum epoch range of 300, but early stopping was implemented with a patience of 20 epochs.

---

## Results

### LaMem — 5-Fold Cross-Validated Performance

The primary benchmark. Human consistency ceiling on LaMem is $\rho$ = 0.68.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean ± SD** |
|---|---|---|---|---|---|---|
| MemNet (Khosla et al., 2015) | — | — | — | — | — | 0.64 |
| AMNet (Fajtl et al., 2018) | — | — | — | — | — | 0.677 |
| ResMem (Needell & Bainbridge, 2022) | — | — | — | — | — | 0.64 |
| ViTMem (Hagen & Espeseth, 2023) | — | — | — | — | — | $\geq 0.64$ |
| PerceptCLIP (Zalcher et al., 2025) | — | — | — | — | — | SOTA |
| *Human ceiling* | — | — | — | — | — | *0.68* |
| **CLIP ViT-L/14 (frozen) + MLP** | `[X]` | `[X]` | `[X]` | `[X]` | `[X]` | **`[X ± X]`** |
| **CLIP-HBA-Mem (frozen) + MLP** | 0.70 | `[X]` | `[X]` | `[X]` | `[X]` | **`[X ± X]`** |

Reported metric: Spearman Rank Correlation Coefficient (SRCC). Prior model results are from published papers; experimental setups vary slightly.

### Out-of-Distribution Generalization

Both models evaluated zero-shot (no fine-tuning on target dataset; LaMem-trained MLP head applied directly).

| Dataset | Human Ceiling | ResMem | ViTMem | CLIP (frozen) + MLP | **CLIP-HBA-Mem** |
|---|---|---|---|---|---|
| MemCat | 0.78 | — | — | `[X]` | **0.71** |
| THINGS Memorability | 0.449 | 0.22 | 0.30 | `[X]` | **0.30** |
| PASCAL-S (images) | — | 0.36 | 0.44 | `[X]` | `[X]` |

MemCat: Goetschalckx & Wagemans (2019); 10,000 images across 5 semantic categories. THINGS Memorability: Kramer, Hebart, Baker & Bainbridge (2023); 26,107 object images, 1M+ ratings. PASCAL-S: Dubey, Peterson, Khosla, Yang, & Ghanem (2015). 

---

## Key Takeaways

- **CLIP-HBA-Mem exceeds the LaMem human consistency ceiling (0.68) on fold 1** with a frozen backbone and no memorability-specific pretraining, suggesting that behavioral alignment on THINGS provides strong downstream signal for memorability.
- **The delta between CLIP-HBA-Mem and vanilla CLIP** on LaMem (`[CLIP-HBA SRCC]` vs. `[CLIP SRCC]`) isolates the contribution of behavioral alignment specifically, above and beyond what generic image-text pretraining provides.
- **On THINGS memorability**, CLIP-HBA-Mem matches the current ViT-based state of the art (ViTMem, $\rho$ = 0.30) despite no memorability-specific training, and significantly outperforms ResMem ($\rho$ = 0.22). This is particularly notable given that CLIP-HBA-Behavior was trained on THINGS images, suggesting the behavioral alignment signal captures object-level semantic structure relevant to memorability.
- **No frozen vanilla CLIP + MLP baseline on LaMem has been previously reported** in the memorability literature. This condition serves as a novel empirical reference point for future work using CLIP-derived models.

---

## Future directions

1. Use transfer function to go from 768-dim back to 66-dim. See which weights from 768 map onto 66 dimensions. Linear mapping, weighting matrix, RDMs. Then we can examine which of the 66 dimensions are most 
2. How does CLIP-HBA-Mem perform on face memorability?
3. How does CLIP-HBA-Mem perform on low-memorability images? What is the lower bound of predicted memorability within the LaMem dataset?
4. At what point in time is memorability encoded? We can look at this using the CLIP-HBA-MEG model.(MEGMem)

5. **What is the relationship between behavioral alignment (SPoSE dimensions) and memorability features?** The THINGS dataset and its SPoSE embeddings allow for interpretable probing of which object dimensions drive the memorability signal.
6. **How does CLIP-HBA-Mem perform on face memorability?** Prior work (Younesi & Mohsenzadeh, 2024) shows generic models fail on face images. Our backbone's behavioral training on diverse THINGS objects may or may not transfer.
7. **Can behavioral alignment improve memorability *modification*?** If the backbone better captures what makes objects memorable, it may be useful for generative applications as well.

---

## References

Goetschalckx, L. & Wagemans, J. (2019). MemCat: A new category-based image set quantified on memorability. *PeerJ*.

Hagen, T. & Espeseth, T. (2023). Image memorability prediction with vision transformers. *arXiv:2301.08647*.

Khosla, A., Raju, A. S., Torralba, A. & Oliva, A. (2015). Understanding and predicting image memorability at a large scale. *ICCV*.

Kramer, M. A., Hebart, M. N., Baker, C. I. & Bainbridge, W. A. (2023). The features underlying the memorability of objects. *Science Advances*.

Needell, C. D. & Bainbridge, W. A. (2022). Embracing new techniques in deep learning for estimating image memorability. *Computational Brain & Behavior*.

Tovar, D. et al. (2025). CLIP-HBA-Behavior. *arXiv:2502.04658*.

Zalcher, A. et al. (2025). Don't judge before you CLIP: A unified approach for perceptual tasks. *TMLR*.
