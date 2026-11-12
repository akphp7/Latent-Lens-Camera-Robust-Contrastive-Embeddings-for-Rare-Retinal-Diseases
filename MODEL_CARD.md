# Model card: Latent Lens

**Not a medical device. Research prototype built for a hackathon (ZEISS India BeyondCode'26). Not validated for clinical use. Do not use it to make care decisions.**

## Model details

- **Name:** Latent Lens (final deployed variant: `b3_fmix_sup`; label-free variant: `b3_fmix`)
- **Author:** Alok Kumar (group Latent Lens, NIT Rourkela), 2026
- **Architecture:** DINOv2-small (`vit_small_patch14_dinov2.lvd142m`, 21.6M parameters, 224×224 input) with a projection head used only for training. The last 4 transformer blocks, the final norm and the head are fine-tuned (7.1M trainable parameters). Output: a 384-dimensional embedding per image.
- **Training objective:** SimCLR-style contrastive loss (τ = 0.2) with retina-preserving augmentations and a **Fourier camera-style swap** between images from different cameras. The final variant adds a supervised-contrastive term using labels of 14 common conditions.
- **Training:** 400 steps, batch 96, learning rate 1e-4, camera-balanced sampling, about 6 minutes on one Kaggle GPU. Warm-started from public DINOv2 weights.
- **On top of the embedding (not part of the encoder):** per-class logistic-regression probes with temperature scaling, a Mahalanobis-distance unfamiliar-image score, simple image-quality checks (contrast, sharpness) and nearest-neighbour retrieval.

## Intended use

- Research and teaching: studying how fundus-image embeddings behave across cameras, few-shot learning for rare conditions, and simple trust signals.
- A demo of "similar known cases + calibrated scores + warnings".

## Out of scope

- Any clinical, diagnostic, screening or triage use.
- Use on images from populations, cameras or conditions far from RFMiD without fresh evaluation.
- Any condition with fewer than 8 training positives (these were not evaluated at all).

## Data

- **Training and probe fitting:** RFMiD (Retinal Fundus Multi-Disease Image Dataset), official split 1,920 train / 640 validation / 640 test, 46 conditions, three camera types (cam0 2144×1424, cam1 4288×2848, cam2 2048×1536; camera inferred from image size).
- **Transfer test:** IDRiD (diabetic retinopathy grading), evaluated through an RFMiD diabetic-retinopathy probe. Same image size as RFMiD cam1.
- Contrastive training sees only RFMiD **train** images and, for held-camera experiments, never the held-out camera. The supervised-contrastive term uses train labels only.
- Datasets and the DINOv2 weights are not redistributed here and keep their own licences (RFMiD and IDRiD: CC BY 4.0).

## Metrics (all from our own runs; 3 seeds; full tables in `docs/RESULTS.md`)

| Measure | Frozen DINOv2-S | Latent Lens (label-free) | Latent Lens + labels (final) |
|:--|--:|--:|--:|
| RFMiD test AUROC, common classes (15) | 0.890 | 0.915 | 0.929 |
| Rare classes (10), macro AUROC | 0.837 | 0.894 | 0.890 |
| Held-out cam0 / cam1 / cam2 AUROC | 0.836 / 0.810 / 0.803 | 0.877 / 0.844 / 0.825 | 0.877 / 0.846 / 0.838 |
| IDRiD DR AUROC (transfer) | 0.899 | 0.908 | 0.917 |
| Calibration error after temperature scaling (ECE) | 0.030 | 0.024 | not measured |

A fully supervised fine-tune scores 0.928 (common), 0.895 (rare) and 0.882 / 0.870 / 0.861 on the held-out cameras, so it is **better on unseen cameras**.

## Limitations and risks

- **Camera information remains in the embedding** (a probe guesses the camera with 0.97 to 0.98 accuracy). The method improves robustness to an unseen camera; it does not make embeddings camera-invariant.
- **Small data.** RFMiD has 3,200 images; cam2 has only 150 training images. Rare-class results rest on 6 to 17 evaluation positives per class and are noisy.
- **Unfamiliar-image flag is weak** (about 0.7 AUROC for an unseen camera; at or below chance for IDRiD vs RFMiD). About 5% of known test images are flagged by design (95th-percentile threshold). Treat it as a hint, not a detector.
- **Retrieval is biased towards the same camera**, so neighbours can look similar for camera reasons. The demo shows the camera for every retrieved case.
- **Development results.** The final variant was chosen after seeing results on the same splits; there is no untouched final test set.
- **No speed advantage** at inference compared with a ResNet50.
- **Fairness and population coverage were not evaluated.** RFMiD comes from a limited set of centres and devices; we have no patient-level demographic information.
- Image-quality checks fail on dark or low-contrast images (about 0.5 to 0.6 AUROC).

## How to cite or credit

Alok Kumar, "Latent Lens: Camera-Robust Contrastive Embeddings for Rare Retinal Diseases", ZEISS India BeyondCode'26 submission, 2026. See `README.md` for the datasets and methods we build on.
