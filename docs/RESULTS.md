# Full results

All numbers come from real runs on Kaggle GPUs (3 training seeds unless stated) and are copied from the run logs. Macro AUROC unless stated. Linear probes are fitted on RFMiD train. "Frozen" is a single run.

Models: **Frozen** = DINOv2-S, no training · **Plain SimCLR** (`b3`) · **Latent Lens** (`b3_fmix`, SimCLR + Fourier camera-style swap) · **Supervised** (`sup`, BCE on common-class labels) · **Latent Lens + labels** (`b3_fmix_sup`).

## 1. All cameras seen

| | Frozen | Plain SimCLR | Latent Lens | Supervised | Latent Lens + labels |
|:--|--:|--:|--:|--:|--:|
| RFMiD test AUROC | 0.890 | 0.898 | 0.915 | 0.928 | **0.929** |
| IDRiD DR AUROC (RFMiD DR probe) | 0.899 | 0.898 | 0.908 | 0.914 | 0.917 |
| IDRiD referable DR AUROC | 0.927 | 0.924 | 0.938 | 0.940 | 0.941 |
| cam1 few-shot, 1 / 5 / 10 shot | 0.654 / 0.739 / 0.774 | 0.691 / 0.776 / 0.805 | 0.706 / 0.807 / 0.836 | 0.833 / 0.914 / 0.931 | 0.778 / 0.867 / 0.891 |
| Camera leakage (probe accuracy at guessing the camera) | 0.992 | 0.982 | 0.984 | 0.970 | 0.975 |

The supervised cam1 few-shot numbers are **not a fair comparison**: those classes are the ones it was trained on.

## 2. Unseen camera (never used for training or probe fitting)

| Held-out camera | Frozen | Plain SimCLR | Latent Lens | Supervised | Latent Lens + labels |
|:--|--:|--:|--:|--:|--:|
| cam0, held-camera AUROC | 0.836 | 0.865 ± 0.004 | 0.877 ± 0.003 | 0.882 ± 0.003 | 0.877 ± 0.001 |
| cam0, 5-shot on that camera | 0.746 | 0.790 | 0.790 | 0.781 | 0.801 |
| cam1, held-camera AUROC | 0.810 | 0.816 ± 0.006 | 0.844 ± 0.002 | 0.870 ± 0.010 | 0.846 ± 0.007 |
| cam1, 5-shot on that camera | 0.739 | 0.779 | 0.806 | 0.841 | 0.827 |
| cam2, held-camera AUROC | 0.803 | 0.742 ± 0.023 | 0.825 ± 0.012 | 0.861 ± 0.020 | 0.838 ± 0.005 |
| cam2, 5-shot on that camera | 0.725 | 0.659 | 0.665 | 0.811 | 0.746 |

Per-seed difference in held-camera AUROC, Latent Lens minus plain SimCLR:

| Held-out camera | Seed 0 / 1 / 2 | Seeds where Latent Lens is better |
|:--|:--|--:|
| cam0 | +0.013, +0.010, +0.014 | 3/3 |
| cam1 | +0.033, +0.023, +0.028 | 3/3 |
| cam2 | +0.125, +0.074, +0.052 | 3/3 |

Latent Lens minus the frozen model: cam0 +0.037 / +0.045 / +0.044, cam1 +0.032 / +0.036 / +0.034, cam2 +0.031 / +0.030 / +0.005 (better in 3/3 seeds each).

cam2 has only 150 training images and few disease positives, so its numbers are noisier. On cam2 few-shot, Latent Lens is still below the frozen model (0.665 vs 0.725 at 5-shot).

## 3. Common vs rare diseases

Common = at least 30 training positives (15 classes, scored on test). Rare = 8 to 29 training positives and at least 4 positives in val+test (10 classes: ERM, MS, CRVO, AH, AION, PT, RT, EDN, RPEC, MHL, scored on val+test pooled).

| | Frozen | Plain SimCLR | Latent Lens | Supervised | Latent Lens + labels |
|:--|--:|--:|--:|--:|--:|
| Rare, macro AUROC | 0.837 | 0.892 ± 0.006 | 0.894 ± 0.002 | 0.895 ± 0.011 | 0.890 ± 0.009 |
| Common, macro AUROC | 0.890 | 0.898 ± 0.001 | 0.915 ± 0.001 | 0.928 ± 0.006 | 0.929 ± 0.004 |

Paired bootstrap over evaluation images (difference of macro AUROC, 95% interval):

| Comparison | Rare | Common |
|:--|:--|:--|
| Latent Lens − Frozen | +0.058 (+0.023 to +0.099) | +0.025 (+0.010 to +0.040) |
| Latent Lens − Plain SimCLR | +0.002 (−0.022 to +0.027), no difference | +0.017 (+0.006 to +0.030) |
| Latent Lens − Supervised | −0.001 (−0.028 to +0.025), no difference | −0.013 (−0.021 to −0.003), supervised better |
| Latent Lens + labels − Latent Lens | −0.004 (−0.035 to +0.024), no difference | +0.014 (+0.007 to +0.021) |
| Latent Lens + labels − Supervised | −0.005 (−0.031 to +0.019), no difference | +0.001 (−0.007 to +0.011), no difference |

**Rare few-shot** (K positive + K negative training images, 10 fixed support sets shared by all models, scored on val+test):

| K | Frozen | Plain SimCLR | Latent Lens | Supervised | Latent Lens + labels |
|--:|--:|--:|--:|--:|--:|
| 3 | 0.714 ± 0.042 | 0.763 ± 0.035 | 0.756 ± 0.037 | 0.767 ± 0.046 | 0.771 ± 0.045 |
| 5 | 0.737 ± 0.019 | 0.789 ± 0.017 | 0.789 ± 0.024 | 0.811 ± 0.015 | 0.811 ± 0.017 |
| 10 | 0.784 ± 0.019 | 0.851 ± 0.011 | 0.849 ± 0.012 | 0.863 ± 0.018 | 0.868 ± 0.014 |

Latent Lens minus frozen, per (class, support set) pair: K=3 +0.042 (better in 68% of pairs, interval +0.002 to +0.080); K=5 +0.052 (71%, +0.007 to +0.091); K=10 +0.065 (79%, +0.026 to +0.103).

**Per rare class** (AUROC, mean over seeds; 6 to 17 positives per class in val+test, so single classes are noisy):

| Class | Train positives | Eval positives | Frozen | Plain SimCLR | Latent Lens | Supervised | Latent Lens + labels |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ERM | 14 | 12 | 0.666 | 0.870 | 0.794 | 0.826 | 0.913 |
| MS | 15 | 12 | 0.745 | 0.839 | 0.795 | 0.892 | 0.785 |
| CRVO | 28 | 17 | 0.969 | 0.974 | 0.949 | 0.971 | 0.960 |
| AH | 16 | 9 | 0.938 | 0.980 | 0.962 | 0.942 | 0.964 |
| AION | 17 | 9 | 0.889 | 0.689 | 0.858 | 0.913 | 0.906 |
| PT | 11 | 6 | 0.765 | 0.957 | 0.942 | 0.840 | 0.846 |
| RT | 14 | 11 | 0.982 | 0.968 | 0.968 | 0.958 | 0.971 |
| EDN | 15 | 9 | 0.832 | 0.909 | 0.951 | 0.920 | 0.921 |
| RPEC | 22 | 10 | 0.764 | 0.774 | 0.783 | 0.799 | 0.721 |
| MHL | 11 | 6 | 0.816 | 0.958 | 0.935 | 0.890 | 0.916 |

**Not evaluated** (fewer than 8 training positives): TV (6), ST (5), RP (6), CWS (3), CB (1), PRH (2), MNF (3), CRAO (2), TD (3), CME (4), PTCR (5), CF (3), VH (1), MCA (1), VS (1), BRAO (2), PLQ (1), HPED (1), CL (1). Two more conditions have no training positives at all.

## 4. Ablation ladder (cam1 never seen, 3 seeds)

| Variant | Camera leakage | Test | Held-camera (cam1) | IDRiD DR | 1 / 5 / 10-shot |
|:--|--:|--:|--:|--:|--:|
| Frozen DINOv2-S | 0.992 | 0.899 | 0.810 | 0.862 | 0.654 / 0.739 / 0.774 |
| Plain SimCLR | 0.976 | 0.902 | 0.816 ± 0.007 | 0.895 | 0.690 / 0.779 / 0.809 |
| + false-negative mask only | 0.975 | 0.902 | 0.817 ± 0.005 | 0.895 | 0.692 / 0.781 / 0.810 |
| **+ Fourier style swap only (Latent Lens)** | 0.978 | 0.917 | **0.844 ± 0.002** | 0.892 | 0.704 / 0.806 / 0.835 |
| + mask + swap | 0.978 | 0.918 | 0.842 ± 0.005 | 0.894 | 0.705 / 0.806 / 0.836 |
| + cross-camera positives (full "b4") | 0.977 | 0.908 | 0.833 ± 0.018 | 0.884 | 0.703 / 0.795 / 0.821 |

Reading: the whole gain comes from the Fourier camera-style swap. False-negative masking does nothing and cross-camera positives slightly hurt on an unseen camera, so neither is in the final model. Camera leakage is not reduced by any variant.

## 5. Trust layer

**Calibration of per-class probes** (ECE on test, lower is better; temperature fitted on validation):

| | Before | After temperature scaling |
|:--|--:|--:|
| Frozen | 0.039 | 0.030 |
| Plain SimCLR | 0.037 | 0.029 |
| Latent Lens | 0.035 | **0.024** |

**Retrieval** (do the 5 nearest training images share a disease with the query?):

| | All gallery | Cross-camera gallery only | Unseen cam1 queries | Same-camera share of top-5 (chance 0.65) |
|:--|--:|--:|--:|--:|
| Frozen | 0.423 | 0.296 | 0.452 | 0.878 |
| Plain SimCLR | 0.528 | 0.402 | 0.514 | 0.892 |
| Latent Lens | **0.538** | **0.417** | **0.521** | 0.843 |

Retrieval is still biased towards the same camera, which is why the demo shows the camera next to every retrieved case.

**Unfamiliar-image score** (AUROC for flagging images from an unseen source; higher is better; 0.5 is chance):

| Score | Unseen camera cam1 (Frozen / Plain / Latent Lens) | IDRiD vs RFMiD |
|:--|:--|:--|
| kNN distance | 0.627 / 0.632 / 0.625 | 0.399 / 0.507 / 0.477 |
| Mahalanobis distance (used in the demo) | 0.724 / 0.662 / 0.712 | 0.39 to 0.51 (at or below chance) |

The flag is a **rough screening aid**, not a reliable out-of-distribution detector. IDRiD shares cam1's camera type, which partly explains the IDRiD result. As an image-quality flag it works for blur, JPEG and noise (Latent Lens 0.76 to 0.91 AUROC) but fails for dark or low-contrast images (about 0.5 to 0.6).

**Robustness to image-quality shifts** (RFMiD test, macro AUROC; probes trained on clean images):

| Condition | Frozen | Plain SimCLR | Latent Lens |
|:--|--:|--:|--:|
| clean | 0.890 | 0.899 | 0.916 |
| blur σ=2 | 0.801 | 0.869 | 0.873 |
| blur σ=4 | 0.690 | 0.801 | 0.759 |
| low resolution 4× | 0.802 | 0.868 | 0.864 |
| JPEG q20 | 0.751 | 0.814 | 0.820 |
| dark ×0.4 | 0.878 | 0.891 | 0.907 |
| low contrast ×0.5 | 0.876 | 0.891 | 0.910 |
| noise sd 15 | 0.764 | 0.816 | 0.812 |

Fine-tuning gives most of the robustness. Latent Lens is better on brightness, contrast and JPEG; plain SimCLR is slightly better on heavy blur.

## 6. Efficiency

| | Latent Lens (DINOv2-S) | ResNet50 (reference only) |
|:--|--:|--:|
| Parameters | 21.6 M | 23.5 M |
| GFLOPs per image | 11.0 | 8.2 |
| CPU latency (2 threads, batch 1) | 85 ms | 82 ms |
| GPU latency (T4, batch 1) | 6.2 ms | 6.8 ms |
| Trainable parameters | 7.1 M | n/a |
| Training time per run | about 6.3 min (Kaggle GPU) | n/a |

There is **no inference-speed advantage**. The saving is in adaptation: a few minutes of training on a free GPU, with no labels for the label-free variant.

## 7. How the final model was chosen

Rule written before step 8 was run: the hybrid (Latent Lens + labels) becomes the final deployed model only if it is not worse than Latent Lens on rare classes and better on common classes and held-out cameras. Result: rare 0.890 vs 0.894 (tie), common 0.929 vs 0.915 (clearly better), held-out cameras tie on cam0 (0.877 both) and better on cam1 (+0.002) and cam2 (+0.013). The rule was met narrowly, so the demo uses the hybrid. The label-free Latent Lens remains our headline method because it needs no labels and carries the camera-robustness result.
