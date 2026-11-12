# Latent Lens: trust layer and efficiency

## Unfamiliar-image score (kNN distance in embedding space), AUROC, higher = better

| model | IDRiD vs RFMiD test (all cameras seen) | unseen camera cam1 vs seen test (cam1 held out) |
|:--|--:|--:|
| Frozen DINOv2-S | 0.399 | 0.627 |
| Plain SimCLR | 0.507+-0.015 | 0.632+-0.005 |
| Latent Lens | 0.477+-0.017 | 0.625+-0.009 |

## Calibration of per-class probes (ECE on test, lower = better; temperature fitted on validation)

| model | ECE before | ECE after temperature scaling |
|:--|--:|--:|
| Frozen DINOv2-S | 0.039 | 0.030+-0.000 |
| Plain SimCLR | 0.037+-0.001 | 0.029+-0.001 |
| Latent Lens | 0.035+-0.001 | 0.024+-0.000 |

## Retrieval: do the 5 nearest training images share a disease with the query? (precision@5)

| model | all gallery | cross-camera gallery only | same-camera share of top-5 (base rate 0.65) | unseen camera cam1 queries |
|:--|--:|--:|--:|--:|
| Frozen DINOv2-S | 0.423 | 0.296 | 0.878+-0.000 | 0.452 |
| Plain SimCLR | 0.528+-0.002 | 0.402+-0.001 | 0.892+-0.005 | 0.514+-0.010 |
| Latent Lens | 0.538+-0.003 | 0.417+-0.007 | 0.843+-0.011 | 0.521+-0.016 |

## Robustness to image-quality shifts (RFMiD test, macro AUROC; probes trained on clean images)

| condition | Frozen DINOv2-S | Plain SimCLR | Latent Lens |
|:--|--:|--:|--:|
| clean | 0.890 | 0.899 | 0.916 |
| blur_sigma2 | 0.801 | 0.869 | 0.873 |
| blur_sigma4 | 0.690 | 0.801 | 0.759 |
| low_res_4x | 0.802 | 0.868 | 0.864 |
| jpeg_q20 | 0.751 | 0.814 | 0.820 |
| dark_x0.4 | 0.878 | 0.891 | 0.907 |
| low_contrast_x0.5 | 0.876 | 0.891 | 0.910 |
| noise_sd15 | 0.764 | 0.816 | 0.812 |

Degraded-vs-clean detection with the unfamiliar-image score (AUROC):

| condition | Frozen DINOv2-S | Plain SimCLR | Latent Lens |
|:--|--:|--:|--:|
| blur_sigma2 | 0.886 | 0.666 | 0.760 |
| blur_sigma4 | 0.995 | 0.862 | 0.911 |
| low_res_4x | 0.932 | 0.680 | 0.766 |
| jpeg_q20 | 0.978 | 0.801 | 0.857 |
| dark_x0.4 | 0.644 | 0.585 | 0.580 |
| low_contrast_x0.5 | 0.596 | 0.598 | 0.510 |
| noise_sd15 | 0.957 | 0.713 | 0.854 |

## Efficiency (224x224, one image)

| model | params (M) | GFLOPs | CPU ms | GPU ms |
|:--|--:|--:|--:|--:|
| DINOv2-S backbone (Latent Lens, frozen DINOv2-S, SimCLR) | 21.6 | 11.0 | 85 (2 threads) | 6.2 |
| ResNet50 (reference only) | 23.5 | 8.2 | 82 (2 threads) | 6.8 |

Trainable during fine-tuning: 7.1M parameters.
Training time per run (median of 3 runs): 6.3 min on the Kaggle GPU.