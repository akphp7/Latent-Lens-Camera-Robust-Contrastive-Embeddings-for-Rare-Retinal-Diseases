# Step 2 results (3 training seeds, cam1 never seen, real Kaggle run, 10 Nov 2026)
mean +- std over training seeds 0,1,2 (frozen is deterministic, one value). "camera leak" = balanced accuracy of a linear camera guesser, lower is better.

| variant | camera leak | val AUROC | test AUROC | held-cam (cam1) AUROC | IDRiD DR | IDRiD referable | 1-shot | 5-shot | 10-shot |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen DINOv2-S | 0.992 | 0.890 | 0.899 | 0.810 | 0.862 | 0.885 | 0.654 | 0.739 | 0.774 |
| b3 (plain SimCLR) | 0.976+-0.002 | 0.887+-0.002 | 0.902+-0.002 | 0.816+-0.007 | 0.895+-0.005 | 0.915+-0.003 | 0.690+-0.005 | 0.779+-0.002 | 0.809+-0.002 |
| b4 (all three parts) | 0.977+-0.003 | 0.895+-0.004 | 0.908+-0.003 | 0.833+-0.018 | 0.884+-0.001 | 0.911+-0.005 | 0.703+-0.005 | 0.795+-0.003 | 0.821+-0.002 |
| b4_no_nn (fn + fmix) | 0.978+-0.001 | 0.899+-0.002 | 0.918+-0.002 | 0.842+-0.005 | 0.894+-0.001 | 0.918+-0.001 | 0.705+-0.002 | 0.806+-0.001 | 0.836+-0.002 |
| b4_no_fmix (nn + fn) | 0.961+-0.006 | 0.899+-0.006 | 0.904+-0.006 | 0.826+-0.008 | 0.883+-0.002 | 0.901+-0.002 | 0.695+-0.001 | 0.792+-0.003 | 0.823+-0.003 |
| b4_no_fn (nn + fmix), 1 seed | 0.980 | 0.894 | 0.907 | 0.813 | 0.886 | 0.914 | 0.702 | 0.792 | 0.820 |

Per-camera mean-centering of embeddings (same rule for every model): linear camera leakage falls from about 0.98 to about 0.13 (frozen 0.18), but val/test AUROC falls by 0.01 to 0.03 and held-cam AUROC by about 0.01, few-shot unchanged. It also removes real disease-prevalence differences between cameras, so it is a diagnostic, not a method.
