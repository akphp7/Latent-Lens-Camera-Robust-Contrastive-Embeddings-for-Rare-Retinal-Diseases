# Step 1 results (seed 0, real Kaggle run, 9 Nov 2026)
Numbers from notebook "Latent Lens 1". Rows with "held out cam1": cam1 was never used for training or probe fitting.

| run | held out | camera leak | val AUROC | test AUROC | held-cam AUROC | IDRiD DR | IDRiD referable | 1-shot | 5-shot | 10-shot |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen_s0 | - | 0.992 | 0.878 | 0.890 | - | 0.899 | 0.927 | 0.654 | 0.739 | 0.774 |
| b3_s0 | - | 0.983 | 0.892 | 0.899 | - | 0.904 | 0.928 | 0.691 | 0.774 | 0.804 |
| b4_s0 | - | 0.987 | 0.901 | 0.916 | - | 0.908 | 0.936 | 0.717 | 0.802 | 0.834 |
| frozen_hold-cam1_s0 | cam1 | 0.992 | 0.890 | 0.899 | 0.810 | 0.862 | 0.885 | 0.654 | 0.739 | 0.774 |
| b3_hold-cam1_s0 | cam1 | 0.975 | 0.888 | 0.900 | 0.815 | 0.899 | 0.915 | 0.685 | 0.778 | 0.807 |
| b4_hold-cam1_s0 | cam1 | 0.980 | 0.892 | 0.906 | 0.813 | 0.883 | 0.910 | 0.703 | 0.791 | 0.819 |
| b4_no_nn_hold-cam1_s0 | cam1 | 0.977 | 0.901 | 0.920 | 0.838 | 0.893 | 0.917 | 0.707 | 0.806 | 0.837 |
| b4_no_fmix_hold-cam1_s0 | cam1 | 0.957 | 0.906 | 0.907 | 0.835 | 0.885 | 0.901 | 0.697 | 0.791 | 0.821 |
| b4_no_fn_hold-cam1_s0 | cam1 | 0.980 | 0.894 | 0.907 | 0.813 | 0.886 | 0.914 | 0.702 | 0.792 | 0.820 |

Single training seed. Few-shot "+-" in the logs is spread over support sets, not over training seeds.
