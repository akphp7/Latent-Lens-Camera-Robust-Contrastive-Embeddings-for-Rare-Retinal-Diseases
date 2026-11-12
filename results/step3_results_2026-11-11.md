# Step 3 results (3 training seeds, real Kaggle run, 11 Nov 2026)
Ablation ladder, cam1 never seen. mean +- std over seeds 0,1,2.

| variant | camera leak | val | test | held-cam (cam1) | IDRiD DR | 1-shot | 5-shot | 10-shot |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen DINOv2-S | 0.992 | 0.890 | 0.899 | 0.810 | 0.862 | 0.654 | 0.739 | 0.774 |
| b3 plain SimCLR | 0.976 | 0.887 | 0.902 | 0.816+-0.007 | 0.895 | 0.690 | 0.779 | 0.809 |
| b3 + fn mask only | 0.975 | 0.887 | 0.902 | 0.817+-0.005 | 0.895 | 0.692 | 0.781 | 0.810 |
| b3 + Fourier style swap only | 0.978 | 0.899 | 0.917 | 0.844+-0.002 | 0.892 | 0.704 | 0.806 | 0.835 |
| b3 + fn + fmix (b4_no_nn) | 0.978 | 0.899 | 0.918 | 0.842+-0.005 | 0.894 | 0.705 | 0.806 | 0.836 |
| b4 (+ cross-camera positives) | 0.977 | 0.895 | 0.908 | 0.833+-0.018 | 0.884 | 0.703 | 0.795 | 0.821 |

All cameras seen, 3 seeds (few-shot on cam1 with cam1 seen): frozen 0.654/0.739/0.774; b3 0.691/0.776/0.805; b4 0.716/0.802/0.831; b4_no_nn 0.707/0.807/0.836. Test AUROC: frozen 0.890, b3 0.898, b4 0.915, b4_no_nn 0.915.
