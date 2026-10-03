# Task 3 — CycleGAN photo↔Monet · ruchika_chaurasia (lineup A, run: lc5_d4_100)

## What I built
Generators: resnet (ngf 64, 9 residual blocks); discriminators: PatchGAN ndf 64, 4 stride-2 layers; instance norm.

**Hyper-parameters:** GAN loss lsgan, λ_cycle 5.0, λ_identity 0.0 (×λ_cycle), Adam lr 0.0002 β1 0.5, batch 1, 100 epochs × 1000 steps (linear decay from epoch 50), image pool 50, AMP True, DiffAugment none, generator EMA off, image 256 (load 286, random crop + flip), seed 8283.

## Why (design justification)
* Same as lc5_100 plus a 4-layer PatchGAN (receptive field ~142x142 instead of 70x70; D params 2.8M -> 7.0M).
* Hypothesis: a 70x70 D only checks local texture; larger patches also judge composition and global colour
  palette, which is where generated Monets still differ from real ones (low precision / density).
* Comparing with lc5_100 isolates the discriminator change.
* Selection rule unchanged: held-out B2A FID / precision on the 500 held-out photos, not the Kaggle score.


## Metrics — both directions (checkpoint `636151d0023c`)
| metric | A2B | B2A |
|---|---|---|
| content_cosine | 0.7277 | 0.5806 |
| cycle_l1 | 0.1165 | 0.1201 |
| fid | 107 | 216.7 |
| gen_coverage | 0.496 | 0.04667 |
| gen_density | 0.548 | 0.0068 |
| gen_precision | 0.56 | 0.03 |
| gen_recall | 0.322 | 0.32 |
| kid_mean | 0.04108 | 0.1743 |
| kid_std | 0.003573 | 0.007481 |
| l1_input_vs_translation | 0.3009 | 0.3945 |
| lpips_input_vs_translation | 0.4877 | 0.6772 |
| n_real_reference | 500 | 300 |
| n_translated | 300 | 500 |

## Training, stability and cost
| metric | value |
|---|---|
| params_G_AB | 11378179 |
| params_G_BA | 11378179 |
| params_D_A | 6959553 |
| params_D_B | 6959553 |
| params_total | 36675464 |
| inference_images_per_sec | 554.5 |
| final_epoch_G_adv_AB | 0.7102 |
| final_epoch_G_adv_BA | 0.9932 |
| final_epoch_D_A | 0.0005807 |
| final_epoch_D_B | 0.07739 |
| final_epoch_cycle_loss_A | 0.5237 |
| final_epoch_cycle_loss_B | 0.5299 |
| final_epoch_identity_loss_A | 0 |
| final_epoch_identity_loss_B | 0 |
| gradnorm_G_mean | 15.67 |
| gradnorm_G_max | 90.65 |
| gradnorm_D_mean | 10.99 |
| gradnorm_D_max | 56.75 |
| gradnorm_G_nonfinite_steps | 43 |
| gradnorm_D_nonfinite_steps | 1 |
| nan_count | 0 |
| amp_overflow_skips | 43 |
| training_time_min | 294.1 |
| train_steps | 100000 |
| epochs | 100 |
| train_image_pairs_per_sec | 5.666 |
| peak_memory_mb | 1428 |
| hardware | NVIDIA GeForce RTX 5090 / Intel(R) Core(TM) Ultra 9 285K |

## Kaggle
Public score: None · private score: None · team rank: None
(`submission.csv` from the official script on this member's `pred_A2B/` + `pred_B2A/`, checkpoint `636151d0023c`).

## Human audit (30 fixed photos, 2 blinded raters)
TODO — run the blinded audit (task3_gan/audit/README.md)

## Training behaviour (write in your own words)
TODO — describe the loss curves (`outputs/loss_curves.png`) and progress grids (`outputs/progress/`):
did cycle loss plateau, did D overpower G, any spikes/NaNs, when did Monet-like texture appear?

## Evidence
Loss log `outputs/loss_log.csv` · curves `outputs/loss_curves.png` · progress grids `outputs/progress/` ·
cycle figures `outputs/figures/` · predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` · raw log `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_lc5_d4_100_20261002_213706.log`.

## Hardware
{"cpu": "Intel(R) Core(TM) Ultra 9 285K", "cpu_count": 24, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31", "python": "3.10.13", "torch": "2.11.0+cu128", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 5090", "gpu_memory_gb": 31.84, "cuda": "12.8", "cudnn": 91900}
