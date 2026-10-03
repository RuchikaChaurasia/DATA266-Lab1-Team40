# Task 3 — CycleGAN photo↔Monet · ruchika_chaurasia (lineup A, run: final200)

## What I built
Generators: resnet (ngf 64, 9 residual blocks); discriminators: PatchGAN ndf 64, 3 stride-2 layers; instance norm.

**Hyper-parameters:** GAN loss lsgan, λ_cycle 10.0, λ_identity 0.1 (×λ_cycle), Adam lr 0.0002 β1 0.5, batch 1, 200 epochs × 1000 steps (linear decay from epoch 100), image pool 50, AMP True, DiffAugment none, image 256 (load 286, random crop + flip), seed 8283.

## Why (design justification)
* long100 outputs are too conservative: photo->Monet precision 0.426 / density 0.296, low input-vs-translation
  LPIPS, and DiffAugment (which made outputs even more conservative) scored worse.
* The identity loss explicitly rewards leaving images unchanged, so its weight is lowered from 0.5 to 0.1.
* Full CycleGAN paper schedule (100 + 100 epochs) instead of 50 + 50.
* Risk: a lower identity weight can introduce colour shifts in Monet->photo.
* Selection between final200 / final200_noid / long100 is made on the held-out metrics (500 held-out photos),
  not on Kaggle scores.


## Metrics — both directions (checkpoint `beb6bd86fbf1`)
| metric | A2B | B2A |
|---|---|---|
| content_cosine | 0.8002 | 0.7742 |
| cycle_l1 | 0.07813 | 0.1021 |
| fid | 88.19 | 93.76 |
| gen_coverage | 0.682 | 0.82 |
| gen_density | 0.7907 | 0.4524 |
| gen_precision | 0.68 | 0.502 |
| gen_recall | 0.52 | 0.68 |
| kid_mean | 0.01904 | 0.01394 |
| kid_std | 0.002371 | 0.002376 |
| l1_input_vs_translation | 0.2305 | 0.2692 |
| lpips_input_vs_translation | 0.3116 | 0.3594 |
| n_real_reference | 500 | 300 |
| n_translated | 300 | 500 |

## Training, stability and cost
| metric | value |
|---|---|
| params_G_AB | 11378179 |
| params_G_BA | 11378179 |
| params_D_A | 2764737 |
| params_D_B | 2764737 |
| params_total | 28285832 |
| inference_images_per_sec | 559.6 |
| final_epoch_G_adv_AB | 0.5881 |
| final_epoch_G_adv_BA | 0.7261 |
| final_epoch_D_A | 0.07458 |
| final_epoch_D_B | 0.1178 |
| final_epoch_cycle_loss_A | 0.6573 |
| final_epoch_cycle_loss_B | 0.8138 |
| final_epoch_identity_loss_A | 0.1101 |
| final_epoch_identity_loss_B | 0.1041 |
| gradnorm_G_mean | 24.06 |
| gradnorm_G_max | 73.56 |
| gradnorm_D_mean | 10.85 |
| gradnorm_D_max | 60.15 |
| gradnorm_G_nonfinite_steps | 79 |
| gradnorm_D_nonfinite_steps | 3 |
| nan_count | 0 |
| amp_overflow_skips | 80 |
| training_time_min | 422.5 |
| train_steps | 200000 |
| epochs | 200 |
| train_image_pairs_per_sec | 7.89 |
| peak_memory_mb | 1619 |
| hardware | NVIDIA GeForce RTX 5090 / Intel(R) Core(TM) Ultra 9 285K |

## Kaggle
Public score: None · private score: None · team rank: None
(`submission.csv` from the official script on this member's `pred_A2B/` + `pred_B2A/`, checkpoint `beb6bd86fbf1`).

## Human audit (30 fixed photos, 2 blinded raters)
See the team report (blinded audit on the team-fixed 30 photos).

## Training behaviour (write in your own words)
See the main Task 3 write-up for the analysis of all runs: `../../results.md` and `../../failure_analysis.md`.

## Evidence
Loss log `outputs/loss_log.csv` · curves `outputs/loss_curves.png` · progress grids `outputs/progress/` ·
cycle figures `outputs/figures/` · predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` · raw log `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_final200_20261002_062852.log`.

## Hardware
{"cpu": "Intel(R) Core(TM) Ultra 9 285K", "cpu_count": 24, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31", "python": "3.10.13", "torch": "2.11.0+cu128", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 5090", "gpu_memory_gb": 31.84, "cuda": "12.8", "cudnn": 91900}
