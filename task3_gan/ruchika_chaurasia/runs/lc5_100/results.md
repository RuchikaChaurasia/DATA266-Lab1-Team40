# Task 3 — CycleGAN photo↔Monet · ruchika_chaurasia (lineup A, run: lc5_100)

## What I built
Generators: resnet (ngf 64, 9 residual blocks); discriminators: PatchGAN ndf 64, 3 stride-2 layers; instance norm.

**Hyper-parameters:** GAN loss lsgan, λ_cycle 5.0, λ_identity 0.0 (×λ_cycle), Adam lr 0.0002 β1 0.5, batch 1, 100 epochs × 1000 steps (linear decay from epoch 50), image pool 50, AMP True, DiffAugment none, generator EMA off, image 256 (load 286, random crop + flip), seed 8283.

## Why (design justification)
* Diagnosis from all runs so far: photo->Monet outputs are too conservative. B2A precision/density rose
  0.26/0.16 (30 ep) -> 0.53/0.45 (final200_noid) as constraints were relaxed (longer training, identity 0.5 -> 0).
* The remaining strong constraint is the cycle loss (lambda 10). Halving it lets G_BA move further from the photo
  while still being reversible. Risk: weaker content preservation (watch content_cosine and cycle_l1).
* 100 epochs because final200 vs long100 showed little gain beyond ~100 epochs.
* Selection rule unchanged: held-out B2A FID / precision on the 500 held-out photos, not the Kaggle score.


## Metrics — both directions (checkpoint `b9eed262df20`)
| metric | A2B | B2A |
|---|---|---|
| content_cosine | 0.7055 | 0.6873 |
| cycle_l1 | 0.1102 | 0.1284 |
| fid | 106 | 96.05 |
| gen_coverage | 0.47 | 0.8233 |
| gen_density | 0.5653 | 0.49 |
| gen_precision | 0.6533 | 0.522 |
| gen_recall | 0.354 | 0.59 |
| kid_mean | 0.0379 | 0.0202 |
| kid_std | 0.003215 | 0.002949 |
| l1_input_vs_translation | 0.3285 | 0.3553 |
| lpips_input_vs_translation | 0.5561 | 0.5719 |
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
| inference_images_per_sec | 446.9 |
| final_epoch_G_adv_AB | 0.5893 |
| final_epoch_G_adv_BA | 0.6274 |
| final_epoch_D_A | 0.09691 |
| final_epoch_D_B | 0.1132 |
| final_epoch_cycle_loss_A | 0.4957 |
| final_epoch_cycle_loss_B | 0.5668 |
| final_epoch_identity_loss_A | 0 |
| final_epoch_identity_loss_B | 0 |
| gradnorm_G_mean | 15.49 |
| gradnorm_G_max | 52.63 |
| gradnorm_D_mean | 12.1 |
| gradnorm_D_max | 55.26 |
| gradnorm_G_nonfinite_steps | 41 |
| gradnorm_D_nonfinite_steps | 5 |
| nan_count | 0 |
| amp_overflow_skips | 44 |
| training_time_min | 281.5 |
| train_steps | 100000 |
| epochs | 100 |
| train_image_pairs_per_sec | 5.921 |
| peak_memory_mb | 1283 |
| hardware | NVIDIA GeForce RTX 5090 / Intel(R) Core(TM) Ultra 9 285K |

## Kaggle
Public score: None · private score: None · team rank: None
(`submission.csv` from the official script on this member's `pred_A2B/` + `pred_B2A/`, checkpoint `b9eed262df20`).

## Human audit (30 fixed photos, 2 blinded raters)
See the team report (blinded audit on the team-fixed 30 photos).

## Training behaviour (write in your own words)
See the main Task 3 write-up for the analysis of all runs: `../../results.md` and `../../failure_analysis.md`.

## Evidence
Loss log `outputs/loss_log.csv` · curves `outputs/loss_curves.png` · progress grids `outputs/progress/` ·
cycle figures `outputs/figures/` · predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` · raw log `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_lc5_100_20261002_213704.log`.

## Hardware
{"cpu": "Intel(R) Core(TM) Ultra 9 285K", "cpu_count": 24, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31", "python": "3.10.13", "torch": "2.11.0+cu128", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 5090", "gpu_memory_gb": 31.84, "cuda": "12.8", "cudnn": 91900}
