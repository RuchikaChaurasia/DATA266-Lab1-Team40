# Task 3 — CycleGAN photo↔Monet · ruchika_chaurasia (lineup A, run: long100_diffaug)

## What I built
Generators: resnet (ngf 64, 9 residual blocks); discriminators: PatchGAN ndf 64, 3 stride-2 layers; instance norm.

**Hyper-parameters:** GAN loss lsgan, λ_cycle 10.0, λ_identity 0.5 (×λ_cycle), Adam lr 0.0002 β1 0.5, batch 1, 100 epochs × 1000 steps (linear decay from epoch 50), image pool 50, AMP True, DiffAugment color,translation,cutout, image 256 (load 286, random crop + flip), seed 8283.

## Why (design justification)
* Same as 'long100' plus DiffAugment (Zhao et al., 2020) on every discriminator input, real and fake.
* Hypothesis: with only 300 Monet paintings, D_A can memorise the training set late in a long run, which
  gives G poor gradients and causes artefacts. Differentiable augmentation applied identically to real and fake
  images prevents that without changing what the generator is asked to produce.
* Tests the effect of data-efficient GAN training on the painting domain (compare with 'long100').


## Metrics — both directions (checkpoint `9fd12c2b6237`)
| metric | A2B | B2A |
|---|---|---|
| content_cosine | 0.8087 | 0.7793 |
| cycle_l1 | 0.07612 | 0.09851 |
| fid | 106.1 | 108.2 |
| gen_coverage | 0.474 | 0.63 |
| gen_density | 0.4153 | 0.2724 |
| gen_precision | 0.5167 | 0.416 |
| gen_recall | 0.516 | 0.6067 |
| kid_mean | 0.03562 | 0.0284 |
| kid_std | 0.0034 | 0.003124 |
| l1_input_vs_translation | 0.1894 | 0.3024 |
| lpips_input_vs_translation | 0.2719 | 0.3474 |
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
| inference_images_per_sec | 559.2 |
| final_epoch_G_adv_AB | 0.4866 |
| final_epoch_G_adv_BA | 0.4662 |
| final_epoch_D_A | 0.1627 |
| final_epoch_D_B | 0.1658 |
| final_epoch_cycle_loss_A | 0.6448 |
| final_epoch_cycle_loss_B | 0.8123 |
| final_epoch_identity_loss_A | 0.3205 |
| final_epoch_identity_loss_B | 0.4125 |
| gradnorm_G_mean | 25.93 |
| gradnorm_G_max | 79.22 |
| gradnorm_D_mean | 13.23 |
| gradnorm_D_max | 74.7 |
| gradnorm_G_nonfinite_steps | 41 |
| gradnorm_D_nonfinite_steps | 3 |
| nan_count | 0 |
| amp_overflow_skips | 42 |
| training_time_min | 249.9 |
| train_steps | 100000 |
| epochs | 100 |
| train_image_pairs_per_sec | 6.668 |
| peak_memory_mb | 1622 |
| hardware | NVIDIA GeForce RTX 5090 / Intel(R) Core(TM) Ultra 9 285K |

## Kaggle
Public score: None · private score: None · team rank: None
(`submission.csv` from the official script on this member's `pred_A2B/` + `pred_B2A/`, checkpoint `9fd12c2b6237`).

## Human audit (30 fixed photos, 2 blinded raters)
See the team report (blinded audit on the team-fixed 30 photos).

## Training behaviour (write in your own words)
See the main Task 3 write-up for the analysis of all runs: `../../results.md` and `../../failure_analysis.md`.

## Evidence
Loss log `outputs/loss_log.csv` · curves `outputs/loss_curves.png` · progress grids `outputs/progress/` ·
cycle figures `outputs/figures/` · predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` · raw log `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_long100_diffaug_20261002_011057.log`.

## Hardware
{"cpu": "Intel(R) Core(TM) Ultra 9 285K", "cpu_count": 24, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31", "python": "3.10.13", "torch": "2.11.0+cu128", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 5090", "gpu_memory_gb": 31.84, "cuda": "12.8", "cudnn": 91900}
