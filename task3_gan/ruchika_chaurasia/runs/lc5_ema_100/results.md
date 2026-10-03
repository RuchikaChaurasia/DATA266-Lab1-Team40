# Task 3 — CycleGAN photo↔Monet · ruchika_chaurasia (lineup A, run: lc5_ema_100)

## What I built
Generators: resnet (ngf 64, 9 residual blocks); discriminators: PatchGAN ndf 64, 3 stride-2 layers; instance norm.

**Hyper-parameters:** GAN loss lsgan, λ_cycle 5.0, λ_identity 0.0 (×λ_cycle), Adam lr 0.0002 β1 0.5, batch 1, 100 epochs × 1000 steps (linear decay from epoch 50), image pool 50, AMP True, DiffAugment none, generator EMA 0.999, image 256 (load 286, random crop + flip), seed 8283.

## Why (design justification)
* Identical to lc5_100 except that the generators used for evaluation are an exponential moving average
  (decay 0.999, about the last 1,000 steps) of the training weights (Yazici et al., 2019).
* Late in training GAN generators keep oscillating around a good solution. Averaging cancels much of that
  oscillation and usually reduces artefacts and FID. The images are still produced only by this CycleGAN.
* Comparing with lc5_100 isolates the effect of EMA. The raw (non-averaged) weights are kept in the checkpoint.
* Selection rule unchanged: held-out B2A FID / precision on the 500 held-out photos, not the Kaggle score.


## Metrics — both directions (checkpoint `7f657a49587c`)
| metric | A2B | B2A |
|---|---|---|
| content_cosine | 0.7352 | 0.7188 |
| cycle_l1 | 0.1089 | 0.1243 |
| fid | 102.7 | 99.53 |
| gen_coverage | 0.554 | 0.7367 |
| gen_density | 0.618 | 0.3428 |
| gen_precision | 0.6367 | 0.434 |
| gen_recall | 0.404 | 0.6733 |
| kid_mean | 0.03466 | 0.01899 |
| kid_std | 0.003266 | 0.002449 |
| l1_input_vs_translation | 0.2845 | 0.3001 |
| lpips_input_vs_translation | 0.4497 | 0.4624 |
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
| inference_images_per_sec | 419.8 |
| final_epoch_G_adv_AB | 0.5179 |
| final_epoch_G_adv_BA | 0.8919 |
| final_epoch_D_A | 0.02806 |
| final_epoch_D_B | 0.1372 |
| final_epoch_cycle_loss_A | 0.4966 |
| final_epoch_cycle_loss_B | 0.5555 |
| final_epoch_identity_loss_A | 0 |
| final_epoch_identity_loss_B | 0 |
| gradnorm_G_mean | 16.28 |
| gradnorm_G_max | 52.21 |
| gradnorm_D_mean | 12.26 |
| gradnorm_D_max | 56.91 |
| gradnorm_G_nonfinite_steps | 37 |
| gradnorm_D_nonfinite_steps | 11 |
| nan_count | 0 |
| amp_overflow_skips | 43 |
| training_time_min | 291 |
| train_steps | 100000 |
| epochs | 100 |
| train_image_pairs_per_sec | 5.727 |
| peak_memory_mb | 1367 |
| hardware | NVIDIA GeForce RTX 5090 / Intel(R) Core(TM) Ultra 9 285K |

## Kaggle
Public score: None · private score: None · team rank: None
(`submission.csv` from the official script on this member's `pred_A2B/` + `pred_B2A/`, checkpoint `7f657a49587c`).

## Human audit (30 fixed photos, 2 blinded raters)
TODO — run the blinded audit (task3_gan/audit/README.md)

## Training behaviour (write in your own words)
TODO — describe the loss curves (`outputs/loss_curves.png`) and progress grids (`outputs/progress/`):
did cycle loss plateau, did D overpower G, any spikes/NaNs, when did Monet-like texture appear?

## Evidence
Loss log `outputs/loss_log.csv` · curves `outputs/loss_curves.png` · progress grids `outputs/progress/` ·
cycle figures `outputs/figures/` · predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` · raw log `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_lc5_ema_100_20261002_213711.log`.

## Hardware
{"cpu": "Intel(R) Core(TM) Ultra 9 285K", "cpu_count": 24, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31", "python": "3.10.13", "torch": "2.11.0+cu128", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 5090", "gpu_memory_gb": 31.84, "cuda": "12.8", "cudnn": 91900}
