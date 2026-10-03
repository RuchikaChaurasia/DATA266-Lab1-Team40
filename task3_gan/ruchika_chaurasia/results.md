# Task 3 — CycleGAN photo↔Monet · ruchika_chaurasia (lineup A)

## What I built
Generators: resnet (ngf 64, 9 residual blocks); discriminators: PatchGAN ndf 64, 3 stride-2 layers; instance norm.

**Hyper-parameters:** GAN loss lsgan, λ_cycle 10.0, λ_identity 0.5 (×λ_cycle), Adam lr 0.0002 β1 0.5, batch 1, 30 epochs × 1000 steps (linear decay from epoch 15), image pool 50, AMP True, image 256 (load 286, random crop + flip), seed 8283.

## Why (design justification)
* Faithful reproduction of the CycleGAN paper's 256x256 setup (ResNet-9 generator, 70x70 PatchGAN,
  LSGAN loss, lambda_cyc = 10, identity 0.5, Adam 2e-4 / beta1 0.5, image pool 50). This is the reference
  configuration, so any teammate's deviation can be judged against it.
* ResNet generator: the residual blocks start close to the identity function, which suits style
  transfer. The photo's layout should survive; only texture and colour change.
* LSGAN (least squares) gives non-saturating gradients and was more stable than BCE in the paper.
* Identity loss 0.5: Monet<->photo translation otherwise tends to shift overall colour (e.g. turning
  daytime photos into sunsets), which hurts content preservation.
* 30 x 1000 unpaired steps fits in a few T4 hours with AMP. Half constant then linear decay follows the
  paper's 100 + 100 schedule, compressed.


## Metrics — both directions (checkpoint `aec6800f8b2e`)
| metric | A2B | B2A |
|---|---|---|
| content_cosine | 0.8014 | 0.7469 |
| cycle_l1 | 0.1095 | 0.1343 |
| fid | 106 | 125 |
| gen_coverage | 0.48 | 0.4667 |
| gen_density | 0.5467 | 0.16 |
| gen_precision | 0.5767 | 0.264 |
| gen_recall | 0.482 | 0.6633 |
| kid_mean | 0.0368 | 0.03864 |
| kid_std | 0.003874 | 0.003959 |
| l1_input_vs_translation | 0.1852 | 0.2619 |
| lpips_input_vs_translation | 0.3151 | 0.4228 |
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
| inference_images_per_sec | 559.8 |
| final_epoch_G_adv_AB | 0.4504 |
| final_epoch_G_adv_BA | 0.5966 |
| final_epoch_D_A | 0.1127 |
| final_epoch_D_B | 0.1663 |
| final_epoch_cycle_loss_A | 1 |
| final_epoch_cycle_loss_B | 1.224 |
| final_epoch_identity_loss_A | 0.5602 |
| final_epoch_identity_loss_B | 0.5812 |
| gradnorm_G_mean | 31.6 |
| gradnorm_G_max | 56.44 |
| gradnorm_D_mean | 16.15 |
| gradnorm_D_max | 28.96 |
| nan_count | 0 |
| amp_overflow_skips | 17 |
| training_time_min | 36.9 |
| train_steps | 30000 |
| epochs | 30 |
| train_image_pairs_per_sec | 13.55 |
| peak_memory_mb | 1619 |
| hardware | NVIDIA GeForce RTX 5090 / Intel(R) Core(TM) Ultra 9 285K |

Gradient norms recomputed without fp16-overflow windows: G non-finite windows 14, D 1 (of 600).

## Kaggle
Public score: None · private score: None · team rank: None
(`submission.csv` from the official script on this member's `pred_A2B/` + `pred_B2A/`, checkpoint `aec6800f8b2e`).

## Human audit (30 fixed photos, 2 blinded raters)
TODO — run the blinded audit (task3_gan/audit/README.md)

## Training behaviour (write in your own words)
TODO — describe the loss curves (`outputs/loss_curves.png`) and progress grids (`outputs/progress/`):
did cycle loss plateau, did D overpower G, any spikes/NaNs, when did Monet-like texture appear?

## Evidence
Loss log `outputs/loss_log.csv` · curves `outputs/loss_curves.png` · progress grids `outputs/progress/` ·
cycle figures `outputs/figures/` · predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` · raw log `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_20261001_231058.log`.

## Hardware
{"cpu": "Intel(R) Core(TM) Ultra 9 285K", "cpu_count": 24, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31", "python": "3.10.13", "torch": "2.11.0+cu128", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 5090", "gpu_memory_gb": 31.84, "cuda": "12.8", "cudnn": 91900}
