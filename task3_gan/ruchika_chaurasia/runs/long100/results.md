# Task 3 — CycleGAN photo↔Monet · ruchika_chaurasia (lineup A, run: long100)

## What I built
Generators: resnet (ngf 64, 9 residual blocks); discriminators: PatchGAN ndf 64, 3 stride-2 layers; instance norm.

**Hyper-parameters:** GAN loss lsgan, λ_cycle 10.0, λ_identity 0.5 (×λ_cycle), Adam lr 0.0002 β1 0.5, batch 1, 100 epochs × 1000 steps (linear decay from epoch 50), image pool 50, AMP True, DiffAugment none, image 256 (load 286, random crop + flip), seed 8283.

## Why (design justification)
* Same architecture and losses as the main lineup-A run (ResNet-9, 70x70 PatchGAN, LSGAN, lambda_cyc 10,
  lambda_id 0.5). Only the training budget changes: 100 x 1000 = 100K iterations instead of 30K.
* Hypothesis: the 30-epoch run was under-trained. Its cycle loss was still falling at epoch 30, and the CycleGAN
  paper trains for ~200 epochs. A longer constant-LR phase followed by linear decay should lower FID in both directions.


## Metrics — both directions (checkpoint `cb67fbc511ff`)
| metric | A2B | B2A |
|---|---|---|
| content_cosine | 0.7928 | 0.738 |
| cycle_l1 | 0.0791 | 0.1062 |
| fid | 94.41 | 103.1 |
| gen_coverage | 0.64 | 0.69 |
| gen_density | 0.5607 | 0.296 |
| gen_precision | 0.6333 | 0.426 |
| gen_recall | 0.534 | 0.6567 |
| kid_mean | 0.02505 | 0.02312 |
| kid_std | 0.002747 | 0.00245 |
| l1_input_vs_translation | 0.2307 | 0.2946 |
| lpips_input_vs_translation | 0.3082 | 0.376 |
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
| inference_images_per_sec | 441 |
| final_epoch_G_adv_AB | 0.4603 |
| final_epoch_G_adv_BA | 0.7218 |
| final_epoch_D_A | 0.07783 |
| final_epoch_D_B | 0.1643 |
| final_epoch_cycle_loss_A | 0.7007 |
| final_epoch_cycle_loss_B | 0.867 |
| final_epoch_identity_loss_A | 0.3094 |
| final_epoch_identity_loss_B | 0.3744 |
| gradnorm_G_mean | 25.51 |
| gradnorm_G_max | 86.29 |
| gradnorm_D_mean | 11.35 |
| gradnorm_D_max | 62.94 |
| gradnorm_G_nonfinite_steps | 42 |
| gradnorm_D_nonfinite_steps | 3 |
| nan_count | 0 |
| amp_overflow_skips | 43 |
| training_time_min | 232 |
| train_steps | 100000 |
| epochs | 100 |
| train_image_pairs_per_sec | 7.184 |
| peak_memory_mb | 1619 |
| hardware | NVIDIA GeForce RTX 5090 / Intel(R) Core(TM) Ultra 9 285K |

## Kaggle
Public score: None · private score: None · team rank: None
(`submission.csv` from the official script on this member's `pred_A2B/` + `pred_B2A/`, checkpoint `cb67fbc511ff`).

## Human audit (30 fixed photos, 2 blinded raters)
TODO — run the blinded audit (task3_gan/audit/README.md)

## Training behaviour (write in your own words)
TODO — describe the loss curves (`outputs/loss_curves.png`) and progress grids (`outputs/progress/`):
did cycle loss plateau, did D overpower G, any spikes/NaNs, when did Monet-like texture appear?

## Evidence
Loss log `outputs/loss_log.csv` · curves `outputs/loss_curves.png` · progress grids `outputs/progress/` ·
cycle figures `outputs/figures/` · predictions `outputs/pred_A2B/`, `outputs/pred_B2A/` · raw log `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_long100_20261002_011017.log`.

## Hardware
{"cpu": "Intel(R) Core(TM) Ultra 9 285K", "cpu_count": 24, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.31", "python": "3.10.13", "torch": "2.11.0+cu128", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 5090", "gpu_memory_gb": 31.84, "cuda": "12.8", "cudnn": 91900}
