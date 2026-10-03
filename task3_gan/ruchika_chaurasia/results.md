# Task 3 — CycleGAN Monet ↔ photo · ruchika_chaurasia (lineup A)

Course convention (from the ISA's evaluation script): **A = Monet, B = photo**. G_AB translates Monet → photo
(`pred_A2B/`) and G_BA translates photo → Monet (`pred_B2A/`, the painting direction).
This folder contains my **main run** (30 epochs). `runs/` contains the seven follow-up experiments. My **final model is
`runs/final200_noid`**, chosen by a rule fixed in advance (see "Choosing the final model").

## What I built
A CycleGAN implemented from scratch with plain PyTorch layers (`src/cyclegan_models.py`): two generators and two
discriminators trained together with adversarial, cycle-consistency and (optionally) identity losses.

| Setting | Main run (`config.yaml`) |
|---|---|
| Generators G_AB, G_BA | ResNet: c7s1-64, two stride-2 downsampling convs, 9 residual blocks (256 channels), two transposed-conv upsampling layers, c7s1-3 + tanh; instance norm; 11.4M parameters each |
| Discriminators D_A, D_B | 70×70 PatchGAN (3 stride-2 layers, ndf 64, instance norm), 2.8M parameters each · 28.3M parameters in total |
| Losses | LSGAN adversarial loss + cycle loss (λ_cyc = 10) + identity loss (λ_id = 0.5 × λ_cyc) |
| Optimiser | Adam, lr 2e-4, β1 = 0.5; constant for 15 epochs, then linear decay to 0 |
| Training | 30 epochs × 1,000 random unpaired steps, batch 1, image pool of 50 past fakes, fp16 mixed precision, seed 8283 |
| Data | 300 Monet paintings and 6,538 photos for training; **500 photos held out** (team-fixed list in `task3_gan/common/team_eval_split.json`) for evaluation and the human audit; resize to 286 → random 256 crop + horizontal flip |
| Hardware | NVIDIA GeForce RTX 5090 (32 GB) · Intel Core Ultra 9 285K |

## Why these choices
* **ResNet-9 generator + 70×70 PatchGAN + LSGAN, λ_cyc = 10, λ_id = 0.5** is the CycleGAN paper's own 256×256 setup, so my
  main run is a faithful baseline that every later change can be compared against.
* **Residual blocks** start close to the identity function, which suits style transfer: the layout of the image should stay
  and only colours and textures should change.
* **LSGAN** (least-squares loss) gives smoother gradients than the original binary cross-entropy GAN loss and was more stable in
  the paper.
* **The image pool** shows the discriminators a mix of current and older fakes, which reduces oscillation.
* **Holding out 500 photos** lets me measure the photo → Monet direction on images the model never trained on.

## The experiment series (all runs, same code and seed)

| Run | What changed | Official FID | MiFID | Kaggle score | Held-out photo→Monet FID | Monet→photo FID | Precision (B2A) | Density (B2A) | Content cosine (B2A) |
|---|---|---|---|---|---|---|---|---|---|
| main (30 ep) | paper setup, 30 epochs | 123.51 | 0.428 | **−61.97** (submitted) | 124.98 | 106.03 | 0.264 | 0.160 | 0.747 |
| long100 | 100 epochs | 104.70 | 0.417 | **−52.56** (submitted) | 103.13 | 94.41 | 0.426 | 0.296 | 0.738 |
| long100_diffaug | + DiffAugment (colour, translation, cutout) | 108.99 | 0.419 | — | 108.18 | 106.06 | 0.416 | 0.272 | 0.779 |
| final200 | 200 epochs, λ_id 0.1 | 103.63 | 0.415 | — | 93.76 | **88.19** | 0.502 | 0.452 | 0.774 |
| **final200_noid** | 200 epochs, **no identity loss** | **101.10** | 0.414 | **−50.76** (submitted, final) | **91.14** | 93.92 | **0.526** | 0.454 | 0.775 |
| lc5_100 | 100 ep, no identity, **λ_cyc 5** | 113.28 | 0.413 | — | 96.05 | 105.98 | 0.522 | **0.490** | 0.687 |
| lc5_d4_100 | + **4-layer discriminator** | 150.45 | 0.433 | — | 216.73 | 106.95 | 0.030 | 0.007 | 0.581 |
| lc5_ema_100 | lc5_100 + **generator EMA** | 111.07 | 0.416 | — | 99.53 | 102.65 | 0.434 | 0.343 | 0.719 |

"Official" FID and MiFID are computed by the ISA's `Part3_Evaluation_Script.ipynb`, run **unchanged** (it averages FID over
both directions using 300 images per side). The Kaggle score equals −(FID + MiFID) / 2. The other columns are my extra
metrics (`task3_gan/common/metrics_task3.py`) on the 500 held-out photos and all 300 paintings. Each run's full metrics are in
its `full_metrics_report.csv`, and its executed notebook is in the run folder.

### What the experiments show
1. **Training length mattered most.** 30 → 100 epochs lowered the official FID from 123.5 to 104.7, and photo → Monet precision
   rose from 0.26 to 0.43. Going from 100 to 200 epochs helped much less (about 1–3 FID points), so training length was mostly
   used up by 200 epochs.
2. **The identity loss is a trade-off.** Removing it (final200_noid vs final200) made photo → Monet more Monet-like (FID 93.8 →
   91.1, precision 0.50 → 0.53) but Monet → photo worse (88.2 → 93.9), because the generated photos drift in colour. This is the
   effect the CycleGAN paper describes.
3. **DiffAugment made things slightly worse** (108.99 vs 104.70 at 100 epochs). Colour jitter and cutout hide exactly the
   colour and brush-stroke statistics the discriminator needs to judge Monet style.
4. **A weaker cycle loss (λ_cyc 5) hurt.** The translations changed much more (LPIPS 0.38 → 0.57) and kept less content
   (cosine 0.78 → 0.69) without becoming better paintings, so the cycle loss is what keeps the content intact.
5. **A larger discriminator caused a collapse.** The 4-layer discriminator memorised the 300 paintings (D_A loss ≈ 0.001), the
   generator received no useful signal, and the outputs became the same striped noise pattern (FID 216.7, precision 0.03). See
   failure 4 in `failure_analysis.md`.
6. **Generator EMA helped slightly** (111.1 vs 113.3), but runs with identical settings already differ a little because GPU
   computations are not perfectly deterministic, so this gap is within single-run noise.

### Choosing the final model
Before comparing the runs I fixed the rule: choose the run with the best **held-out photo → Monet FID and precision**, not the
best Kaggle score. **final200_noid** wins on both (FID 91.14, precision 0.526), and it also has the best official FID (101.10),
so the held-out choice and the leaderboard agree.

## Training behaviour, convergence and stability
**Main run (30 epochs, `outputs/loss_curves.png`):** all curves are smooth with no spikes. The cycle losses fall steadily
(from about 4.0 to 1.00 / 1.22) and the identity losses to about 0.56–0.58, but both are **still falling at the end**, so the
model was under-trained, which the longer runs confirmed. The discriminator losses settle around 0.11–0.17 and the generator
adversarial losses around 0.45–0.60. The mean discriminator outputs separate slowly (D_A on real ≈ 0.76, on fake ≈ 0.24 at the
end): the discriminators are winning a little, but there is no collapse. 0 NaN losses; 17 of 30,000 steps were skipped by fp16
overflow protection.

**Final model (final200_noid, `runs/final200_noid/outputs/loss_curves.png`):**
* The **cycle losses fall smoothly for all 200 epochs** (about 4.0 → 0.70 / 0.85) and are still slowly improving at the end.
* The **Monet discriminator D_A becomes very confident** during the constant-learning-rate phase: by about step 75K its outputs
  are ≈ 0.93 on real paintings and ≈ 0.07 on fakes, its loss falls to about 0.05, and the photo → Monet generator's adversarial
  loss rises from 0.4 to 0.8. In other words, D_A is winning.
* There are **four short instability events** (around steps 49K, 55K, 79K and 99K): D_A's loss jumps to 0.4–0.8 for a moment,
  with matching spikes in the generator loss and gradient norms. Training recovers each time within a few hundred steps, and
  there were no NaN losses.
* After the learning rate starts to decay (step 100K), and especially from about step 125K, the game becomes more balanced: D_A's
  outputs move back to ≈ 0.80 / 0.20, its loss rises slightly to about 0.10, and the spikes stop. The photo discriminator D_B
  stays stable for the whole run (real ≈ 0.71, fake ≈ 0.29).
* So the main stability risk in this task is the **Monet discriminator overpowering the generator**, because it has only 300
  paintings to learn from. The 4-layer discriminator run shows what happens when it wins completely.

## Verifying the cycle-consistency constraint
If the cycle loss works, an image translated and then translated back must be much closer to the original than the translation
is. The notebook checks this for every run, and it holds in both directions, e.g. main run:
Monet → photo → Monet L1 **0.110** vs Monet-vs-photo-version L1 0.185; photo → Monet → photo L1 **0.134** vs 0.262 ("cycle
constraint VERIFIED" in the log). For final200_noid the photo cycle error is **0.103**. The figures
`outputs/figures/cycle_B2A_photo2monet.png` and `cycle_A2B_monet2photo.png` show input / translation / reconstruction side by
side: the reconstructions are close to the inputs. However, failure 4 shows that a small cycle error alone does **not** prove
good translations, because a collapsed model can hide the image in noise. That is why I also report FID, KID, precision/recall
and content similarity.

## Visual quality
On typical daytime landscapes the final model keeps the scene layout (boats, people, rivers, trees) and adds a softer pastel
palette and visible brush-stroke texture: misty scenes turn warm cream, autumn tones appear in trees, snowy branches get
speckled colourful strokes (`runs/final200_noid/outputs/figures/cycle_B2A_photo2monet.png`). Weak points are large flat areas
(artificial blocky texture), night scenes (brightened), very smooth scenes like waterfalls (only small texture changes), and, in
Monet → photo, invented lighting such as sunsets and dusk. Details are in `failure_analysis.md`.

## Cost
Main run, which had the GPU to itself: **36.9 min** for 30 epochs, **13.5 image pairs per second**, peak GPU memory **1.62 GB**,
inference **560 images per second**, **28,285,832** parameters (G_AB and G_BA 11.4M each, D_A and D_B 2.8M each). The longer runs
were trained two or three at a time on the same GPU, so their wall-clock times are longer than they would be alone and are not
directly comparable (final200_noid: about 6.3 hours for 200 epochs while sharing the GPU).

## Kaggle
| Submission | Score |
|---|---|
| main run (30 epochs), ckpt `aec6800f8b2e` | −61.9685 |
| long100, ckpt `cb67fbc511ff` | −52.5597 |
| **final200_noid, ckpt `bcbcfc109a7e`** | **−50.7560** |

Every `submission.csv` was produced by running the ISA's evaluation script unchanged on my own generators' outputs. Team 40's
final rank and the private scores will be recorded after the competition closes (Oct 6) and reported in the team report.

## Human audit
The blinded 30-sample audit (2 raters, style / content / artefacts, Cohen's κ) uses the team-fixed audit photos. Its results go
in `task3_gan/audit/audit_results.csv` and the team report.

## Shortcomings and limitations
* All conclusions come from **single runs** (one seed). GPU non-determinism already causes small differences between identical
  settings, so gaps of 1–3 FID points should not be over-interpreted.
* **FID on only 300 images** is noisy and biased upwards. The official script uses 300 images per side, so the absolute values
  are high for everyone and only comparisons are meaningful.
* The Monet → photo metrics use the **training paintings** as inputs, because only 300 paintings exist; only the photo → Monet
  direction is measured on truly held-out inputs.
* The Monet domain is **small and narrow** (300 daytime paintings), which explains most failures (night scenes, flat areas) and
  the discriminator overfitting.
* Removing the identity loss improved the Kaggle score but made Monet → photo colours less faithful. A small identity weight
  might be a better overall compromise.

## Evidence
* Main run: `src/task3_gan.ipynb`, `checkpoints/cyclegan_final.pt` (`aec6800f8b2e`), `full_metrics_report.csv`,
  `outputs/loss_curves.png`, `outputs/progress/` (one image grid per epoch), `outputs/figures/`, `outputs/pred_A2B/`,
  `outputs/pred_B2A/`, `submission.csv`, `outputs/official_eval_executed.ipynb`
* Every run: `runs/<run>/task3_gan_<run>.ipynb`, `full_metrics_report.csv`, `submission.csv`, `outputs/`; configs with the
  reason for each change: `config_<run>.yaml`
* Checkpoints in git: main run, `runs/long100`, `runs/final200_noid` (the three submitted models). The other runs' checkpoints
  are kept in the Drive archive.
* Raw logs and manifests: `reproducibility/raw_logs/task3_gan_ruchika_chaurasia_*`, `reproducibility/manifests/`
