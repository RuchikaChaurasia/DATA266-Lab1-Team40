# Task 3 — Failure analysis (ruchika_chaurasia)

Final model: **final200_noid** (ResNet-9 CycleGAN, 200 epochs, no identity loss, checkpoint `bcbcfc109a7e`).
Course convention: A = Monet, B = photo, so G_BA = photo → Monet and G_AB = Monet → photo.
All examples come from my own figures:
`runs/final200_noid/outputs/figures/cycle_B2A_photo2monet.png`, `worst_cycle_B2A.png`, `cycle_A2B_monet2photo.png`
and `runs/lc5_d4_100/outputs/figures/cycle_B2A_photo2monet.png`. The "worst" figure shows the six held-out photos with the
highest cycle-reconstruction error, so it is a good place to look for failures.

---

## Failure 1: Large flat, saturated areas → muddy colours and checkerboard texture

*Where:* `worst_cycle_B2A.png`, column 1 (a bright, cloudless blue sky with an aeroplane contrail) and column 6
(lighthouse in front of a pink sky).

**What I observed:** In the blue-sky photo, the Monet version turns the clean blue into a muddy purple-brown with a
visible **blocky, checkerboard-like pattern**, and the white contrail becomes an orange-yellow streak. The reconstruction
comes back **dark navy** instead of bright blue, so the colour information is lost, not just changed. The pink sky behind the
lighthouse turns beige with horizontal banding.

**Why:** Monet paintings never contain large perfectly flat areas; every region has brush strokes. The generator therefore
"invents" texture where there is none, and in a featureless region that texture becomes a repetitive pattern. The checkerboard
look is a known side effect of the transposed convolutions used for upsampling in the generator. Saturated pure blue is also
rare in the 300 Monet paintings, so the generator maps it to colours it has seen more often.

## Failure 2: Unusual lighting (night scene) → brightness is not preserved

*Where:* `worst_cycle_B2A.png`, column 2 (an almost black night sky with a tree silhouette).

**What I observed:** The "Monet" version of a dark night photo is a **light pinkish-grey** image. The darkness of the scene
is gone, and only a faint outline of the tree remains. The reconstruction comes back as a lighter, blue-grey sky instead of
near-black.

**Why:** Almost all of the 300 Monet paintings are daylight scenes with a light, pastel palette, so the photo → Monet generator
has learned "Monet = light and pastel" and applies it even when it destroys the scene. This is a training-data coverage problem:
there are practically no night scenes in domain A. My final model has **no identity loss** (λ_id = 0), and the identity loss
is exactly the term that would discourage such large colour and brightness changes.

## Failure 3: Monet → photo hallucinates lighting and loses content

*Where:* `cycle_A2B_monet2photo.png`, columns 3, 4 and 5.

**What I observed:** The pale, misty river painting (column 3) becomes a **dark blue dusk scene**. The Houses of Parliament
painting (column 5) gets a dramatic **red sunset sky** that is not in the painting. In column 4 the buildings of the Venice
painting almost **disappear into haze**; the output is mostly an orange-beige blur, and even the reconstruction cannot bring
the buildings back. In columns 3 and 6 the **painter's signature** in the corner is still visible in the generated "photo".

**Why:** Without the identity loss, nothing stops G_AB from shifting colours and brightness, which matches my metrics: the
Monet → photo FID got worse when I removed the identity loss (final200 with λ_id = 0.1: 88.2; final200_noid: 93.9). Very soft,
low-contrast paintings give the generator little structure to keep, so details are lost. The signature survives because the
cycle loss rewards keeping anything that the other generator needs to rebuild the painting.

## Failure 4 (training failure): the 4-layer-discriminator run collapsed and "hid" the image in noise

*Where:* `runs/lc5_d4_100/outputs/figures/cycle_B2A_photo2monet.png` (run with a larger 4-layer discriminator and cycle weight 5).

**What I observed:** Every "Monet" output of this run is a washed-out image covered by the **same vertical stripe and
coloured-noise pattern**. None of them look like paintings. The surprising part is the third row: the **reconstructions are
almost perfect**, and they even restore colours that are not visible in the translation, such as the orange sunset in
column 3. The metrics confirm the failure: held-out photo → Monet FID **216.7**, precision **0.03**, density **0.007** (my final
model: 91.1 / 0.526 / 0.454). During training, its Monet discriminator loss fell to about **0.001**.

**Why:** The larger discriminator (7M parameters) memorised the 300 Monet paintings and could reject every generated image,
so the generator stopped getting useful feedback (a discriminator overfitting on a tiny dataset). The generator still had to
satisfy the cycle loss, so it learned to **hide the photo's information in a nearly invisible high-frequency pattern** that
the other generator can decode. This is a known CycleGAN weakness ("CycleGAN, a master of steganography", Chu et al., 2017).
It shows that a low cycle-reconstruction error alone does **not** prove that the translation is good, which is why I also
report FID, KID, precision/recall and content similarity.

---

## Summary

My final model works well on typical daytime landscapes (rivers, trees, coasts): it keeps the scene layout and adds a pastel
palette and visible brush-stroke texture (`cycle_B2A_photo2monet.png`). It fails on inputs unlike Monet's paintings: large
flat saturated areas get artificial texture, and night scenes get brightened. Both are caused by the small and narrow Monet
training set (300 daytime paintings). The Monet → photo direction shows colour and lighting hallucinations, which become
worse without the identity loss. This is the trade-off I measured: removing it improved photo → Monet but hurt Monet → photo.
Finally, the collapsed 4-layer-discriminator run shows how fragile GAN training is with so little data, and why the cycle loss
alone cannot be trusted as a quality measure. Possible fixes include a small identity weight (around 0.1) as a compromise,
resize-convolution upsampling instead of transposed convolutions to remove checkerboard patterns, and a stronger or regularised
discriminator setup (for example spectral normalisation) to prevent discriminator overfitting.
