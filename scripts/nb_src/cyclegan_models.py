"""CycleGAN building blocks (Zhu et al., 2017), written with plain torch.nn layers.

* ResnetGenerator  — c7s1-64, d128, d256, R256 x n_blocks, u128, u64, c7s1-3 (paper's 256x256 generator)
* UnetGenerator    — 8-level encoder/decoder with skip connections (pix2pix-style), 256x256 input
* PatchDiscriminator — 70x70 PatchGAN (n_layers=3) or larger receptive field (n_layers=4)
* ImagePool        — history buffer of 50 generated images used to update the discriminators
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m, std=0.02):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _norm(c):
    return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)


class ResnetBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), _norm(c), nn.ReLU(True),
                                  nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), _norm(c))

    def forward(self, x):
        return x + self.body(x)          # residual connection keeps the input content


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9, **_):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), _norm(ngf), nn.ReLU(True)]
        c = ngf
        for _ in range(2):                                          # downsample x2
            layers += [nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), _norm(2 * c), nn.ReLU(True)]
            c *= 2
        layers += [ResnetBlock(c) for _ in range(n_blocks)]
        for _ in range(2):                                          # upsample x2
            layers += [nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1, output_padding=1), _norm(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UnetGenerator(nn.Module):
    """Encoder: 256 -> 1 in 8 stride-2 convs. Decoder mirrors it; skip connections concatenate encoder features."""

    def __init__(self, ngf=64, num_downs=8, dropout=False, **_):
        super().__init__()
        chs = [ngf * m for m in (1, 2, 4, 8)] + [ngf * 8] * (num_downs - 4)
        self.down = nn.ModuleList()
        in_c = 3
        for i, c in enumerate(chs):
            layers = [] if i == 0 else [nn.LeakyReLU(0.2, True)]
            layers.append(nn.Conv2d(in_c, c, 4, 2, 1))
            if 0 < i < num_downs - 1:            # no norm on the first layer or the 1x1 bottleneck
                layers.append(_norm(c))
            self.down.append(nn.Sequential(*layers))
            in_c = c
        self.up = nn.ModuleList()
        for i in reversed(range(num_downs)):
            out_c = 3 if i == 0 else chs[i - 1]
            in_up = chs[i] if i == num_downs - 1 else chs[i] * 2   # concat with skip except innermost
            layers = [nn.ReLU(True), nn.ConvTranspose2d(in_up, out_c, 4, 2, 1)]
            if i == 0:
                layers.append(nn.Tanh())
            else:
                layers.append(_norm(out_c))
                if dropout and i >= num_downs - 3:
                    layers.append(nn.Dropout(0.5))
            self.up.append(nn.Sequential(*layers))

    def forward(self, x):
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
        x = self.up[0](skips[-1])
        for j, u in enumerate(self.up[1:], start=2):
            x = u(torch.cat([x, skips[-j]], dim=1))
        return x


class PatchDiscriminator(nn.Module):
    """Outputs a grid of real/fake logits, each judging one overlapping image patch."""

    def __init__(self, ndf=64, n_layers=3, **_):
        super().__init__()
        layers = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1), _norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1), _norm(ndf * mult), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(ndf * mult, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ImagePool:
    """Shrivastava et al. (2017): update D with a mix of current and past fakes to reduce oscillation."""

    def __init__(self, size=50):
        self.size, self.images = size, []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img); out.append(img)
            elif random.random() > 0.5:
                i = random.randrange(self.size)
                old = self.images[i].clone(); self.images[i] = img; out.append(old)
            else:
                out.append(img)
        return torch.cat(out, 0)


def build_models(mc: dict):
    if mc["generator"] == "resnet":
        G = lambda: ResnetGenerator(ngf=mc["ngf"], n_blocks=mc.get("n_res_blocks", 9))
    elif mc["generator"] == "unet":
        G = lambda: UnetGenerator(ngf=mc["ngf"], dropout=mc.get("gen_dropout", False))
    else:
        raise ValueError(mc["generator"])
    D = lambda: PatchDiscriminator(ndf=mc["ndf"], n_layers=mc.get("d_layers", 3))
    G_AB, G_BA, D_A, D_B = G(), G(), D(), D()
    for m in (G_AB, G_BA, D_A, D_B):
        m.apply(init_weights)
    return G_AB, G_BA, D_A, D_B


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- DiffAugment
# Zhao, S., Liu, Z., Lin, J., Zhu, J.-Y., & Han, S. (2020). Differentiable Augmentation for Data-Efficient
# GAN Training. NeurIPS. Re-implemented from the paper's reference code. Images are in [-1, 1].
def _brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _saturation(x):
    m = x.mean(dim=1, keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + m


def _contrast(x):
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + m


def _translation(x, ratio=0.125):
    sx, sy = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-sy, sy + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(x.size(0), device=x.device), torch.arange(x.size(2), device=x.device),
                                torch.arange(x.size(3), device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    xp = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x, ratio=0.5):
    cs = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - cs[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - cs[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(x.size(0), device=x.device), torch.arange(cs[0], device=x.device),
                                torch.arange(cs[1], device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - cs[0] // 2, min=0, max=x.size(2) - 1)
    gy = torch.clamp(gy + oy - cs[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


_AUGS = {"color": [_brightness, _saturation, _contrast], "translation": [_translation], "cutout": [_cutout]}


def DiffAugment(x, policy="color,translation,cutout"):
    """Apply the same KIND of random augmentation to real and fake images before the discriminator."""
    for p in [p.strip() for p in policy.split(",") if p.strip()]:
        for f in _AUGS[p]:
            x = f(x)
    return x.contiguous()
