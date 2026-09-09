#!/usr/bin/env python3
"""Regenerate the tutorial's schematic figures (docs/figures/)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'figures')


def encoder_decoder():
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=200); ax.set_xlim(0, 11); ax.set_ylim(0, 5.0); ax.axis('off')
    green, gold, grey, ink = '#2f5d3a', '#d9a53a', '#8a8f8a', '#1a1a1a'

    def box(x, y, w, h, color, label, sub=None, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=color, ec='none'))
        ax.text(x + w / 2, y + h / 2 + (0.11 if sub else 0), label, ha='center', va='center', color='white',
                fontsize=fs, fontweight='bold')
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.15, sub, ha='center', va='center', color='white', fontsize=7.5)

    def arrow(x0, y0, x1, y1, color=ink, ls='-'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, color=color, lw=1.2,
                                     linestyle=ls))

    W = 0.85
    ys = [3.55, 2.75, 1.95, 1.15]; hs = [0.8, 0.68, 0.56, 0.44]
    labels = ['stride 4', 'stride 8', 'stride 16', 'stride 32']
    ex = [1.55 + i * 0.95 for i in range(4)]
    dx = [8.60 - i * 0.95 for i in range(4)]
    box(0.2, 2.35, 1.0, 0.8, grey, 'Input tile', 'H × W × C')
    box(9.8, 2.35, 1.0, 0.8, grey, 'Class map', 'H × W')
    for i in range(4):
        box(ex[i], ys[i] - hs[i] / 2, W, hs[i], green, f'E{i + 1}', labels[i], fs=8.5)
        box(dx[i], ys[i] - hs[i] / 2, W, hs[i], gold, f'D{i + 1}', labels[i], fs=8.5)
        if i > 0:
            arrow(ex[i - 1] + W / 2, ys[i - 1] - hs[i - 1] / 2, ex[i] + W / 2, ys[i] + hs[i] / 2 + 0.02)
            arrow(dx[i] + W / 2, ys[i] + hs[i] / 2, dx[i - 1] + W / 2, ys[i - 1] - hs[i - 1] / 2 - 0.02)
        if i < 3:
            arrow(ex[i] + W, ys[i], dx[i], ys[i], color=green, ls=(0, (3, 2)))
    arrow(1.2, 2.75, ex[0], ys[0] - 0.1)
    arrow(dx[0] + W, ys[0] - 0.1, 9.8, 2.75)
    arrow(ex[3] + W, ys[3], dx[3], ys[3])
    ax.text(ex[0], 4.55, 'Encoder (backbone)', ha='left', fontsize=9.5, color=green, fontweight='bold')
    ax.text(ex[0], 4.30, 'pretrained; per stage: resolution ÷ 2, channels × 2', ha='left', fontsize=8, color=green)
    ax.text(dx[0] + W, 4.55, 'Decoder (decode head)', ha='right', fontsize=9.5, color=gold, fontweight='bold')
    ax.text(dx[0] + W, 4.30, 'fuses scales, upsamples to input size', ha='right', fontsize=8, color=gold)
    ax.text(5.5, 0.35, 'dashed: skip connections (U-Net, UPerNet, low-level branch of DeepLabV3+); '
            'E4 → D4 is the bottleneck with the most abstract features', ha='center', fontsize=7.5, color=grey)
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, 'encoder_decoder.png'), bbox_inches='tight', facecolor='white')
    print('written', os.path.join(OUT, 'encoder_decoder.png'))


if __name__ == '__main__':
    encoder_decoder()
