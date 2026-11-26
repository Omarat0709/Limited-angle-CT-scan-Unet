"""
Figure for Section 4.9 (unseen shapes): the same five mixed-shape validation
phantoms reconstructed by the ellipse-trained baseline and by the network
retrained on mixed shapes, so the contribution of the shape prior is visible
on identical inputs.

Panels are cropped from the two existing comparison figures, which share the
same validation split and layout:
  training/runs/xeval/base_on_shapes/comparison.png  (ellipse-trained on shapes)
  training/runs/shapes800/comparison.png             (shapes-trained on shapes)
Per-image PSNR values are the ones printed on those figures.
Run from the repository root:  python analysis/make_fig_shapes_paired.py
Writes analysis/fig_shapes_paired.png
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "..", "training", "runs", "xeval", "base_on_shapes", "comparison.png")
B = os.path.join(HERE, "..", "training", "runs", "shapes800", "comparison.png")

ROWS = [(42, 385), (425, 768), (808, 1151)]                 # input, unet, gt
COLS = [(32, 375), (418, 761), (804, 1147), (1190, 1533), (1576, 1919)]

psnr_in   = [15.3, 16.0, 15.8, 16.6, 16.8]
psnr_ell  = [32.6, 31.4, 36.4, 37.8, 38.5]
psnr_shp  = [33.2, 33.9, 37.9, 37.9, 40.7]

a = np.array(Image.open(A).convert("L"))
b = np.array(Image.open(B).convert("L"))
def crop(img, r, c):
    (y0, y1), (x0, x1) = ROWS[r], COLS[c]
    return img[y0:y1, x0:x1]

rows = [("Limited-angle FBP input", [crop(a, 0, c) for c in range(5)], psnr_in),
        ("Ellipse-trained U-Net",   [crop(a, 1, c) for c in range(5)], psnr_ell),
        ("Shapes-trained U-Net",    [crop(b, 1, c) for c in range(5)], psnr_shp),
        ("Ground truth",            [crop(a, 2, c) for c in range(5)], None)]

fig, axes = plt.subplots(4, 5, figsize=(12.5, 10.6))
for r, (label, imgs, ps) in enumerate(rows):
    for c in range(5):
        ax = axes[r, c]
        ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if ps is not None:
            ax.set_title(f"{ps[c]:.1f} dB", fontsize=10, pad=3)
    axes[r, 0].set_ylabel(label, fontsize=11, labelpad=6)
fig.subplots_adjust(left=0.05, right=0.995, top=0.97, bottom=0.01, wspace=0.04, hspace=0.12)
out = os.path.join(HERE, "fig_shapes_paired.png")
fig.savefig(out, dpi=170)
print("wrote", out)
