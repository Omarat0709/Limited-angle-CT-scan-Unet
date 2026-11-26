"""
Evaluate a trained FBPConvNet checkpoint and make the thesis result figure.
===========================================================================

WHAT THIS FILE DOES
-------------------
Training already reports PSNR/SSIM, but this script is the dedicated, after-the-
fact evaluator you point at a finished run. It:
  1. Reloads the best checkpoint and rebuilds the exact model it was saved from.
  2. Re-creates the SAME validation split (same seed) so it scores on the held-
     out images the model never trained on.
  3. Reports mean PSNR/SSIM for BOTH the raw FBP inputs (the floor) and the
     U-Net outputs (the improvement), plus the gain between them.
  4. Saves a qualitative comparison figure: columns are sample images, rows are
     [FBP-LA input | U-Net output | ground truth], with per-image PSNR labelled.
     This figure is the one you put in the results chapter.

HOW TO RUN (from the repository root)
-------------------------------------
    python training/evaluate_unet.py --run training/runs/<run-name>
    # or point straight at a specific checkpoint:
    python training/evaluate_unet.py --ckpt training/runs/<run-name>/best_model.pt

Outputs (written next to the checkpoint):
    eval_summary.json   mean PSNR/SSIM (FBP vs U-Net) and the gains
    comparison.png      the side-by-side qualitative figure
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from unet_model import FBPConvNet
from ct_dataset import build_datasets
from metrics import psnr_ssim, batch_psnr_ssim

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "data", "pilot_v1")


def load_model(ckpt_path, device):
    """Rebuild the model from a checkpoint and load its trained weights.

    The checkpoint stored the config, so we reconstruct the network with the
    SAME width and residual setting it was trained with -- otherwise the saved
    weights would not fit. Then we switch to eval() mode for inference."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = FBPConvNet(base_channels=cfg.get("base_channels", 64),
                       residual=not cfg.get("no_residual", False)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()    # pure inference, no gradients needed
def collect(model, loader, device, max_examples=5):
    """Run the model over the loader; return mean metrics + a few example
    triples (input, prediction, ground-truth) to draw later.

    We accumulate FBP-vs-GT and U-Net-vs-GT metrics in parallel so the summary
    shows the before/after on the identical set of images."""
    fbp_p, fbp_s = 0.0, 0.0      # FBP-input vs GT
    net_p, net_s = 0.0, 0.0      # U-Net-output vs GT
    n = 0
    examples = []
    for x, y in loader:
        x_d = x.to(device)
        pred = model(x_d).cpu()
        bs = x.size(0)
        fp, fs = batch_psnr_ssim(x, y);  fbp_p += fp * bs; fbp_s += fs * bs
        np_, ns = batch_psnr_ssim(pred, y); net_p += np_ * bs; net_s += ns * bs
        n += bs
        # Stash a handful of individual images for the comparison figure.
        if len(examples) < max_examples:
            for i in range(min(bs, max_examples - len(examples))):
                examples.append((x[i, 0].numpy(), pred[i, 0].numpy(),
                                 y[i, 0].numpy()))
    return {
        "n": n,
        "fbp_psnr": fbp_p / n, "fbp_ssim": fbp_s / n,
        "unet_psnr": net_p / n, "unet_ssim": net_s / n,
    }, examples


def save_comparison(examples, out_path):
    """Draw the 3-row figure (input / output / GT) for a handful of samples,
    annotating each reconstruction with its PSNR. All images are clamped to
    [0,1] and shown on a fixed gray scale so columns are visually comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = len(examples)
    fig, axes = plt.subplots(3, cols, figsize=(3 * cols, 9))
    if cols == 1:                       # keep indexing uniform for a single col
        axes = axes[:, None]
    row_titles = ["FBP-LA (input)", "U-Net (output)", "Ground truth"]
    for c, (inp, pred, gt) in enumerate(examples):
        ip, _ = psnr_ssim(inp, gt)      # input PSNR (the floor for this image)
        pp, _ = psnr_ssim(pred, gt)     # output PSNR (after the network)
        imgs = [(inp, f"{ip:.1f} dB"), (pred, f"{pp:.1f} dB"), (gt, "")]
        for r, (img, tag) in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            # Label the row only on the leftmost column; append PSNR per image.
            title = row_titles[r] if c == 0 else ""
            if tag:
                title = (title + "  " if title else "") + tag
            if title:
                ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None,
                   help="run dir (uses best_model.pt inside it)")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42,
                   help="MUST match the training seed to reproduce the split")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--examples", type=int, default=5,
                   help="how many sample columns in comparison.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    # Resolve which checkpoint to load.
    if args.ckpt:
        ckpt_path = args.ckpt
    elif args.run:
        ckpt_path = os.path.join(args.run, "best_model.pt")
    else:
        raise SystemExit("Provide --run <dir> or --ckpt <file>")
    out_dir = os.path.dirname(os.path.abspath(ckpt_path))

    device = torch.device(args.device)
    model, ckpt = load_model(ckpt_path, device)
    print(f"Loaded {ckpt_path} | trained epoch {ckpt.get('epoch','?')} | "
          f"val PSNR at save {ckpt.get('val_psnr', float('nan')):.2f} dB")

    # Same seed + same val_frac => the identical held-out validation set.
    _, val_ds, _, _ = build_datasets(args.data, val_frac=args.val_frac,
                                     seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    summary, examples = collect(model, val_loader, device,
                                max_examples=args.examples)
    summary["delta_psnr"] = summary["unet_psnr"] - summary["fbp_psnr"]
    summary["delta_ssim"] = summary["unet_ssim"] - summary["fbp_ssim"]

    # Print the headline before/after table.
    print("\n=== Validation set ({} images) ===".format(summary["n"]))
    print(f"  FBP-LA  : {summary['fbp_psnr']:.2f} dB / {summary['fbp_ssim']:.3f}")
    print(f"  U-Net   : {summary['unet_psnr']:.2f} dB / {summary['unet_ssim']:.3f}")
    print(f"  Gain    : +{summary['delta_psnr']:.2f} dB / "
          f"+{summary['delta_ssim']:.3f} SSIM")
    print(f"  (OS-SART classical bar to beat: 25.1 dB / 0.819)")

    # Save the machine-readable summary and the qualitative figure.
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    cmp_path = os.path.join(out_dir, "comparison.png")
    save_comparison(examples, cmp_path)
    print(f"\nSaved: {os.path.join(out_dir, 'eval_summary.json')}")
    print(f"Saved: {cmp_path}")


if __name__ == "__main__":
    main()
