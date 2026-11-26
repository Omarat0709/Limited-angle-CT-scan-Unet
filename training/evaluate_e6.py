"""
E6: Evaluate the trained network on real LoDoPaB-CT anatomy.
============================================================

WHAT THIS FILE DOES
-------------------
Scores the three-way comparison (limited-angle FBP / OS-SART / trained U-Net)
on the real-anatomy dataset produced by build_e6_dataset.py. This is the
direct measurement of the synthetic-to-real domain gap that Section 4.7 of
the thesis predicts: the network was trained exclusively on random-ellipse
phantoms, and these are clinical thorax CT slices it has never seen.

Unlike the synthetic evaluations, per-image statistics matter here (real
anatomy varies far more between slices than random ellipses do), so this
script reports mean +/- standard deviation, writes a per-image CSV for later
distribution plots, and saves a qualitative figure.

HOW TO RUN (from the repository root, inside the `tigre` env; TIGRE itself not needed)
------------------------------------------------------------------------
    python training/evaluate_e6.py --run training/runs/<run-name>

Outputs (into the run folder):
    e6_summary.json      mean/std PSNR & SSIM for FBP, OS-SART, U-Net
    e6_per_image.csv     one row per slice (for box plots / histograms)
    e6_comparison.png    FBP / OS-SART / U-Net / GT grid on real slices
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

from unet_model import FBPConvNet
from metrics import psnr_ssim

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "data", "e6_lodopab_test")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = FBPConvNet(base_channels=cfg.get("base_channels", 64),
                       residual=not cfg.get("no_residual", False)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def unet_infer(model, img2d, device):
    x = torch.from_numpy(img2d.astype(np.float32))[None, None].to(device)
    return model(x)[0, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None,
                   help="run dir (uses best_model.pt inside it)")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path")
    p.add_argument("--data", default=DEFAULT_DATA,
                   help="E6 dataset dir from build_e6_dataset.py")
    p.add_argument("--examples", type=int, default=4,
                   help="sample columns in the comparison figure")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    if args.ckpt:
        ckpt_path = args.ckpt
    elif args.run:
        ckpt_path = os.path.join(args.run, "best_model.pt")
    else:
        raise SystemExit("Provide --run <dir> or --ckpt <file>")
    out_dir = os.path.dirname(os.path.abspath(ckpt_path))

    device = torch.device(args.device)
    model, ckpt = load_model(ckpt_path, device)
    print(f"Loaded {ckpt_path} | trained epoch {ckpt.get('epoch', '?')}")

    inputs = sorted(glob.glob(os.path.join(args.data, "inputs", "*.npy")))
    if not inputs:
        raise SystemExit(f"No inputs in {args.data} - run build_e6_dataset.py "
                         "first (or run_e6.bat).")
    have_sart = os.path.isdir(os.path.join(args.data, "ossart"))
    print(f"Evaluating on {len(inputs)} real LoDoPaB-CT slices "
          f"({'with' if have_sart else 'WITHOUT'} OS-SART column)")

    methods = ["fbp", "ossart", "unet"] if have_sart else ["fbp", "unet"]
    per_image = []
    examples = []
    for i, in_path in enumerate(inputs):
        name = os.path.basename(in_path)
        fbp = np.load(in_path).astype(np.float32)
        gt = np.load(os.path.join(args.data, "targets", name)).astype(np.float32)
        recons = {"fbp": fbp, "unet": unet_infer(model, fbp, device)}
        if have_sart:
            recons["ossart"] = np.load(
                os.path.join(args.data, "ossart", name)).astype(np.float32)
        row = {"image": name}
        for m in methods:
            ps, ss = psnr_ssim(recons[m], gt)
            row[f"{m}_psnr"] = ps
            row[f"{m}_ssim"] = ss
        per_image.append(row)
        if len(examples) < args.examples:
            examples.append((recons, gt, row))
        if (i + 1) % 25 == 0 or (i + 1) == len(inputs):
            print(f"  {i + 1}/{len(inputs)}")

    # ---- Aggregate: mean and std per method ----
    summary = {"n": len(per_image), "data": os.path.abspath(args.data),
               "checkpoint": os.path.abspath(ckpt_path)}
    for m in methods:
        for k in ("psnr", "ssim"):
            vals = np.array([r[f"{m}_{k}"] for r in per_image])
            summary[f"{m}_{k}"] = float(vals.mean())
            summary[f"{m}_{k}_std"] = float(vals.std())
    if "ossart_psnr" in summary:
        summary["unet_minus_ossart_psnr"] = (summary["unet_psnr"]
                                             - summary["ossart_psnr"])
    summary["unet_minus_fbp_psnr"] = summary["unet_psnr"] - summary["fbp_psnr"]

    print("\n=== E6: real-data evaluation (mean +/- std) ===")
    for m in methods:
        print(f"  {m.upper():7s} {summary[f'{m}_psnr']:6.2f} "
              f"+/- {summary[f'{m}_psnr_std']:.2f} dB | "
              f"SSIM {summary[f'{m}_ssim']:.3f} "
              f"+/- {summary[f'{m}_ssim_std']:.3f}")
    print(f"  U-Net vs its FBP input: "
          f"{summary['unet_minus_fbp_psnr']:+.2f} dB")

    with open(os.path.join(out_dir, "e6_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    csv_path = os.path.join(out_dir, "e6_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    # ---- Qualitative figure: rows = methods + GT, columns = examples ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row_defs = [("fbp", "FBP-LA (input)"), ("ossart", "OS-SART"),
                ("unet", "U-Net")] if have_sart else \
               [("fbp", "FBP-LA (input)"), ("unet", "U-Net")]
    nrows = len(row_defs) + 1
    cols = len(examples)
    fig, axes = plt.subplots(nrows, cols, figsize=(3 * cols, 3 * nrows))
    if cols == 1:
        axes = axes[:, None]
    for c, (recons, gt, row) in enumerate(examples):
        for r, (m, label) in enumerate(row_defs):
            ax = axes[r, c]
            ax.imshow(np.clip(recons[m], 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            title = (label + "  " if c == 0 else "") + f"{row[f'{m}_psnr']:.1f} dB"
            ax.set_title(title, fontsize=9)
        ax = axes[nrows - 1, c]
        ax.imshow(np.clip(gt, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if c == 0:
            ax.set_title("Ground truth (real)", fontsize=9)
    fig.suptitle("E6 - Trained on synthetic ellipses, tested on real "
                 "LoDoPaB-CT anatomy", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "e6_comparison.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    print(f"\nSaved: {os.path.join(out_dir, 'e6_summary.json')}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
