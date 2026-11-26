"""
E1: Held-out Shepp-Logan test + OS-SART comparison on the validation set.
=========================================================================

WHAT THIS FILE DOES
-------------------
The pilot dataset is random-ellipse phantoms; Shepp-Logan was deliberately held
out so it can serve as a standard, truly unseen test image. This script closes
the two gaps left by evaluate_unet.py:

  1. SHEPP-LOGAN TEST: simulates the same limited-angle scan of the standard
     Shepp-Logan phantom, reconstructs with FBP and OS-SART, runs the trained
     U-Net on the FBP input, and scores all three against the ground truth.
     -> answers "does the network generalise beyond its training distribution?"

  2. OS-SART ON THE VAL SET: the dataset stores only (FBP input, ground truth),
     so the classical iterative bar was never measured on the val images. This
     script re-projects each val ground truth with TIGRE (same geometry/angles
     as generation, read from the dataset's meta.json) and reconstructs with
     OS-SART, giving a fair three-way FBP / OS-SART / U-Net table on the
     identical held-out images.
     -> turns "25.1 dB on Shepp-Logan" into a proper like-for-like column.

REQUIREMENTS
------------
Needs TIGRE (unlike train/evaluate), so run inside the `tigre` conda env,
from the repository root:

    python training/evaluate_e1.py --run training/runs/<run-name>

Outputs (written into the run folder):
    e1_summary.json              all the numbers below, machine-readable
    e1_shepp_logan.png           GT / FBP / OS-SART / U-Net figure (thesis-ready)

NOTES ON FAIRNESS
-----------------
* Metrics use the SAME convention as metrics.py (clamp to [0,1], data_range=1),
  so every number is directly comparable to Table 4.2 and eval_summary.json.
* For noisy datasets (meta.json noise_i0 > 0) the re-projection adds Poisson
  noise with the same I0. The noise REALISATION differs from the one baked into
  the stored FBP inputs (different draws), which is the standard, fair setup:
  each method sees an equally noisy sinogram, not literally the same one.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

from unet_model import FBPConvNet
from ct_dataset import build_datasets
from metrics import psnr_ssim

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "data", "pilot_v1")
IMG_SIZE = 256


# ----------------------------------------------------------------------------
# Model loading (same pattern as evaluate_unet.py)
# ----------------------------------------------------------------------------

def load_model(ckpt_path, device):
    """Rebuild the network from its checkpoint config and load the weights."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = FBPConvNet(base_channels=cfg.get("base_channels", 64),
                       residual=not cfg.get("no_residual", False)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def unet_infer(model, img2d, device):
    """Run one (H, W) numpy image through the network, return (H, W) numpy."""
    x = torch.from_numpy(img2d.astype(np.float32))[None, None].to(device)
    return model(x)[0, 0].cpu().numpy()


# ----------------------------------------------------------------------------
# TIGRE simulation (geometry matched to generate_dataset.py / the demo)
# ----------------------------------------------------------------------------

def load_sim_config(data_dir):
    """Read the scan parameters this dataset was generated with, so the
    OS-SART comparison uses the IDENTICAL geometry. Falls back to the
    baseline 120-deg / 60-projection setup if meta.json is absent."""
    cfg = {"limited_angle_deg": 120.0, "n_proj_limited": 60, "noise_i0": 0}
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for k in cfg:
            if k in meta:
                cfg[k] = meta[k]
    return cfg


def make_geometry(size=IMG_SIZE):
    """Same parallel-beam geometry as limited_angle_demo.py."""
    import tigre
    geo = tigre.geometry(mode="parallel",
                         nVoxel=np.array([1, size, size]),
                         default=True)
    geo.nDetector = np.array([1, size])
    geo.sDetector = geo.dDetector * geo.nDetector
    return geo


def simulate_recons(gt2d, geo_clean, angles, noise_i0=0, rng=None,
                    sart_iters=20):
    """Project a ground-truth slice over the limited angles and reconstruct
    with both FBP and OS-SART. Returns (fbp, ossart) as (H, W) float32.

    TIGRE mutates the geometry inside Ax, so we work on a fresh deep copy
    (same defensive pattern as generate_dataset.simulate_pair)."""
    import tigre
    import tigre.algorithms as algs
    geo = copy.deepcopy(geo_clean)
    vol = gt2d[np.newaxis, :, :].astype(np.float32)
    proj = tigre.Ax(vol, geo, angles)

    if noise_i0 and noise_i0 > 0 and rng is not None:
        # Poisson noise in the transmission domain -- identical model to the
        # dataset generator, so noisy datasets get a fair noisy comparison.
        counts = noise_i0 * np.exp(-proj)
        counts = rng.poisson(np.maximum(counts, 1)).astype(np.float32)
        proj = -np.log(np.maximum(counts, 1) / noise_i0).astype(np.float32)

    fbp = algs.fbp(proj, geo, angles)[0].astype(np.float32)
    sart = algs.ossart(proj, geo, angles, niter=sart_iters)[0].astype(np.float32)
    return fbp, sart


# ----------------------------------------------------------------------------
# Part 1 -- Shepp-Logan held-out test
# ----------------------------------------------------------------------------

def shepp_logan_test(model, device, geo_clean, angles, sim_cfg, rng,
                     sart_iters, out_dir):
    """Scan/reconstruct the standard Shepp-Logan phantom, run the U-Net on the
    FBP input, score everything, and save the 4-panel thesis figure."""
    from tigre.utilities import sample_loader
    gt = sample_loader.load_head_phantom(np.array([1, IMG_SIZE, IMG_SIZE]))[0]
    gt = np.clip(gt, 0.0, 1.0).astype(np.float32)   # metric convention: [0,1]

    fbp, sart = simulate_recons(gt, geo_clean, angles,
                                noise_i0=sim_cfg["noise_i0"], rng=rng,
                                sart_iters=sart_iters)
    unet = unet_infer(model, fbp, device)

    results = {}
    for name, img in [("fbp", fbp), ("ossart", sart), ("unet", unet)]:
        p, s = psnr_ssim(img, gt)
        results[f"{name}_psnr"] = p
        results[f"{name}_ssim"] = s

    # --- thesis figure: GT | FBP | OS-SART | U-Net, PSNR/SSIM in titles ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Ground truth", gt, None),
              ("FBP-LA", fbp, ("fbp_psnr", "fbp_ssim")),
              (f"OS-SART ({sart_iters} it.)", sart, ("ossart_psnr", "ossart_ssim")),
              ("U-Net", unet, ("unet_psnr", "unet_ssim"))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, img, keys) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if keys:
            title += (f"\n{results[keys[0]]:.1f} dB / "
                      f"{results[keys[1]]:.3f} SSIM")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Held-out Shepp-Logan test "
                 f"({sim_cfg['limited_angle_deg']:.0f}$^\\circ$ wedge, "
                 f"{sim_cfg['n_proj_limited']} projections)", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "e1_shepp_logan.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {fig_path}")
    return results


# ----------------------------------------------------------------------------
# Part 2 -- OS-SART on the validation set
# ----------------------------------------------------------------------------

def val_set_comparison(model, device, val_pairs, geo_clean, angles, sim_cfg,
                       rng, sart_iters, max_val=0):
    """Three-way comparison on the held-out val images.

    FBP and U-Net are scored from the STORED inputs (exactly what Table 4.2
    used); OS-SART is computed fresh by re-projecting each ground truth."""
    if max_val and max_val > 0:
        val_pairs = val_pairs[:max_val]
    n = len(val_pairs)
    acc = {k: 0.0 for k in ("fbp_psnr", "fbp_ssim", "ossart_psnr",
                            "ossart_ssim", "unet_psnr", "unet_ssim")}
    for i, (in_path, tg_path) in enumerate(val_pairs):
        fbp = np.load(in_path).astype(np.float32)     # stored limited-angle FBP
        gt = np.load(tg_path).astype(np.float32)
        _, sart = simulate_recons(gt, geo_clean, angles,
                                  noise_i0=sim_cfg["noise_i0"], rng=rng,
                                  sart_iters=sart_iters)
        unet = unet_infer(model, fbp, device)
        for name, img in [("fbp", fbp), ("ossart", sart), ("unet", unet)]:
            p, s = psnr_ssim(img, gt)
            acc[f"{name}_psnr"] += p
            acc[f"{name}_ssim"] += s
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  val {i + 1}/{n}")
    return {k: v / n for k, v in acc.items()} | {"n": n}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None,
                   help="run dir (uses best_model.pt inside it)")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path")
    p.add_argument("--data", default=DEFAULT_DATA,
                   help="dataset dir (MUST be the one the model trained on)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42,
                   help="MUST match the training seed to reproduce the split")
    p.add_argument("--sart-iters", type=int, default=20,
                   help="OS-SART iterations (20 = same as the baseline demo)")
    p.add_argument("--max-val", type=int, default=0,
                   help="limit val images for a quick check (0 = all)")
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
    print(f"Loaded {ckpt_path} | trained epoch {ckpt.get('epoch', '?')} | "
          f"val PSNR at save {ckpt.get('val_psnr', float('nan')):.2f} dB")

    sim_cfg = load_sim_config(args.data)
    print(f"Scan config from dataset: {sim_cfg['limited_angle_deg']:.0f} deg, "
          f"{sim_cfg['n_proj_limited']} projections, "
          f"noise I0={sim_cfg['noise_i0']}")
    geo_clean = make_geometry()
    angles = np.linspace(0, np.deg2rad(sim_cfg["limited_angle_deg"]),
                         sim_cfg["n_proj_limited"],
                         endpoint=False).astype(np.float32)
    rng = np.random.default_rng(args.seed)

    # --- Part 1: Shepp-Logan ---
    print("\n[1/2] Held-out Shepp-Logan test...")
    sl = shepp_logan_test(model, device, geo_clean, angles, sim_cfg, rng,
                          args.sart_iters, out_dir)

    # --- Part 2: val set three-way ---
    print("\n[2/2] OS-SART vs FBP vs U-Net on the validation set...")
    _, _, _, val_pairs = build_datasets(args.data, val_frac=args.val_frac,
                                        seed=args.seed)
    vs = val_set_comparison(model, device, val_pairs, geo_clean, angles,
                            sim_cfg, rng, args.sart_iters,
                            max_val=args.max_val)

    # --- Report ---
    def row(tag, d):
        return (f"  {tag:22s} FBP {d['fbp_psnr']:6.2f} dB / {d['fbp_ssim']:.3f}"
                f"   OS-SART {d['ossart_psnr']:6.2f} dB / {d['ossart_ssim']:.3f}"
                f"   U-Net {d['unet_psnr']:6.2f} dB / {d['unet_ssim']:.3f}")

    print("\n=== E1 summary (PSNR / SSIM) ===")
    print(row("Shepp-Logan (held-out)", sl))
    print(row(f"Val set (n={vs['n']})", vs))

    summary = {"checkpoint": os.path.abspath(ckpt_path),
               "data": os.path.abspath(args.data),
               "sim_config": sim_cfg,
               "sart_iters": args.sart_iters,
               "shepp_logan": sl,
               "val_set": vs}
    out_json = os.path.join(out_dir, "e1_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
