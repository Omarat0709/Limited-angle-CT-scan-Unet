r"""E10 — Hybrid reconstruction: U-Net prior + iterative data consistency.
=========================================================================

MOTIVATION (from the thesis's own findings)
-------------------------------------------
The characterisation showed two complementary failure modes. Classical
iterative methods (OS-SART) never contradict the measured rays but cannot
fill the missing wedge; the learned post-processor fills the wedge from its
prior but nothing ever checks its output against the measurements — which is
exactly why its out-of-distribution failures are silent (Sections 4.13, 4.16).

THE HYBRID
----------
A sinogram-completion scheme with the trained U-Net UNCHANGED:

    x0 = UNet(FBP(b))                        # the standard pipeline
    repeat `--cycles` times:
        x = DC(x)      # sinogram completion: project x over a dense 180-deg
                       #  grid, overwrite the measured angles' rows with the
                       #  actual measurements b, invert with FBP -- the data
                       #  is honoured exactly, the prior fills only the wedge
        x = UNet(x)    # (between cycles only) re-apply the prior
    output: x after the final DC step        # ends measurement-consistent

The measured sinogram b is re-simulated as Ax(ground truth) with the exact
geometry and angle set recorded in the dataset's meta.json — identical to how
the dataset generator produced the inputs, so it IS the measured data for the
noiseless datasets this script supports (pilot_v1, pilot_shapes,
e6_lodopab_test). Do not point it at the Poisson-noise datasets: their noise
realisations cannot be reproduced here, so Ax(GT) would not be the measured b.

USAGE (from the repository root, tigre env)
-----------------------------
  python reconstruct_hybrid.py --data data/pilot_v1        --split val
  python reconstruct_hybrid.py --data data/pilot_shapes    --split val
  python reconstruct_hybrid.py --data data/e6_lodopab_test --split all

Outputs (under training\runs\hybrid_e10\<dataset-name>\):
  hybrid_summary.json     mean PSNR/SSIM for FBP input, U-Net, and hybrid
  hybrid_comparison.png   FBP / U-Net / Hybrid / GT columns with PSNRs
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "training"))

import tigre  # noqa: E402
import tigre.algorithms as algs  # noqa: E402
from generate_dataset import make_geometry  # noqa: E402  (same geometry, guaranteed)
from metrics import psnr_ssim  # noqa: E402  (same conventions as all evals)

import torch  # noqa: E402
from unet_model import FBPConvNet  # noqa: E402


# ----------------------------- pieces ---------------------------------------

def load_net(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    net = FBPConvNet(base_channels=cfg.get("base_channels", 64),
                     residual=not cfg.get("no_residual", False)).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net


@torch.no_grad()
def apply_net(net, img, device):
    t = torch.from_numpy(img[None, None].astype(np.float32)).to(device)
    return net(t)[0, 0].cpu().numpy().astype(np.float32)


def angles_from_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    if "angles_deg" in meta:                       # newer datasets record them
        a = np.deg2rad(np.asarray(meta["angles_deg"], dtype=np.float32))
    else:                                          # original contiguous protocol
        limited = float(meta.get("limited_angle_deg", 120.0))
        nproj = int(meta.get("n_proj_limited", 60))
        a = np.deg2rad(np.linspace(0.0, limited, nproj,
                                   endpoint=False).astype(np.float32))
    return a.astype(np.float32), meta


def build_full_grid(angles_meas, full_deg=180.0, step_deg=1.0):
    """Dense 180-deg angle grid for the completion step; the measured angles
    must sit on it exactly (true for all 0.5-proj/deg datasets: they lie on a
    2-deg grid, a subset of the 1-deg grid)."""
    full = np.deg2rad(np.arange(0.0, full_deg, step_deg)).astype(np.float32)
    idx = []
    for a in angles_meas:
        j = int(np.argmin(np.abs(full - a)))
        if abs(float(full[j]) - float(a)) > np.deg2rad(0.05):
            raise SystemExit(f"measured angle {np.rad2deg(float(a)):.3f} deg "
                             f"is not on the {step_deg}-deg DC grid")
        idx.append(j)
    return full, np.asarray(idx, dtype=int)


def sino_dc(x, b, geo, angles_full, idx_meas):
    """Sinogram-completion data consistency.

    Why not warm-started OS-SART/SIRT? Because iterative solvers converge to
    THEIR solution (~the classical bar) regardless of initialisation -- the
    iterations progressively overwrite the prior's content in the unmeasured
    wedge, which is exactly the content worth keeping. (Measured empirically:
    warm-started DC dragged a 36 dB image to ~20 dB in-distribution while
    lifting a 15 dB image to ~19 dB on real data -- both marching toward the
    same fixed point.)

    Instead, keep each information source where it is authoritative: simulate
    the full 180-deg sinogram of the U-Net image (the prior fills the wedge),
    overwrite the measured angles' rows with the actual measurements (the data
    is honoured EXACTLY where it exists), and invert the now-complete sinogram
    with FBP, which is well-conditioned on complete data."""
    g = copy.deepcopy(geo)
    proj_full = tigre.Ax(np.ascontiguousarray(x[None].astype(np.float32)),
                         g, angles_full)
    proj_full[idx_meas] = b
    out = algs.fbp(proj_full, g, angles_full)
    return np.maximum(out[0], 0.0)


def sino_dc_ossart(x, b, geo, angles_full, idx_meas, n_iters):
    """OS-SART inversion of the composite sinogram, warm-started at x.

    Unlike FBP, a least-squares solver degrades gracefully when the prior rows
    and the measured rows disagree (the out-of-distribution case): conflicts
    are averaged rather than backprojected as streaks."""
    g = copy.deepcopy(geo)
    proj_full = tigre.Ax(np.ascontiguousarray(x[None].astype(np.float32)),
                         g, angles_full)
    proj_full[idx_meas] = b
    out = algs.ossart(proj_full, copy.deepcopy(geo), angles_full, n_iters,
                      init=np.ascontiguousarray(x[None].astype(np.float32)))
    return np.maximum(out[0], 0.0)


def tradeoff_sweep(net, device, pairs_fn, geo, angles, checkpoints, n_images):
    """The prior<->data trade-off curve: warm-started OS-SART on the MEASURED
    data only, scored at iteration checkpoints. Iteration 0 is the pure U-Net;
    infinity is the classical solution. Returns mean PSNR per checkpoint."""
    totals = {c: 0.0 for c in checkpoints}
    count = 0
    for x_in, y in pairs_fn():
        if count >= n_images:
            break
        g = copy.deepcopy(geo)
        b = tigre.Ax(y[None].astype(np.float32), g, angles)
        x = apply_net(net, x_in, device)
        done = 0
        for c in checkpoints:
            if c > done:
                x = np.maximum(
                    algs.ossart(b, copy.deepcopy(geo), angles, c - done,
                                init=np.ascontiguousarray(
                                    x[None].astype(np.float32)))[0], 0.0)
                done = c
            ps, _ = psnr_ssim(x, y)
            totals[c] += ps
        count += 1
    return {c: totals[c] / count for c in checkpoints}, count


def hybrid_reconstruct(net, device, fbp_in, target, geo, angles,
                       angles_full, idx_meas, cycles, solver="ossart",
                       dc_iters=15):
    """Returns (x_unet, x_hybrid, b). b is re-simulated from the target."""
    g = copy.deepcopy(geo)                        # TIGRE mutates geo in Ax
    b = tigre.Ax(target[None].astype(np.float32), g, angles)
    x = apply_net(net, fbp_in, device)
    x_unet = x
    for c in range(cycles):
        if solver == "fbp":
            x = sino_dc(x, b, g, angles_full, idx_meas)
        else:
            x = sino_dc_ossart(x, b, g, angles_full, idx_meas, dc_iters)
        if c < cycles - 1:
            x = apply_net(net, x, device)
    return x_unet, x, b


# ----------------------------- main -----------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default=os.path.join("training", "runs", "20260622_154318",
                                        "best_model.pt"))
    p.add_argument("--data", default=os.path.join("data", "pilot_v1"))
    p.add_argument("--split", choices=["val", "all"], default="val",
                   help="'val': the standard held-out split (seed 42, 10%%); "
                        "'all': every pair (use for the e6 test set)")
    p.add_argument("--cycles", type=int, default=1,
                   help="prior/data-consistency cycles (ends on DC, so the "
                        "output always honours the measurements exactly)")
    p.add_argument("--grid-step-deg", type=float, default=0.25,
                   help="angular spacing of the dense 180-deg completion grid "
                        "(0.25 => 720 views, above the ~400-view Nyquist "
                        "requirement at 256 px)")
    p.add_argument("--dc-solver", choices=["fbp", "ossart"], default="ossart",
                   help="inversion of the composite sinogram: 'fbp' is exact "
                        "on consistent data but streaks when prior and "
                        "measurements disagree; 'ossart' averages conflicts")
    p.add_argument("--dc-iters", type=int, default=15,
                   help="OS-SART iterations for --dc-solver ossart")
    p.add_argument("--sweep", action="store_true",
                   help="measure the prior<->data trade-off curve instead: "
                        "warm-started OS-SART on measured data only, scored "
                        "at iteration checkpoints, first --sweep-n images")
    p.add_argument("--sweep-n", type=int, default=20)
    p.add_argument("--sweep-deep", action="store_true",
                   help="extend the sweep to 50/75/100/150 iterations "
                        "to locate the semi-convergence peak")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug", action="store_true",
                   help="first image only: decompose where the DC step loses "
                        "accuracy (oracle full-sinogram FBP ceiling vs "
                        "prior-row and seam effects), then exit")
    args = p.parse_args()

    device = torch.device(args.device)
    net = load_net(args.ckpt, device)
    angles, meta = angles_from_meta(args.data)
    angles_full, idx_meas = build_full_grid(angles,
                                            step_deg=args.grid_step_deg)
    geo = make_geometry()

    if args.split == "val":
        # EXACTLY the held-out split every other evaluation uses -- do not
        # re-derive it here, import it, so train images can never leak in.
        from ct_dataset import build_datasets
        _, val_ds, _, _ = build_datasets(args.data, val_frac=args.val_frac,
                                         seed=args.seed)
        def pairs():
            for k in range(len(val_ds)):
                x_t, y_t = val_ds[k]
                yield (x_t.numpy()[0].astype(np.float32),
                       y_t.numpy()[0].astype(np.float32))
        n_total = len(val_ds)
    else:  # all pairs, in file order (the e6 test set is wholly held out)
        inp_dir = os.path.join(args.data, "inputs")
        tgt_dir = os.path.join(args.data, "targets")
        names = sorted(os.listdir(tgt_dir))
        def pairs():
            for name in names:
                yield (np.load(os.path.join(inp_dir, name)).astype(np.float32),
                       np.load(os.path.join(tgt_dir, name)).astype(np.float32))
        n_total = len(names)
    print(f"{args.data} | {n_total} images | {len(angles)} angles | "
          f"cycles={args.cycles} grid_step={args.grid_step_deg} deg")

    if args.sweep:
        checkpoints = [0, 1, 2, 3, 5, 8, 12, 20, 30]
        if args.sweep_deep:
            checkpoints += [50, 75, 100, 150]
        curve, n_used = tradeoff_sweep(net, device, pairs, geo, angles,
                                       checkpoints, args.sweep_n)
        out_dir = os.path.join("training", "runs", "hybrid_e10",
                               os.path.basename(os.path.normpath(args.data)))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tradeoff_sweep.json"), "w") as f:
            json.dump({"data": args.data, "n": n_used,
                       "checkpoints": checkpoints,
                       "psnr": {str(c): curve[c] for c in checkpoints}}, f,
                      indent=2)
        print(f"\n=== prior<->data trade-off ({args.data}, {n_used} images) ===")
        print("  OS-SART iters (0 = pure U-Net) -> mean PSNR:")
        for c in checkpoints:
            print(f"    {c:>3}: {curve[c]:6.2f} dB")
        print(f"Saved: {out_dir}\\tradeoff_sweep.json")
        return

    if args.debug:
        x_in, y = next(pairs())
        g = copy.deepcopy(geo)
        x_unet = apply_net(net, x_in, device)
        b = tigre.Ax(y[None].astype(np.float32), g, angles)
        # 1. oracle: FBP of the ground truth's own complete sinogram
        proj_true = tigre.Ax(y[None].astype(np.float32), copy.deepcopy(geo),
                             angles_full)
        x_oracle = np.maximum(algs.fbp(proj_true, copy.deepcopy(geo),
                                       angles_full)[0], 0.0)
        # 2. prior-only: FBP of the U-Net image's synthesised sinogram
        proj_prior = tigre.Ax(x_unet[None].astype(np.float32),
                              copy.deepcopy(geo), angles_full)
        x_prior = np.maximum(algs.fbp(proj_prior, copy.deepcopy(geo),
                                      angles_full)[0], 0.0)
        # 3. the hybrid (measured rows swapped in)
        x_hyb = (sino_dc(x_unet, b, geo, angles_full, idx_meas)
                 if args.dc_solver == "fbp" else
                 sino_dc_ossart(x_unet, b, geo, angles_full, idx_meas,
                                args.dc_iters))
        print("\n=== DEBUG: single-image decomposition of the DC step ===")
        for label, img in [("FBP input", x_in), ("U-Net", x_unet),
                           ("oracle  (FBP of TRUE full sinogram)", x_oracle),
                           ("prior   (FBP of U-Net's sinogram)", x_prior),
                           ("hybrid  (measured rows swapped in)", x_hyb)]:
            ps, ss = psnr_ssim(img, y)
            print(f"  {label:<38}: {ps:6.2f} dB / {ss:.3f}")
        print("Reading: if the oracle is low, the inversion grid is the "
              "bottleneck (raise density via --grid-step-deg 0.5 or 0.25). "
              "If the oracle is high but the hybrid is low, the seam between "
              "measured and synthesised rows is the bottleneck (needs "
              "blending).")
        return

    sums = {k: [0.0, 0.0] for k in ("fbp", "unet", "hyb")}
    examples = []
    for i, (x_in, y) in enumerate(pairs()):
        x_unet, x_hyb, _ = hybrid_reconstruct(
            net, device, x_in, y, geo, angles, angles_full, idx_meas,
            args.cycles, solver=args.dc_solver, dc_iters=args.dc_iters)
        for key, img in (("fbp", x_in), ("unet", x_unet), ("hyb", x_hyb)):
            ps, ss = psnr_ssim(img, y)
            sums[key][0] += ps
            sums[key][1] += ss
        if len(examples) < args.examples:
            examples.append((x_in, x_unet, x_hyb, y))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_total}")

    n = float(n_total)
    summary = {
        "data": args.data, "ckpt": args.ckpt, "n": int(n),
        "cycles": args.cycles, "grid_step_deg": args.grid_step_deg,
        "dc_solver": args.dc_solver, "dc_iters": args.dc_iters,
        "fbp_psnr": sums["fbp"][0] / n, "fbp_ssim": sums["fbp"][1] / n,
        "unet_psnr": sums["unet"][0] / n, "unet_ssim": sums["unet"][1] / n,
        "hybrid_psnr": sums["hyb"][0] / n, "hybrid_ssim": sums["hyb"][1] / n,
    }
    summary["hybrid_minus_unet_psnr"] = (summary["hybrid_psnr"]
                                         - summary["unet_psnr"])

    out_dir = os.path.join("training", "runs", "hybrid_e10",
                           os.path.basename(os.path.normpath(args.data)))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "hybrid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.data} ({int(n)} images) ===")
    print(f"  FBP    : {summary['fbp_psnr']:.2f} dB / {summary['fbp_ssim']:.3f}")
    print(f"  U-Net  : {summary['unet_psnr']:.2f} dB / {summary['unet_ssim']:.3f}")
    print(f"  Hybrid : {summary['hybrid_psnr']:.2f} dB / {summary['hybrid_ssim']:.3f}"
          f"   ({summary['hybrid_minus_unet_psnr']:+.2f} dB vs U-Net)")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = len(examples)
        fig, axes = plt.subplots(4, cols, figsize=(3 * cols, 12))
        if cols == 1:
            axes = axes[:, None]
        rows = ["FBP-LA (input)", "U-Net", "U-Net + DC (hybrid)", "Ground truth"]
        for c, (xi, xu, xh, y) in enumerate(examples):
            imgs = [xi, xu, xh, y]
            for r, img in enumerate(imgs):
                ax = axes[r, c]
                ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                title = rows[r] if c == 0 else ""
                if r < 3:
                    ps, _ = psnr_ssim(img, y)
                    title = (title + "  " if title else "") + f"{ps:.1f} dB"
                if title:
                    ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "hybrid_comparison.png"), dpi=130)
        print(f"Saved: {out_dir}\\hybrid_summary.json and hybrid_comparison.png")
    except Exception as e:
        print(f"Figure skipped: {e}")


if __name__ == "__main__":
    main()
