"""
Figure 4.7 (v2): dose response of the learned reconstruction, with the
noiseless-trained network evaluated WITHOUT retraining added as a second line.

Reads the numbers straight from the run folders:
  retrained at each dose : training/runs/noise_i0_1e4, noise_i0_1e5, noise_i0_1e6, noise_noiseless
  no retraining          : training/runs/xeval/clean_on_i0_1e4, _1e5, _1e6
                           (noiseless point = noise_noiseless evaluated on its own val set)
Run from the repository root:  python analysis/make_fig_noise_response_v2.py
Writes analysis/fig_noise_response_v2.png (the original file is untouched).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "training", "runs")

def load(rel):
    with open(os.path.join(RUNS, rel, "eval_summary.json")) as f:
        return json.load(f)

doses   = ["1e4", "1e5", "1e6"]
labels  = ["$10^4$", "$10^5$", "$10^6$", "noiseless"]
retrain = [load(f"noise_i0_{d}") for d in doses] + [load("noise_noiseless")]
xeval   = [load(f"xeval/clean_on_i0_{d}") for d in doses] + [load("noise_noiseless")]

x = list(range(4))
fbp_psnr = [r["fbp_psnr"] for r in retrain]
re_psnr  = [r["unet_psnr"] for r in retrain]
xe_psnr  = [r["unet_psnr"] for r in xeval]
re_ssim  = [r["unet_ssim"] for r in retrain]
xe_ssim  = [r["unet_ssim"] for r in xeval]

GREEN, AMBER, GREY = "#2e7d4f", "#c47a1c", "#9a9a9a"

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))

def series(ax, y, color, lab, marker, dy, ref=None, tol=0):
    """ref/tol: skip the label where this series sits within tol of ref (lines overlap).
    dy may be a single offset or a per-point list."""
    ax.plot(x, y, color=color, marker=marker, ms=8, lw=2, label=lab, zorder=3)
    dys = dy if isinstance(dy, (list, tuple)) else [dy] * len(y)
    for xi, yi, d in zip(x, y, dys):
        if ref is not None and abs(yi - ref[xi]) <= tol:
            continue
        ax.annotate(f"{yi:.1f}" if max(y) > 2 else f"{yi:.3f}", (xi, yi),
                    textcoords="offset points", xytext=(0, d), ha="center",
                    fontsize=9, color="#333333")

series(a, re_psnr, GREEN, "U-Net retrained at each dose", "o", 9)
series(a, xe_psnr, AMBER, "U-Net trained noiseless, no retraining", "s", -16, ref=re_psnr, tol=0.5)
a.plot(x, fbp_psnr, color=GREY, ls="--", marker="s", ms=6, lw=1.5, label="FBP input", zorder=2)
a.set_ylim(10, 42); a.set_ylabel("Validation PSNR (dB)")
a.set_title("(a) Reconstruction quality vs dose")

series(b, re_ssim, GREEN, "U-Net retrained at each dose", "o", [9, -16, 9, 9])
series(b, xe_ssim, AMBER, "U-Net trained noiseless, no retraining", "s", -16, ref=re_ssim, tol=0.012)
b.set_ylim(0.75, 1.0); b.set_ylabel("Validation SSIM")
b.set_title("(b) Structural similarity vs dose")

for ax in (a, b):
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Photon count $I_0$ (dose)")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
a.legend(loc="lower right", fontsize=9, frameon=False)

fig.tight_layout()
out = os.path.join(HERE, "fig_noise_response_v2.png")
fig.savefig(out, dpi=200)
print("wrote", out)
print("retrained PSNR", [round(v, 2) for v in re_psnr])
print("no-retrain PSNR", [round(v, 2) for v in xe_psnr])
print("retrained SSIM", [round(v, 3) for v in re_ssim])
print("no-retrain SSIM", [round(v, 3) for v in xe_ssim])
