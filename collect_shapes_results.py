"""Collect the out-of-distribution shape-test results into one table.

Run by scripts\run_shapes.bat from the repository root; output lands in
logs\shapes_results.txt. Safe to run at any time -- missing runs are reported
as (missing), not errors.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "training", "runs")


def load(*parts):
    path = os.path.join(RUNS, *parts, "eval_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


HDR = (f"{'model -> test set':<34}{'FBP dB':>8}{'UNet dB':>9}{'gain':>8}"
       f"{'FBP SSIM':>10}{'UNet SSIM':>11}")
RULE = "-" * len(HDR)


def row(name, s):
    if s is None:
        return f"{name:<34}  (missing)"
    return (f"{name:<34}{s['fbp_psnr']:>8.2f}{s['unet_psnr']:>9.2f}"
            f"{s['unet_psnr'] - s['fbp_psnr']:>+8.2f}"
            f"{s['fbp_ssim']:>10.3f}{s['unet_ssim']:>11.3f}")


print("=== Out-of-distribution geometry (mixed shapes) ===")
print(HDR)
print(RULE)
print(row("ellipse-trained -> ellipses (ref)", load("20260622_154318")))
print(row("ellipse-trained -> SHAPES (a)", load("xeval", "base_on_shapes")))
print(row("shapes-trained  -> shapes (b)", load("shapes800")))
print(row("shapes-trained  -> ellipses", load("xeval", "shapes_on_pilot")))
print()
print("Read row (a) against the reference row: the drop is the price of the")
print("ellipse-only prior on unseen geometry. Read row (b) against (a): how much")
print("of that price shape-diverse training buys back. The last row shows")
print("whether shape diversity cost any in-distribution performance.")
print()
print("Qualitative check: training\\runs\\xeval\\base_on_shapes\\comparison.png")
print("-- look for rounded corners, bent edges, or erased shapes in the")
print("U-Net column wherever the ground truth contains a square or triangle.")
