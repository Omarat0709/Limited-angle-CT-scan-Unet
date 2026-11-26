"""Collect the Week-2 queue results into one table.

Reads eval_summary.json from the seed-repeat, angle-pattern study, data-efficiency
and cross-eval folders and prints grouped tables. Run by run_week2.bat from
the repository root; output lands in logs\week2_results.txt. Safe to run at
any time -- missing runs are reported as (missing), not errors.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "training", "runs")


def load(*parts):
    path = os.path.join(RUNS, *parts, "eval_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


HDR = (f"{'run':<26}{'FBP dB':>8}{'UNet dB':>9}{'gain':>8}"
       f"{'FBP SSIM':>10}{'UNet SSIM':>11}")
RULE = "-" * len(HDR)


def row(name, s):
    if s is None:
        return f"{name:<26}  (missing)"
    return (f"{name:<26}{s['fbp_psnr']:>8.2f}{s['unet_psnr']:>9.2f}"
            f"{s['unet_psnr'] - s['fbp_psnr']:>+8.2f}"
            f"{s['fbp_ssim']:>10.3f}{s['unet_ssim']:>11.3f}")


print("=== B. Seed robustness (baseline config, pilot_v1) ===")
print(HDR)
print(RULE)
seed_psnrs = []
for name in ["20260622_154318", "baseline_seed43", "baseline_seed44"]:
    s = load(name)
    print(row(name, s))
    if s:
        seed_psnrs.append(s["unet_psnr"])
if len(seed_psnrs) >= 2:
    print(f"U-Net PSNR over {len(seed_psnrs)} seeds: "
          f"{statistics.mean(seed_psnrs):.2f} +/- "
          f"{statistics.stdev(seed_psnrs):.2f} dB (mean +/- sample std)")

print()
print("=== D. Angle-pattern study (0.5 proj/deg throughout) ===")
print("--- 120-deg total coverage (comparator: contiguous baseline) ---")
print(HDR)
print(RULE)
print(row("contiguous (baseline)", load("20260622_154318")))
for name in ["gap60", "multigap3x20", "multigap6x10",
             "offset45", "offset90"]:
    print(row(name, load(name)))
print("--- 150-deg total coverage (comparator: severity_150deg) ---")
print(row("contiguous (150deg)", load("severity_150deg")))
print(row("gap30", load("gap30")))

print()
print("=== E. Data efficiency (data-size sweep + 1600-pair extension) ===")
print(HDR)
print(RULE)
print(row("datasize_n200", load("datasize_n200")))
print(row("datasize_n400", load("datasize_n400")))
print(row("n800 (baseline)", load("20260622_154318")))
print(row("datasize_n1600", load("datasize_n1600")))

print()
print("=== A. Cross-condition evaluations (no retraining) ===")
print(HDR)
print(RULE)
for name in ["base_on_90deg", "base_on_150deg",
             "clean_on_i0_1e6", "clean_on_i0_1e5", "clean_on_i0_1e4"]:
    print(row(name, load("xeval", name)))
print()
print("Reference points: severity_90deg trained-in-domain 33.43 dB; "
      "severity_150deg 40.79 dB;")
print("noise_i0_1e6 37.40 dB; noise_i0_1e5 35.43 dB; noise_i0_1e4 33.09 dB.")
