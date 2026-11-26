"""
Diagnostic: why did the noisy limited-angle reconstructions fog out? (E3 post-mortem)
=====================================================================================

WHAT THIS ANSWERS
-----------------
In E3 the network's outputs retained the outer rim but lost the interior. This
script establishes the cause quantitatively, and it is the evidence for the
answer given to the supervisor:

    With attenuation values in [0, 1] and 1 mm voxels, TIGRE interprets the
    phantom as mu in mm^-1. A ray crossing ~200 mm of phantom therefore
    accumulates a line integral of order 50-200. The transmitted photon count
    is I0 * exp(-p), so for interior rays the expected count underflows to
    far below one photon. generate_dataset.py clamps counts at 1, and
    -log(1/I0) then returns the SAME CONSTANT for every starved ray: the
    interior information is destroyed before the network ever sees it.

    Real soft tissue is about 0.02 mm^-1 at diagnostic energies, so the
    phantom's effective attenuation was roughly fifty times too high.

WHAT IT PRINTS / SAVES
----------------------
  * distribution of line-integral values for a phantom (min/median/max)
  * expected photon counts at each I0 tested, and the FRACTION OF RAYS STARVED
    (expected count < 1, i.e. clamped and information-free)
  * the same statistics after physical rescaling (--mu-water)
  * analysis/noise_diagnostic.png -- sinogram, photon-count map and the
    starved-ray mask, before and after rescaling

HOW TO RUN (from the repository root, inside the `tigre` env)
-----------------------------------------------
    python diagnose_noise.py
    python diagnose_noise.py --mu-water 0.02 --i0 1e4 1e5 1e6
"""

import argparse
import copy

import numpy as np
import tigre
import tigre.algorithms as algs

from generate_dataset import random_ellipse_phantom, make_geometry

IMG_SIZE = 256


def line_integrals(phantom, geo_clean, angles, mu_water=None):
    """Forward project; optionally rescale attenuation to physical units first."""
    geo = copy.deepcopy(geo_clean)
    vol = phantom[np.newaxis, :, :].astype(np.float32)
    if mu_water:
        vol = vol * float(mu_water)          # [0,1] -> [0, mu_water] mm^-1
    return tigre.Ax(vol, geo, angles)


def report(proj, i0_list, label):
    """Print the photon-starvation statistics that explain the fogging."""
    p = proj.ravel()
    print(f"\n--- {label} ---")
    print(f"  line integrals   min {p.min():8.3f}   median {np.median(p):8.3f}   "
          f"max {p.max():8.3f}")
    for i0 in i0_list:
        counts = i0 * np.exp(-p)
        starved = np.mean(counts < 1.0) * 100.0
        weak = np.mean(counts < 100.0) * 100.0
        print(f"  I0 = {i0:>9.0e} : median counts {np.median(counts):11.3e} | "
              f"rays with <1 photon (destroyed): {starved:5.1f}% | "
              f"<100 photons (very noisy): {weak:5.1f}%")
    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limited-deg", type=float, default=120.0)
    p.add_argument("--n-proj", type=int, default=60)
    p.add_argument("--mu-water", type=float, default=0.02,
                   help="physical attenuation of the densest tissue, mm^-1 "
                        "(water/soft tissue at diagnostic energy ~0.02)")
    p.add_argument("--i0", type=float, nargs="+",
                   default=[1e4, 1e5, 1e6, 1e7],
                   help="photon counts to evaluate")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    phantom = random_ellipse_phantom(rng)
    geo_clean = make_geometry()
    angles = np.linspace(0, np.deg2rad(args.limited_deg), args.n_proj,
                         endpoint=False).astype(np.float32)

    print("=" * 74)
    print("E3 POST-MORTEM: why the interior fogged out")
    print("=" * 74)
    print(f"geometry: {IMG_SIZE}x{IMG_SIZE}, 1 mm voxels, "
          f"{args.limited_deg:.0f} deg wedge, {args.n_proj} projections")

    proj_raw = line_integrals(phantom, geo_clean, angles)
    report(proj_raw, args.i0, "AS RUN IN E3 (phantom values used directly as mu)")

    proj_phys = line_integrals(phantom, geo_clean, angles, mu_water=args.mu_water)
    report(proj_phys, args.i0,
           f"AFTER PHYSICAL RESCALING (mu_max = {args.mu_water} mm^-1)")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        os.makedirs("analysis", exist_ok=True)
        i0 = 1e5
        fig, ax = plt.subplots(2, 3, figsize=(14, 8))
        for r, (pr, name) in enumerate([(proj_raw, "as run in E3"),
                                        (proj_phys, f"rescaled (mu={args.mu_water})")]):
            sino = pr[:, 0, :]
            counts = i0 * np.exp(-sino)
            ax[r, 0].imshow(sino, cmap="gray", aspect="auto")
            ax[r, 0].set_title(f"sinogram — {name}\nmax line integral {sino.max():.1f}")
            im = ax[r, 1].imshow(np.log10(np.maximum(counts, 1e-30)), cmap="magma",
                                 aspect="auto")
            ax[r, 1].set_title("log10(expected photon count)")
            plt.colorbar(im, ax=ax[r, 1], fraction=0.046)
            ax[r, 2].imshow(counts < 1.0, cmap="Reds", aspect="auto")
            pct = np.mean(counts < 1.0) * 100
            ax[r, 2].set_title(f"rays with <1 photon (red)\n{pct:.1f}% destroyed")
            for c in range(3):
                ax[r, c].set_xlabel("detector"); ax[r, c].set_ylabel("angle")
        fig.suptitle(f"Photon starvation at I0 = {i0:.0e}: the cause of the "
                     f"fogged interior", fontsize=13)
        fig.tight_layout()
        fig.savefig("analysis/noise_diagnostic.png", dpi=130)
        print("\nSaved: analysis/noise_diagnostic.png")
    except Exception as e:
        print(f"figure skipped: {e}")

    print("\nCONCLUSION")
    print("  If a large fraction of rays register under one photon, the log "
          "transform\n  saturates and the interior carries no information. The "
          "network then returns\n  the conditional mean (a smooth grey disc) — "
          "which is the correct MSE answer\n  to an input with no signal, not a "
          "failure of the architecture.")


if __name__ == "__main__":
    main()
