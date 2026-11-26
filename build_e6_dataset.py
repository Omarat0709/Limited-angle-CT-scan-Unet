"""
E6 dataset builder: real LoDoPaB-CT slices -> limited-angle evaluation set.
===========================================================================

WHAT THIS FILE DOES
-------------------
Takes ground-truth slices from the public LoDoPaB-CT benchmark (real clinical
thorax CT, Leuschner et al. 2021, Zenodo record 3384092), pushes them through
the SAME simulation pipeline as the synthetic datasets (parallel beam, 256x256,
limited-angle wedge, TIGRE forward projection, FBP reconstruction), and stores
them in the exact directory format the rest of the project consumes:

    <out>/inputs/   xxxxx.npy  - limited-angle FBP of the real slice (256x256)
    <out>/targets/  xxxxx.npy  - the real ground-truth slice (256x256, [0,1])
    <out>/ossart/   xxxxx.npy  - OS-SART reconstruction of the same sinogram
                                 (computed here because only this script sees
                                 the projections; enables the three-way
                                 comparison in evaluate_e6.py)
    <out>/meta.json            - provenance + config
    <out>/preview.png          - sanity-check figure (input / OS-SART / GT)

WHY: everything upstream of the network is IDENTICAL to the synthetic pilot
(geometry, angles, reconstruction), so any performance change measured on this
dataset isolates exactly one variable -- the image distribution (real anatomy
vs random ellipses). This is the E6 domain-gap experiment.

DATA LAYOUT (from the Zenodo record)
------------------------------------
Each ground_truth_<part>_%03d.hdf5 file holds an HDF5 dataset "data" of shape
(128, 362, 362); sample n of a part lives in file floor(n/128), row n%128.
Valid sample counts: train 35820, validation 3522, test 3553 (the last file of
each part is zero-padded to 128 rows, so we must not read past the count).

HOW TO RUN (from the repository root, inside the `tigre` env; needs TIGRE + h5py)
-------------------------------------------------------------------
    python build_e6_dataset.py --h5-dir data/lodopab_raw --n 300
    # run_e6.bat does download + extraction + this + evaluation in one click.

Slices are picked evenly across the whole part (not the first N), so many
different patients are represented.
"""

import argparse
import copy
import glob
import json
import os

import numpy as np
import h5py
from skimage.transform import resize

import tigre
import tigre.algorithms as algs

IMG_SIZE = 256
VALID_COUNTS = {"train": 35820, "validation": 3522, "test": 3553}


def make_geometry(size=IMG_SIZE):
    """Same parallel-beam geometry as generate_dataset.py / the baselines."""
    geo = tigre.geometry(mode="parallel",
                         nVoxel=np.array([1, size, size]),
                         default=True)
    geo.nDetector = np.array([1, size])
    geo.sDetector = geo.dDetector * geo.nDetector
    return geo


def load_slice(h5_dir, part, index):
    """Fetch one ground-truth slice by global sample index."""
    fname = os.path.join(h5_dir, f"ground_truth_{part}_{index // 128:03d}.hdf5")
    with h5py.File(fname, "r") as f:
        img = f["data"][index % 128]
    return np.asarray(img, dtype=np.float32)


def prepare_gt(img362, size=IMG_SIZE):
    """LoDoPaB slice (362x362) -> project convention (256x256, [0,1])."""
    img = resize(img362, (size, size), order=1, anti_aliasing=True,
                 preserve_range=True).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def simulate(gt2d, geo_clean, angles, sart_iters):
    """Identical scan simulation to the synthetic pipeline: project the slice
    over the limited wedge, reconstruct with FBP (network input) and OS-SART
    (classical comparator). Fresh geometry copy because TIGRE mutates it."""
    geo = copy.deepcopy(geo_clean)
    vol = gt2d[np.newaxis, :, :].astype(np.float32)
    proj = tigre.Ax(vol, geo, angles)
    fbp = algs.fbp(proj, geo, angles)[0].astype(np.float32)
    sart = algs.ossart(proj, geo, angles, niter=sart_iters)[0].astype(np.float32)
    return fbp, sart


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-dir", default="data/lodopab_raw",
                   help="folder containing ground_truth_<part>_*.hdf5")
    p.add_argument("--part", default="test",
                   choices=list(VALID_COUNTS),
                   help="LoDoPaB split to draw from (test = evaluation)")
    p.add_argument("--n", type=int, default=300, help="number of slices")
    p.add_argument("--out", default=None,
                   help="output dir (default: data/e6_lodopab_<part>)")
    p.add_argument("--limited-deg", type=float, default=120.0)
    p.add_argument("--n-proj", type=int, default=60)
    p.add_argument("--sart-iters", type=int, default=20)
    args = p.parse_args()

    out = args.out or f"data/e6_lodopab_{args.part}"
    n_avail = VALID_COUNTS[args.part]
    n = min(args.n, n_avail)

    files = glob.glob(os.path.join(args.h5_dir,
                                   f"ground_truth_{args.part}_*.hdf5"))
    if not files:
        raise SystemExit(
            f"No ground_truth_{args.part}_*.hdf5 found in {args.h5_dir}.\n"
            "Download ground_truth_test.zip from "
            "https://zenodo.org/records/3384092 and extract it there "
            "(run_e6.bat does this automatically).")

    # Evenly spaced global indices -> spans the full part, many patients.
    indices = np.unique(np.round(np.linspace(0, n_avail - 1, n)).astype(int))

    for sub in ("inputs", "targets", "ossart"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    geo_clean = make_geometry()
    angles = np.linspace(0, np.deg2rad(args.limited_deg), args.n_proj,
                         endpoint=False).astype(np.float32)

    print(f"Building E6 dataset: {len(indices)} real slices from "
          f"LoDoPaB-CT {args.part} | {args.limited_deg:.0f} deg, "
          f"{args.n_proj} projections | out: {out}")

    for k, idx in enumerate(indices):
        gt = prepare_gt(load_slice(args.h5_dir, args.part, int(idx)))
        fbp, sart = simulate(gt, geo_clean, angles, args.sart_iters)
        np.save(os.path.join(out, "inputs", f"{k:05d}.npy"), fbp)
        np.save(os.path.join(out, "targets", f"{k:05d}.npy"), gt)
        np.save(os.path.join(out, "ossart", f"{k:05d}.npy"), sart)
        if (k + 1) % 25 == 0 or (k + 1) == len(indices):
            print(f"  {k + 1}/{len(indices)}")

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({
            "source": "LoDoPaB-CT (Zenodo 3384092), ground truth slices",
            "part": args.part,
            "n": len(indices),
            "sample_indices": indices.tolist(),
            "original_size": 362, "img_size": IMG_SIZE,
            "geometry": "parallel-beam 2D (as limited_angle_demo.py)",
            "limited_angle_deg": args.limited_deg,
            "n_proj_limited": args.n_proj,
            "noise_i0": 0,
            "sart_iters": args.sart_iters,
            "tigre_version": getattr(tigre, "__version__", "unknown"),
        }, f, indent=2)

    # Preview: first 4 slices as (FBP input | OS-SART | GT) rows.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = min(4, len(indices))
        fig, axes = plt.subplots(3, cols, figsize=(3 * cols, 9))
        titles = ["FBP-LA input", "OS-SART", "Ground truth (real)"]
        for c in range(cols):
            for r, sub in enumerate(("inputs", "ossart", "targets")):
                img = np.load(os.path.join(out, sub, f"{c:05d}.npy"))
                ax = axes[r, c] if cols > 1 else axes[r]
                ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if c == 0:
                    ax.set_title(titles[r], fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "preview.png"), dpi=120)
        print(f"Preview saved to {out}/preview.png")
    except Exception as e:
        print(f"Preview skipped: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
