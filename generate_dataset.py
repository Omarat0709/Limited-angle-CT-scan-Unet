"""
Multi-phantom dataset generator for limited-angle CT deep learning.

Generates N random ellipse phantoms (Shepp-Logan-style), simulates
limited-angle parallel-beam CT with TIGRE, and saves (FBP_limited, ground_truth)
training pairs for the FBPConvNet-style U-Net.

Geometry and angle settings are matched to limited_angle_demo.py (the baseline
script): 256x256, parallel beam, 60 projections over a 120-degree wedge.

Run on the Windows machine inside the `tigre` conda env:
    python generate_dataset.py --n 800 --out data/pilot_v1

Outputs:
    <out>/inputs/   xxxx.npy  - limited-angle FBP reconstruction (float32, 256x256)
    <out>/targets/  xxxx.npy  - ground-truth phantom            (float32, 256x256)
    <out>/meta.json           - generation config for reproducibility
    <out>/preview.png         - 4 sample pairs for sanity checking
"""

import argparse
import copy
import json
import os
import numpy as np

import tigre
import tigre.algorithms as algs

# ----------------------------- config defaults -----------------------------
# Matched to limited_angle_demo.py (the baseline script): 60 projections
# over a 120-degree wedge, 256x256, parallel beam.

IMG_SIZE = 256
LIMITED_ANGLE_DEG = 120.0       # baseline: 120-degree dental-CT-style wedge
N_PROJ_LIMITED = 60             # baseline: 60 projections over that wedge
NOISE_I0 = 0                    # 0 = noiseless; e.g. 1e5 for Poisson noise
SEED = 42

# ----------------------------- phantom generation --------------------------

def random_ellipse_phantom(rng, size=IMG_SIZE):
    """Random Shepp-Logan-style phantom: outer skull ellipse, brain tissue,
    4-9 random internal ellipses. Values in [0, 1]."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size),
                         indexing="ij")
    img = np.zeros((size, size), dtype=np.float32)

    # Outer "skull" ellipse
    a_out = rng.uniform(0.75, 0.92)
    b_out = rng.uniform(0.60, 0.88)
    theta = rng.uniform(0, np.pi)
    skull = _ellipse_mask(xx, yy, 0.0, 0.0, a_out, b_out, theta)
    img[skull] = 1.0

    # Inner "brain" ellipse, slightly smaller, lower intensity
    inner = _ellipse_mask(xx, yy, 0.0, 0.0, a_out * 0.92, b_out * 0.92, theta)
    img[inner] = rng.uniform(0.15, 0.35)

    # Random internal structures
    n_ellipses = rng.integers(4, 10)
    for _ in range(n_ellipses):
        cx = rng.uniform(-0.5, 0.5)
        cy = rng.uniform(-0.5, 0.5)
        a = rng.uniform(0.04, 0.35)
        b = rng.uniform(0.04, 0.35)
        th = rng.uniform(0, np.pi)
        delta = rng.uniform(-0.4, 0.5)  # intensity perturbation
        e = _ellipse_mask(xx, yy, cx, cy, a, b, th) & inner
        img[e] = np.clip(img[e] + delta, 0.0, 1.0)

    return img


def _ellipse_mask(xx, yy, cx, cy, a, b, theta):
    ct, st = np.cos(theta), np.sin(theta)
    x = (xx - cx) * ct + (yy - cy) * st
    y = -(xx - cx) * st + (yy - cy) * ct
    return (x / a) ** 2 + (y / b) ** 2 <= 1.0


def _rect_mask(xx, yy, cx, cy, a, b, theta):
    """Rotated rectangle; a and b are the half-widths along its own axes."""
    ct, st = np.cos(theta), np.sin(theta)
    x = (xx - cx) * ct + (yy - cy) * st
    y = -(xx - cx) * st + (yy - cy) * ct
    return (np.abs(x) <= a) & (np.abs(y) <= b)


def _triangle_mask(xx, yy, verts):
    """Point-in-triangle via the half-plane sign test.

    verts: (3, 2) array of (x, y) vertices. A point is inside when the three
    cross-product signs agree (all >= 0 or all <= 0)."""
    def sign(p1, p2):
        return ((xx - p2[0]) * (p1[1] - p2[1])
                - (p1[0] - p2[0]) * (yy - p2[1]))
    d1 = sign(verts[0], verts[1])
    d2 = sign(verts[1], verts[2])
    d3 = sign(verts[2], verts[0])
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def random_shapes_phantom(rng, size=IMG_SIZE):
    """Like random_ellipse_phantom, but the internal structures are drawn from
    a MIXED shape family: ellipse, square, rectangle, triangle (uniformly).

    Purpose (out-of-distribution geometry test): the baseline network has
    only ever seen elliptical structures, so its learned prior may 'ellipsify'
    anything it reconstructs -- rounding corners, bending edges, or erasing
    unfamiliar shapes -- especially where the missing wedge removes exactly the
    directional frequencies that sharp corners need. Squares and triangles
    make that failure mode visible and measurable. Skull/brain outline and
    intensity conventions are identical to random_ellipse_phantom, so results
    are comparable like-for-like."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size),
                         indexing="ij")
    img = np.zeros((size, size), dtype=np.float32)

    # Outer "skull" and inner "brain" ellipses -- same as the ellipse phantom
    a_out = rng.uniform(0.75, 0.92)
    b_out = rng.uniform(0.60, 0.88)
    theta = rng.uniform(0, np.pi)
    skull = _ellipse_mask(xx, yy, 0.0, 0.0, a_out, b_out, theta)
    img[skull] = 1.0
    inner = _ellipse_mask(xx, yy, 0.0, 0.0, a_out * 0.92, b_out * 0.92, theta)
    img[inner] = rng.uniform(0.15, 0.35)

    # Random internal structures, mixed shape family
    n_shapes = rng.integers(4, 10)
    for _ in range(n_shapes):
        cx = rng.uniform(-0.5, 0.5)
        cy = rng.uniform(-0.5, 0.5)
        th = rng.uniform(0, np.pi)
        delta = rng.uniform(-0.4, 0.5)  # same intensity perturbation range
        kind = rng.choice(["ellipse", "square", "rectangle", "triangle"])
        if kind == "ellipse":
            m = _ellipse_mask(xx, yy, cx, cy,
                              rng.uniform(0.04, 0.35),
                              rng.uniform(0.04, 0.35), th)
        elif kind == "square":
            s = rng.uniform(0.04, 0.28)
            m = _rect_mask(xx, yy, cx, cy, s, s, th)
        elif kind == "rectangle":
            m = _rect_mask(xx, yy, cx, cy,
                           rng.uniform(0.06, 0.35),
                           rng.uniform(0.03, 0.18), th)
        else:  # triangle: 3 vertices at random angles/radii around the centre
            angles = np.sort(rng.uniform(0, 2 * np.pi, size=3))
            radii = rng.uniform(0.06, 0.35, size=3)
            verts = np.stack([cx + radii * np.cos(angles),
                              cy + radii * np.sin(angles)], axis=1)
            m = _triangle_mask(xx, yy, verts)
        m = m & inner
        img[m] = np.clip(img[m] + delta, 0.0, 1.0)

    return img

# ----------------------------- TIGRE simulation ----------------------------

def make_geometry(size=IMG_SIZE):
    """Same geometry as limited_angle_demo.py."""
    geo = tigre.geometry(mode="parallel",
                         nVoxel=np.array([1, size, size]),
                         default=True)
    geo.nDetector = np.array([1, size])
    geo.sDetector = geo.dDetector * geo.nDetector
    return geo


def build_angles(mode="contiguous", limited_deg=LIMITED_ANGLE_DEG,
                 n_proj=N_PROJ_LIMITED, offset_deg=0.0, gap_deg=20.0,
                 n_gaps=3, full_deg=180.0):
    """Return the acquisition angles (radians) for a given sampling pattern.

    Modes
    -----
    contiguous : the original protocol -- one wedge of `limited_deg`, starting
                 at `offset_deg`. offset_deg rotates WHERE the measured wedge
                 sits, which is the 'missing wedge in a different place' test.
    gap        : a nearly complete scan of `full_deg` with ONE missing chunk of
                 `gap_deg` beginning at `offset_deg` (small data loss test).
    multigap   : a complete scan of `full_deg` interrupted by `n_gaps` missing
                 chunks of `gap_deg`, spread evenly around the range
                 (detector dropout / obstructed scan).

    Projection density is held at n_proj/limited_deg for `contiguous`, and for
    the gap modes the same density is applied over the retained angles, so the
    number of measurements scales with what is actually acquired.
    """
    if mode == "contiguous":
        a = np.linspace(0.0, limited_deg, n_proj, endpoint=False) + offset_deg
        return np.deg2rad(a).astype(np.float32)

    density = n_proj / float(limited_deg)          # projections per degree
    n_full = int(round(full_deg * density))
    all_deg = np.linspace(0.0, full_deg, n_full, endpoint=False)

    if mode == "gap":
        gaps = [(offset_deg, offset_deg + gap_deg)]
    elif mode == "multigap":
        starts = np.linspace(0.0, full_deg, n_gaps, endpoint=False) + offset_deg
        gaps = [(s, s + gap_deg) for s in starts]
    else:
        raise ValueError(f"unknown angle mode: {mode}")

    keep = np.ones_like(all_deg, dtype=bool)
    for g0, g1 in gaps:
        # wrap into [0, full_deg) so gaps may straddle the end of the range
        rel = np.mod(all_deg - g0, full_deg)
        keep &= ~(rel < (g1 - g0))
    return np.deg2rad(all_deg[keep]).astype(np.float32)


def simulate_pair(phantom, geo_clean, angles_limited, i0=NOISE_I0, rng=None,
                  mu_water=0.0):
    """Project phantom over the given angles, add optional Poisson noise,
    reconstruct with FBP. Returns (fbp_limited, phantom).

    mu_water : if > 0, the phantom (values in [0,1]) is rescaled to physical
        attenuation coefficients with this value as the maximum, in mm^-1,
        BEFORE projection, and the reconstruction is rescaled back afterwards.
        This matters enormously for the noise model: with 1 mm voxels and
        mu in [0,1] the line integrals reach ~200, so I0*exp(-p) underflows
        and every interior ray is photon-starved (see diagnose_noise.py).
        Soft tissue at diagnostic energies is about 0.02 mm^-1.
    """
    geo = copy.deepcopy(geo_clean)  # TIGRE mutates geo in Ax - use fresh copy
    vol = phantom[np.newaxis, :, :].astype(np.float32)  # (1, H, W)
    scale = float(mu_water) if mu_water and mu_water > 0 else 1.0
    proj = tigre.Ax(vol * scale, geo, angles_limited)

    if i0 and i0 > 0 and rng is not None:
        # Poisson noise in the transmission domain
        counts = i0 * np.exp(-proj)
        counts = rng.poisson(np.maximum(counts, 1e-12)).astype(np.float32)
        proj = -np.log(np.maximum(counts, 1) / i0).astype(np.float32)

    rec = algs.fbp(proj, geo, angles_limited)
    return (rec[0] / scale).astype(np.float32), phantom

# ----------------------------- main ----------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=800, help="number of phantoms")
    p.add_argument("--out", type=str, default="data/pilot_v1")
    p.add_argument("--limited-deg", type=float, default=LIMITED_ANGLE_DEG)
    p.add_argument("--n-proj", type=int, default=N_PROJ_LIMITED,
                   help="projections over the limited range (baseline: 60)")
    p.add_argument("--noise-i0", type=float, default=NOISE_I0)
    p.add_argument("--mu-water", type=float, default=0.0,
                   help="rescale phantom to physical attenuation with this "
                        "maximum in mm^-1 before projection (0.02 = soft "
                        "tissue). REQUIRED for a meaningful noise model; see "
                        "diagnose_noise.py")
    p.add_argument("--angle-mode", default="contiguous",
                   choices=["contiguous", "gap", "multigap"],
                   help="angular sampling pattern")
    p.add_argument("--offset-deg", type=float, default=0.0,
                   help="rotates the measured wedge (contiguous) or the first "
                        "missing chunk (gap/multigap)")
    p.add_argument("--gap-deg", type=float, default=20.0,
                   help="width of each missing chunk, gap/multigap modes")
    p.add_argument("--n-gaps", type=int, default=3,
                   help="number of missing chunks, multigap mode")
    p.add_argument("--full-deg", type=float, default=180.0,
                   help="nominal complete angular range for gap modes")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--shapes", action="store_true",
                   help="internal structures drawn from a mixed shape family "
                        "(ellipse/square/rectangle/triangle) instead of "
                        "ellipses only -- the out-of-distribution "
                        "geometry test")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "targets"), exist_ok=True)

    geo_clean = make_geometry()
    n_proj_limited = args.n_proj
    angles_limited = build_angles(
        mode=args.angle_mode, limited_deg=args.limited_deg,
        n_proj=args.n_proj, offset_deg=args.offset_deg,
        gap_deg=args.gap_deg, n_gaps=args.n_gaps, full_deg=args.full_deg)

    print(f"Generating {args.n} pairs | mode={args.angle_mode} | "
          f"{len(angles_limited)} projections | "
          f"angles {np.rad2deg(angles_limited.min()):.1f}-"
          f"{np.rad2deg(angles_limited.max()):.1f} deg | "
          f"noise I0={args.noise_i0} | mu_water={args.mu_water}")

    phantom_fn = random_shapes_phantom if args.shapes else random_ellipse_phantom
    for i in range(args.n):
        phantom = phantom_fn(rng)
        fbp_la, gt = simulate_pair(phantom, geo_clean, angles_limited,
                                   i0=args.noise_i0, rng=rng,
                                   mu_water=args.mu_water)
        np.save(os.path.join(args.out, "inputs", f"{i:05d}.npy"), fbp_la)
        np.save(os.path.join(args.out, "targets", f"{i:05d}.npy"), gt)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n}")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "n": args.n, "img_size": IMG_SIZE,
            "geometry": "parallel-beam 2D (as limited_angle_demo.py)",
            "limited_angle_deg": args.limited_deg,
            "n_proj_limited": len(angles_limited),
            "n_proj_requested": n_proj_limited,
            "angle_mode": args.angle_mode,
            "offset_deg": args.offset_deg,
            "gap_deg": args.gap_deg,
            "n_gaps": args.n_gaps,
            "full_deg": args.full_deg,
            "angles_deg": np.rad2deg(angles_limited).round(3).tolist(),
            "noise_i0": args.noise_i0,
            "mu_water": args.mu_water,
            "shapes": bool(args.shapes),
            "seed": args.seed,
            "tigre_version": getattr(tigre, "__version__", "unknown"),
        }, f, indent=2)

    # Preview figure: 4 random pairs
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        idx = rng.choice(args.n, size=min(4, args.n), replace=False)
        fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6))
        for k, j in enumerate(idx):
            inp = np.load(os.path.join(args.out, "inputs", f"{j:05d}.npy"))
            tgt = np.load(os.path.join(args.out, "targets", f"{j:05d}.npy"))
            axes[0, k].imshow(inp, cmap="gray"); axes[0, k].set_title(f"FBP-LA #{j}")
            axes[1, k].imshow(tgt, cmap="gray"); axes[1, k].set_title(f"GT #{j}")
            for ax in (axes[0, k], axes[1, k]):
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "preview.png"), dpi=120)
        print(f"Preview saved to {args.out}/preview.png")
    except Exception as e:  # preview is non-critical
        print(f"Preview skipped: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
