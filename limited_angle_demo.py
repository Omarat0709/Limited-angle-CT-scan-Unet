"""
limited_angle_demo.py
=====================
Phase 3 starter for "Deep Learning for Limited Angle X-ray CT Imaging"
(Omar El Attar, MSc Dissertation, University of Bath).

This script:
  1. Builds a 2D Shepp-Logan phantom (the standard CT testbench).
  2. Defines a parallel-beam geometry in TIGRE.
  3. Simulates a FULL-angle sinogram (ground-truth scan) and reconstructs it
     with FBP and SART  -> these are our "good" baselines.
  4. Simulates a LIMITED-angle sinogram (dental-CT-style 120 degree wedge)
     and reconstructs the same way -> these are our "bad" inputs.
  5. Computes PSNR and SSIM of every reconstruction against the phantom.
  6. Plots phantom / sinograms / reconstructions side by side.

Run inside your (tigre) conda env:
    python limited_angle_demo.py
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import tigre
import tigre.algorithms as algs
from tigre.utilities import sample_loader   # ships with TIGRE; gives us phantoms


# ----------------------------------------------------------------------
# 1.  GEOMETRY
# ----------------------------------------------------------------------
# We use a parallel-beam geometry because the math is cleaner and it matches
# how most CT-DL papers (Jin et al. 2017 FBPConvNet etc.) report results.
# For a CBCT or fan-beam study later, switch to tigre.geometry(mode="cone").

N = 256                                    # image is N x N pixels
geo = tigre.geometry(mode="parallel",
                     nVoxel=np.array([1, N, N]),   # TIGRE volumes are 3-D, depth=1 for 2-D
                     default=True)
geo.nDetector = np.array([1, N])          # 1-pixel-tall detector for a 2-D slice
geo.sDetector = geo.dDetector * geo.nDetector
geo_clean = copy.deepcopy(geo)             # snapshot before tigre mutates it


# ----------------------------------------------------------------------
# 2.  PHANTOM (ground truth)
# ----------------------------------------------------------------------
# Shepp-Logan is the classic CT test image: a brain-like ellipse arrangement
# with sharp edges and low-contrast features that stress reconstruction algorithms.
phantom_2d = sample_loader.load_head_phantom(geo.nVoxel)[0]    # take central slice
phantom = phantom_2d[np.newaxis, :, :].astype(np.float32)      # add depth axis -> (1, N, N)


# ----------------------------------------------------------------------
# 3.  FULL-ANGLE SCAN  (the ideal reference)
# ----------------------------------------------------------------------
n_full = 180                                            # 180 angles over 180 deg
angles_full = np.linspace(0, np.pi, n_full, endpoint=False).astype(np.float32)

# Forward project: this is the "A" operator in Ax = b
proj_full = tigre.Ax(phantom, geo, angles_full)

# Reconstruct from full data with FBP and SART
fbp_full  = algs.fbp(proj_full, geo, angles_full)
sart_full = algs.ossart(proj_full, geo, angles_full, niter=20)


# ----------------------------------------------------------------------
# 4.  LIMITED-ANGLE SCAN  (the project's actual input)
# ----------------------------------------------------------------------
# Dental CT scenario: only a 120 degree wedge of angles available.
# This is what your U-Net will learn to "complete".
n_limited = 60
angles_limited = np.linspace(0, np.deg2rad(120), n_limited, endpoint=False).astype(np.float32)

geo_limited = copy.deepcopy(geo_clean)     # fresh copy — geo is already mutated to (180,3)
proj_limited = tigre.Ax(phantom, geo_limited, angles_limited)
fbp_limited  = algs.fbp(proj_limited, geo_limited, angles_limited)
sart_limited = algs.ossart(proj_limited, geo_limited, angles_limited, niter=20)


# ----------------------------------------------------------------------
# 5.  METRICS
# ----------------------------------------------------------------------
gt = phantom[0]                                         # 2-D ground truth
data_range = gt.max() - gt.min()                        # for SSIM/PSNR

def evaluate(recon, name):
    img = recon[0]                                      # drop depth axis
    p = psnr(gt, img, data_range=data_range)
    s = ssim(gt, img, data_range=data_range)
    print(f"  {name:18s}  PSNR = {p:6.2f} dB   SSIM = {s:.4f}")
    return p, s

print("\n--- Full-angle reconstructions (ideal case) ---")
evaluate(fbp_full,    "FBP  (180 angles)")
evaluate(sart_full,   "SART (180 angles)")
print("--- Limited-angle reconstructions (your project's domain) ---")
evaluate(fbp_limited,  "FBP  (60 / 120 deg)")
evaluate(sart_limited, "SART (60 / 120 deg)")


# ----------------------------------------------------------------------
# 6.  PLOT
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

axes[0,0].imshow(gt,            cmap="gray");                axes[0,0].set_title("Ground truth phantom")
axes[0,1].imshow(proj_full[:,0,:], cmap="gray", aspect="auto"); axes[0,1].set_title("Full sinogram (180 deg)")
axes[0,2].imshow(fbp_full[0],   cmap="gray");                axes[0,2].set_title("FBP – full")
axes[0,3].imshow(sart_full[0],  cmap="gray");                axes[0,3].set_title("SART – full")

axes[1,0].imshow(gt,            cmap="gray");                axes[1,0].set_title("Ground truth phantom")
axes[1,1].imshow(proj_limited[:,0,:], cmap="gray", aspect="auto"); axes[1,1].set_title("Limited sinogram (120 deg)")
axes[1,2].imshow(fbp_limited[0], cmap="gray");               axes[1,2].set_title("FBP – limited")
axes[1,3].imshow(sart_limited[0], cmap="gray");              axes[1,3].set_title("SART – limited")

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Full vs Limited-Angle CT  (Shepp-Logan, parallel beam)", fontsize=14)
plt.tight_layout()
plt.show()
