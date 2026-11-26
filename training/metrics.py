"""
Image-quality metrics for limited-angle CT reconstruction.
==========================================================

WHAT THESE NUMBERS MEAN
-----------------------
* PSNR (Peak Signal-to-Noise Ratio, in decibels): essentially a log-scaled
  version of mean squared error. Higher is better; +3 dB is roughly "half the
  error". It is the headline number in CT reconstruction papers.
* SSIM (Structural Similarity, 0..1): closer to how a human perceives image
  quality -- it rewards getting structure, contrast and luminance right, not
  just per-pixel closeness. 1.0 is a perfect match.

WHY THE CONVENTIONS HERE MATTER
-------------------------------
The ground truth lives in [0, 1], so we use a fixed data_range of 1.0, and we
clamp the reconstruction/prediction to [0, 1] before scoring. Clamping is the
standard, fair choice: both FBP and the U-Net can overshoot outside [0, 1], and
the comparison should be made on displayable images. Using the SAME convention
as limited_angle_demo.py keeps every number directly comparable to the reported
baselines:
    FBP limited-angle    ~17.6 dB / 0.234 SSIM   (the floor the U-Net receives)
    OS-SART limited-angle ~25.1 dB / 0.819 SSIM  (the classical bar to beat)
    full-angle ceiling   ~35-39 dB
(Those figures were measured on Shepp-Logan; on the random-ellipse pilot the FBP
floor is a bit different, which is expected -- different images.)
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

DATA_RANGE = 1.0   # ground truth is in [0, 1], so the max possible value is 1


def _prep(img):
    """Standardise an image before scoring: to float64 numpy, clamped to [0,1].
    Every metric below runs inputs through this so the convention is applied in
    exactly one place."""
    arr = np.asarray(img, dtype=np.float64)
    return np.clip(arr, 0.0, 1.0)


def psnr(pred, target, data_range=DATA_RANGE):
    """PSNR of one image vs its ground truth (skimage convention: ref first)."""
    return float(sk_psnr(_prep(target), _prep(pred), data_range=data_range))


def ssim(pred, target, data_range=DATA_RANGE):
    """SSIM of one image vs its ground truth."""
    return float(sk_ssim(_prep(target), _prep(pred), data_range=data_range))


def psnr_ssim(pred, target, data_range=DATA_RANGE):
    """Both metrics for a single 2D image pair, returned as (psnr, ssim).
    Slightly cheaper than calling psnr() and ssim() separately."""
    p = _prep(pred)
    t = _prep(target)
    return (float(sk_psnr(t, p, data_range=data_range)),
            float(sk_ssim(t, p, data_range=data_range)))


def batch_psnr_ssim(preds, targets, data_range=DATA_RANGE):
    """Mean PSNR/SSIM over a whole batch.

    This is the version the training/eval loops call. It accepts either numpy
    arrays or torch tensors, and either (N, H, W) or (N, 1, H, W) shape -- the
    latter is what comes straight out of the model -- and averages the per-image
    metrics. skimage works on one 2D image at a time, hence the loop."""
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)
    if preds.ndim == 4:            # (N, 1, H, W) -> drop the channel axis
        preds = preds[:, 0]
        targets = targets[:, 0]
    ps, ss = [], []
    for p, t in zip(preds, targets):
        a, b = psnr_ssim(p, t, data_range=data_range)
        ps.append(a)
        ss.append(b)
    return float(np.mean(ps)), float(np.mean(ss))


def _to_numpy(x):
    """Accept a torch tensor (possibly on GPU) or an array; return a CPU numpy
    array. detach() drops gradient tracking; .cpu() pulls it off the GPU."""
    if hasattr(x, "detach"):       # quacks like a torch tensor
        return x.detach().cpu().float().numpy()
    return np.asarray(x)
