# U-Net training (FBPConvNet) — limited-angle CT artifact removal

Post-processing U-Net that takes a limited-angle FBP reconstruction and returns
a clean image, following Jin et al. (2017), *FBPConvNet*. This is the next phase
after the baseline experiment and pilot dataset.

## Files

| File | Purpose |
|------|---------|
| `unet_model.py`    | FBPConvNet model: U-Net (64→1024) + global residual skip. |
| `ct_dataset.py`    | Dataset over `data/<set>/{inputs,targets}/*.npy`; deterministic train/val split. |
| `metrics.py`       | PSNR/SSIM matched to the baseline convention (data_range=1, clamp to [0,1]). |
| `train_unet.py`    | Training loop: Adam + MSE, val PSNR/SSIM per epoch, checkpoints, CSV log, learning-curve figure. |
| `evaluate_unet.py` | Load a checkpoint, report FBP vs U-Net PSNR/SSIM, save a thesis comparison figure. |

## Environment

Developed on a single RTX 4090 workstation. The `tigre` conda env already has PyTorch
(cu121), scikit-image and matplotlib, so it works as-is — TIGRE itself is **not**
imported by any training file. A minimal env also works:

```
pip install torch torchvision scikit-image matplotlib numpy
```

## Run it

From the repository root (so the relative `data/pilot_v1` path resolves).
Use the **script form** below, not `python -m training.train_unet` (the modules
import each other as siblings).

```bash
# train (mixed precision recommended on the 4090)
python training/train_unet.py --epochs 100 --batch-size 16 --amp

# evaluate the best checkpoint + make the comparison figure
python training/evaluate_unet.py --run training/runs/<run-name>
```

Useful flags: `--base-channels 32` (halve width if VRAM is tight),
`--no-residual` (plain-U-Net ablation), `--lr`, `--val-frac`, `--num-workers`.

## What to expect

- **Speed**: ~720 training images at 256×256, base-64 U-Net (~31M params). On a
  4090 expect roughly a few seconds per epoch with `--amp`; 100 epochs in a few
  minutes. (If a first run is slow, lower `--num-workers` to 2 on Windows.)
- **Outputs** in `training/runs/<timestamp>/`:
  - `best_model.pt`, `last_model.pt` — checkpoints (state + config + metrics).
  - `metrics.csv` — per-epoch train/val loss, val PSNR, val SSIM, lr.
  - `learning_curves.png` — loss, val PSNR (with FBP-floor and OS-SART-bar
    reference lines), val SSIM.
  - `config.json` — exact run settings.
- **From evaluate**: `eval_summary.json` (mean FBP vs U-Net PSNR/SSIM and the
  gain) and `comparison.png` (input / U-Net / GT columns, per-image PSNR) — the
  qualitative result figure for the dissertation.

## How this connects to the baselines

The pilot's purpose is to fix where the method sits in the performance window:

- **FBP limited-angle ≈ floor** — the input the network receives. `train_unet.py`
  measures this on the actual val set at startup (lands in the mid-teens dB on
  the random-ellipse pilot; ~17.6 dB on Shepp-Logan).
- **OS-SART limited-angle ≈ 25.1 dB / 0.819** — the classical bar to beat.
- **Full-angle ≈ 35–39 dB** — the ceiling.

**Reading the learning curves (the pilot's real question):** if val PSNR is
still climbing when training ends, the network is data-limited → the production
dataset should be larger than 800. If it plateaus well before the last epoch,
800 was enough and effort should go to architecture/loss instead. The generation
config is frozen — only `--n` changes when you scale up.

## Notes / design choices

- **Residual learning** (output = input + network(input)) is the FBPConvNet
  trick: the FBP and clean images are mostly identical, so the network only
  learns the artifact. Toggle off with `--no-residual` for an ablation.
- **No per-image normalisation**: targets are already in [0,1] and FBP inputs
  share that scale, so PSNR/SSIM at data_range=1 stay directly comparable to the
  reported baselines.
- **Shepp-Logan is held out** — it is not in this random-ellipse pilot, so it
  remains a clean, standard, unseen test phantom for a later evaluation.
