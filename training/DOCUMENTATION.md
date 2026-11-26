# Software Documentation — Deep Learning Pipeline for Limited-Angle CT

**Project:** Deep Learning for Limited-Angle X-ray CT Imaging
**Component:** FBPConvNet-style U-Net post-processing (training & evaluation)
**Location:** `training/`
**Author:** Omar El Attar (MSc AI for Engineering & Design, University of Bath)

---

## 1. Purpose and context

In limited-angle computed tomography the scanner can only collect projections
over a restricted angular range (here a 120° wedge with 60 projections, a
dental-CT-style scenario) rather than a full 180–360° rotation. Classical
reconstruction by Filtered Back-Projection (FBP) on such incomplete data
produces images riddled with characteristic streaks and "missing-wedge" blur:
edges whose orientation falls inside the unmeasured angular range are simply not
recorded, and the back-projection of the measured rays leaves uncancelled smear.

This software trains a convolutional neural network to **remove those artifacts**
as a post-processing step. The network takes the corrupted FBP image as input
and outputs a cleaned image that approximates the artifact-free ground truth.
The approach follows Jin et al. (2017), *"Deep Convolutional Neural Network for
Inverse Problems in Imaging"*, whose method is known as **FBPConvNet**.

This document explains every module, the design decisions behind them, and how
the pieces fit together. It is written to be read alongside the source files,
which are themselves heavily commented.

---

## 2. Where this sits in the overall project

The thesis pipeline has three stages. This component is the third.

| Stage | Script | Output |
|-------|--------|--------|
| 1. Baseline experiment | `limited_angle_demo.py` | Quantifies the damage on a known phantom; fixes the performance window. |
| 2. Dataset generation | `generate_dataset.py` | 800 (FBP-limited, ground-truth) training pairs in `data/pilot_v1/`. |
| 3. **Learning** | **`training/`** (this component) | A trained U-Net, learning curves, and a quantified improvement over the baseline. |

The **performance window** the network operates in, established by stage 1:

- **Floor — FBP limited-angle (~17.6 dB / 0.234 SSIM on Shepp-Logan):** the raw
  input the network receives. The quality you have with no learning.
- **Classical bar — OS-SART limited-angle (~25.1 dB / 0.819 SSIM):** the best a
  conventional iterative reconstruction achieves. The number to beat.
- **Ceiling — full-angle reconstruction (~35–39 dB):** what is achievable when
  no angles are missing. The theoretical best.

Success for this component is a network that, given an unseen limited-angle FBP
image, returns a reconstruction that climbs well above the floor and ideally
past the classical bar.

---

## 3. The data

`generate_dataset.py` produced 800 pairs of 256×256 float32 images:

- **Input** (`inputs/xxxxx.npy`): a limited-angle FBP reconstruction — streaky,
  blurred, with values that can overshoot outside [0, 1].
- **Target** (`targets/xxxxx.npy`): the ground-truth phantom that produced it,
  with pixel values in [0, 1].

The phantoms are randomised Shepp-Logan-style ellipse drawings (an outer
"skull", an inner region, and 4–9 random internal ellipses). They are synthetic
on purpose: the ground truth is mathematically exact (it does not exist for a
real scan), the data is free of licensing/ethics constraints, and — crucially —
the artifact the network must learn is a property of the *acquisition geometry*,
not the anatomy, so it behaves identically on ellipses and on real tissue. Jin
et al. validated exactly this strategy.

Shepp-Logan itself is deliberately **held out** of the random pilot so it can
serve later as a standard, unseen test phantom.

---

## 4. Architecture of the code

Five Python modules with a clean separation of concerns. None of them import
TIGRE — the simulation is finished; this stage works only on the `.npy` files.

```
training/
├── unet_model.py      the neural network (model definition only)
├── ct_dataset.py      loads .npy pairs, deterministic train/val split
├── metrics.py         PSNR / SSIM, matched to the baseline conventions
├── train_unet.py      the training loop  (entry point)
├── evaluate_unet.py   load a checkpoint, score it, make the result figure
└── README.md          quick-start run instructions
```

Data flows like this:

```
data/pilot_v1/*.npy
      │
      ▼
ct_dataset.py ── build_datasets() ──► train set + val set (deterministic split)
      │
      ▼
train_unet.py ── DataLoader ──► batches ──► FBPConvNet (unet_model.py)
      │                                            │
      │                                   prediction vs ground truth
      │                                            │
      │                                     MSE loss ──► Adam updates weights
      │
      ├──► metrics.py  (PSNR/SSIM on the val set each epoch)
      └──► runs/<name>/  (checkpoints, metrics.csv, learning_curves.png)
                  │
                  ▼
evaluate_unet.py ── best_model.pt ──► eval_summary.json + comparison.png
```

---

## 5. Module-by-module explanation

### 5.1 `unet_model.py` — the network

**What it is.** A U-Net: an encoder that repeatedly shrinks the image while
growing the number of feature channels, a decoder that grows it back, and "skip
connections" that hand fine detail directly across from encoder to decoder.

**Why a U-Net.** Removing a streak that runs across the whole image requires the
network to "see" the whole image at once. Shrinking the spatial resolution in
the encoder is what gives later layers that wide field of view; the decoder then
restores full resolution, and the skip connections return the fine detail that
shrinking would otherwise destroy.

**The shape, concretely.** Four downsampling steps:

```
spatial: 256 → 128 → 64 → 32 → 16 → 32 → 64 → 128 → 256
channels: 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 1
          └──────── encoder ────────┘ └──────── decoder ────────┘
```

(256 must be divisible by 16, which it is.) Each level is a "double convolution"
block — two repetitions of `Conv2d 3×3 → BatchNorm → ReLU` — which is the
standard U-Net building block.

**The key trick: residual learning.** A limited-angle FBP image and the clean
image are *mostly identical* — the object is roughly in the right place, it is
just corrupted. So instead of asking the network to redraw the whole image, we
ask it to predict only the **correction** (the artifact), and we add that back
to the input:

```
output = input + network(input)
```

This is the central idea of FBPConvNet. It makes training dramatically easier
because the network starts from "change nothing" and only has to learn the
difference. The `residual=False` flag disables this, producing a plain U-Net —
useful as an **ablation** to demonstrate in the thesis that the residual
connection actually helps.

The model has ~31 million trainable parameters at `base_channels=64`.

### 5.2 `ct_dataset.py` — feeding data to PyTorch

PyTorch's `DataLoader` (which batches, shuffles, and parallel-loads) needs a
`Dataset` object exposing two methods: `__len__` (how many samples) and
`__getitem__` (return sample *i*). `LimitedAngleCTDataset` provides these,
loading one `.npy` pair off disk per call and adding the channel axis that
convolutions expect: a `(256, 256)` array becomes a `(1, 256, 256)` tensor.

Two design points matter for the thesis:

- **Deterministic split** (`split_pairs`): a seeded shuffle means the same seed
  always yields the same train/val membership. Without this, learning curves
  from two runs would not be comparable, and evaluation could accidentally score
  on images the model trained on.
- **No normalisation**: targets are already in [0, 1] and inputs share that
  scale, so metrics computed at `data_range=1` stay directly comparable to the
  reported FBP/OS-SART baselines.

`list_pairs` includes guards that raise an error on any missing file or
input/target name mismatch — a silent misalignment would train the network on
wrong pairs and quietly ruin the result, so it is made loud instead.

### 5.3 `metrics.py` — measuring quality

Two metrics, both standard in CT reconstruction:

- **PSNR** (decibels): a log-scaled mean-squared-error. Higher is better; about
  +3 dB means "half the error".
- **SSIM** (0–1): structural similarity, closer to human perception of quality.

Every image is run through one helper (`_prep`) that converts to float64 and
clamps to [0, 1] before scoring. Clamping is the fair, standard choice because
both FBP and the network can overshoot outside [0, 1], and the comparison should
be made on displayable images. Using `data_range=1.0` and this exact convention
is what keeps the numbers comparable to the baselines. `batch_psnr_ssim` accepts
either NumPy arrays or torch tensors and both `(N,H,W)` and `(N,1,H,W)` shapes,
so the same function serves training, evaluation, and the baseline measurement.

### 5.4 `train_unet.py` — the training loop (entry point)

This is the script you run. In order, it:

1. **Parses options** — every training choice (`--epochs`, `--batch-size`,
   `--lr`, `--base-channels`, `--amp`, …) is a command-line flag with a sensible
   default, so hyperparameters can be swept without editing code.
2. **Sets seeds** for full reproducibility.
3. **Builds the data** and measures the **FBP baseline on the validation set**
   at startup — computed from the real data, not hardcoded, so the "floor" is
   always honest for this exact split.
4. **Builds the model, loss, optimiser, scheduler:**
   - *Loss:* `MSELoss` (mean squared error). Minimising MSE is equivalent to
     maximising PSNR, which is why it is the training objective.
   - *Optimiser:* `Adam` — performs the gradient-descent weight updates.
   - *Scheduler:* `ReduceLROnPlateau(mode="max")` — when validation PSNR stops
     improving for 8 epochs, it halves the learning rate to squeeze out extra
     accuracy without manual babysitting.
   - *AMP:* optional mixed-precision (`torch.amp`) for speed and lower VRAM on
     the RTX 4090.
5. **Trains.** Each epoch loops over batches: forward pass → compute loss →
   backpropagate gradients → update weights. After every epoch it evaluates on
   the validation set, appends a row to `metrics.csv`, saves `last_model.pt` and
   (if improved) `best_model.pt`, and redraws `learning_curves.png`.

**The `if __name__ == "__main__"` guard is required on Windows**: `DataLoader`
worker processes re-import the script, and without the guard each worker would
re-launch training.

### 5.5 `evaluate_unet.py` — scoring a finished run

Points at a run folder (or a specific checkpoint), rebuilds the model from the
config stored inside the checkpoint, re-creates the **same** validation split
(same seed), and reports mean PSNR/SSIM for both the raw FBP inputs and the
U-Net outputs, plus the gain between them. It also saves `comparison.png` — a
grid with rows *[FBP input | U-Net output | ground truth]* and per-image PSNR —
which is the qualitative result figure for the dissertation.

---

## 6. How to run

From the repository root, inside the `tigre` conda environment (which already
has PyTorch, scikit-image and matplotlib):

```bash
# optional sanity check (under a minute): does the whole pipeline run?
python training/train_unet.py --epochs 3 --batch-size 16 --amp

# full run (~15–30 min on the RTX 4090)
python training/train_unet.py --epochs 100 --batch-size 16 --amp

# evaluate the best checkpoint and produce the result figure
python training/evaluate_unet.py --run training/runs/<run-name>
```

Run as `python training/train_unet.py` (script form), **not**
`python -m training.train_unet` — the modules import each other as siblings.

Each module also self-tests when run directly (e.g. `python training/unet_model.py`
prints the model's input/output shapes and parameter count;
`python training/ct_dataset.py` prints the split sizes and a sample tensor).

---

## 7. Reading the results

The learning curves answer the pilot dataset's core question — **is 800 samples
enough?**

- If **validation PSNR is still climbing** at the final epoch, the network is
  data-limited → generate a larger production dataset (the generation config is
  frozen; only `--n` changes).
- If it **plateaus well before the end**, 800 was sufficient → spend further
  effort on architecture or loss design instead.

The loss plot is the overfitting check: training and validation loss tracking
each other is healthy; validation loss rising while training loss keeps falling
signals overfitting. The PSNR plot draws the FBP floor and the OS-SART bar as
reference lines, so progress is readable at a glance.

---

## 8. Output artifacts (per run)

Everything for a run lands in `training/runs/<run-name>/`:

| File | Contents |
|------|----------|
| `config.json` | The exact settings used (for reproducibility). |
| `metrics.csv` | Per-epoch train loss, val loss, val PSNR, val SSIM, learning rate. |
| `learning_curves.png` | Loss, val PSNR (with reference lines), val SSIM. Redrawn every epoch. |
| `best_model.pt` | Checkpoint with the highest validation PSNR (weights + config + metrics). |
| `last_model.pt` | Final-epoch checkpoint (to resume or continue). |
| `eval_summary.json` | (from evaluate) Mean FBP-vs-U-Net PSNR/SSIM and the gains. |
| `comparison.png` | (from evaluate) The qualitative result figure for the thesis. |

---

## 9. Key references

- K. H. Jin, M. T. McCann, E. Froustey, M. Unser, *"Deep Convolutional Neural
  Network for Inverse Problems in Imaging"*, IEEE Transactions on Image
  Processing, 2017. — The FBPConvNet method this component implements.
- O. Ronneberger, P. Fischer, T. Brox, *"U-Net: Convolutional Networks for
  Biomedical Image Segmentation"*, MICCAI, 2015. — The U-Net architecture.
- A. Biguri et al., *"TIGRE: a MATLAB-GPU toolbox for CBCT image
  reconstruction"*, 2016. — The toolbox used to simulate the data (stages 1–2).
