# Deep Learning for Limited-Angle X-ray CT

Code and results for my MSc dissertation, *Deep Learning for Limited-Angle X-ray
Computed Tomography: Characterising the Reach and Limits of a Learned
Post-Processing Reconstruction* (MSc Artificial Intelligence for Engineering and
Design, University of Bath, September 2026). Supervisor: Prof. Manuchehr
Soleimani.

## Overview

Limited-angle CT arises when a scanner cannot rotate through the full angular
range, as in dental and C-arm systems. Filtered Back Projection (FBP) on this
incomplete data gives images with streaks and missing edges.

This project trains a residual U-Net (FBPConvNet, Jin et al., 2017) to remove
those artefacts from the FBP image. The scan is a 2D parallel-beam protocol at
256x256 with 60 projections over a 120 degree wedge, simulated with the
[TIGRE](https://github.com/CERN/TIGRE) toolbox. Ten experiments (E1 to E10) then
measure where the method works and where it fails. They cover angular severity,
noise, training set size, unseen shapes, real anatomy from LoDoPaB-CT and a
combination with iterative reconstruction.

![Baseline result: limited-angle FBP input, U-Net output and ground truth](training/runs/20260622_154318/comparison.png)

## Main results

PSNR in dB on the held-out validation split (80 images) unless stated. Every
number below can be traced to a `*_summary.json` file in `training/runs/` or to
a table in `logs/`.

| Experiment | Setting | Result |
|---|---|---|
| Baseline | 120 degrees, 800 pairs, seed 42 | FBP 15.94, OS-SART 17.22, U-Net 36.45 (SSIM 0.388 to 0.960) |
| Seeds | seeds 42, 43, 44 | U-Net 36.77 +/- 0.30 |
| E1 | held-out Shepp-Logan phantom | FBP 21.72, OS-SART 26.68, U-Net 17.48 |
| E2 | no residual connection | U-Net 36.89 |
| E3 | Poisson noise on the original attenuation scale | Invalid. `diagnose_noise.py` shows the photon starvation that caused it. Replaced by E7 |
| E4 | 90 and 150 degree wedges, retrained | 33.43 and 40.79 |
| E4 transfer | 120 degree model on 90 and 150 degree data | 16.57 and 20.75 |
| E5 | 200, 400, 800, 1600 training pairs | 33.08, 35.24, 36.45, 38.93 |
| E6 | ellipse-trained model on 300 LoDoPaB-CT slices | FBP 18.10, OS-SART 27.23, U-Net 15.03 |
| E6b | trained on LoDoPaB-CT, fine-tuned and from scratch | 33.26 and 33.52. The fine-tuned model drops to 18.12 on the original phantoms |
| E7 | physical attenuation scale, retrained at I0 = 1e6, 1e5, 1e4 | 37.40, 35.43, 33.09 (noiseless 36.67) |
| E7 transfer | noiseless model on I0 = 1e6, 1e5, 1e4 | 37.11, 35.54, 24.37 |
| E8 | 120 degrees of coverage split into different patterns | one 60 degree gap 37.15, 3x20 degree gaps 38.01, 6x10 degree gaps 38.19 |
| E9 | unseen shapes (squares, triangles) | ellipse-trained 35.30, shapes-trained 36.21 |
| E10 | OS-SART iterations started from the U-Net output (20 images) | phantoms 36.46 to 37.91, LoDoPaB-CT 14.87 to 24.52 |

The short version is that the network works well inside its training
distribution and fails outside it without any visible warning. On real anatomy
the ellipse-trained model scores below its own FBP input on every one of the
300 slices while still producing clean-looking images. Training on matched data
fixes this. Adding iterative steps after the network repairs part of the
failure, but on real data it stays below OS-SART run from zero (27.23), so no
single method wins in both cases.

![E10 trade-off between the learned prior and the measured data](analysis/e10_tradeoff_curves.png)

## Repository layout

```
limited_angle_demo.py      baseline: Shepp-Logan under full and limited angle
generate_dataset.py        synthetic phantom generator (ellipses or mixed shapes)
build_e6_dataset.py        builds the LoDoPaB-CT evaluation and training sets
diagnose_noise.py          E3 post-mortem on the noise model
reconstruct_hybrid.py      E10: U-Net prior combined with OS-SART
collect_*_results.py       gather run summaries into the tables in logs/
training/                  model, dataset, metrics, training and evaluation scripts
training/runs/             one folder per run: config, metrics, curves, figures
analysis/                  figure scripts and the figures used in the dissertation
scripts/                   Windows batch files that ran each experiment queue
logs/                      console logs and result tables from the actual runs
data/                      meta.json for every dataset (the data itself is not stored)
```

`training/README.md` and `training/DOCUMENTATION.md` describe the training code
in more detail.

## Setup

The work was done on Windows with Python 3.10, a CUDA 12.1 build of PyTorch and
a single RTX 4090.

1. Install PyTorch for your GPU from pytorch.org.
2. `pip install -r requirements.txt`
3. Build TIGRE from source by following the instructions in the
   [TIGRE repository](https://github.com/CERN/TIGRE). This project used commit
   `ab0bd51`.

TIGRE is only needed for data generation and the scripts that run classical
reconstructions (`generate_dataset.py`, `build_e6_dataset.py`,
`training/evaluate_e1.py`, `reconstruct_hybrid.py`). Training and the standard
evaluation need PyTorch only.

## Reproducing the baseline

Run everything from the repository root.

```
python limited_angle_demo.py
python generate_dataset.py --n 800 --out data/pilot_v1
python training/train_unet.py --epochs 100 --batch-size 16 --amp
python training/evaluate_unet.py --run training/runs/<run-name>
```

One training run takes about 45 minutes on the RTX 4090.

## Experiments and the scripts that ran them

The batch files expect a conda environment called `tigre` and change to the
repository root themselves, so they can be started from anywhere.

| Experiment | Batch file | Run folders in `training/runs/` |
|---|---|---|
| E1 to E5 | `scripts\run_experiments.bat` | `20260622_154318` (baseline), `e2_no_residual`, `e3_noise_1e5`, `e4_90deg`, `e4_150deg`, `e5_n200`, `e5_n400` |
| E6 | `scripts\run_e6.bat` | `20260622_154318\e6_summary.json` |
| E6b | `scripts\run_e6b.bat` | `e6b_finetune`, `e6b_scratch` |
| E7 | `scripts\run_e7_noise.bat` | `e7_clean`, `e7_i0_1e6`, `e7_i0_1e5`, `e7_i0_1e4` |
| Seeds, E8, 1600 pairs, transfer tests | `scripts\run_week2.bat` | `baseline_seed43`, `baseline_seed44`, `gap30`, `gap60`, `multigap3x20`, `multigap6x10`, `offset45`, `offset90`, `e5_n1600`, `xeval\*` |
| E9 | `scripts\run_e9_shapes.bat` | `shapes800`, `xeval\base_on_shapes`, `xeval\shapes_on_pilot` |
| E10 | `scripts\run_e10_hybrid.bat`, `scripts\run_e10_sweep_deep.bat` | `hybrid_e10` |

Two notes on reading the run folders. The baseline run keeps its original
timestamp name, `20260622_154318`. In `e6b_finetune`, `eval_summary.json` holds
the forgetting test on the original phantoms and `e6_summary.json` holds the
LoDoPaB-CT score.

## What is not in the repository

- The datasets (about 15 GB). They can be regenerated, see `data/README.md`.
- The model checkpoints (about 124 MB per run). Retraining reproduces them.
- TIGRE itself, which is installed separately.

The batch files were moved into `scripts/` for this repository and their paths
were updated to match. In the logs and run configs, the absolute folder path of
my machine was replaced with `.` and nothing else was changed.

## Acknowledgements

Thank you to Prof. Manuchehr Soleimani for supervising the project and for the
introduction to TIGRE. Real CT data comes from the LoDoPaB-CT benchmark, which
is built on the LIDC-IDRI collection.

As declared in the dissertation, Claude was used as an assistive tool for coding
support on the experiment scripts. It was not used to fabricate or analyse any
results or data.

## References

- Jin, K.H., McCann, M.T., Froustey, E. and Unser, M. (2017) 'Deep convolutional
  neural network for inverse problems in imaging', *IEEE Transactions on Image
  Processing*, 26(9), pp. 4509-4522.
- Biguri, A., Dosanjh, M., Hancock, S. and Soleimani, M. (2016) 'TIGRE: a
  MATLAB-GPU toolbox for CBCT image reconstruction', *Biomedical Physics and
  Engineering Express*, 2(5), 055010.
- Leuschner, J., Schmidt, M., Baguer, D.O. and Maass, P. (2021) 'LoDoPaB-CT, a
  benchmark dataset for low-dose computed tomography reconstruction',
  *Scientific Data*, 8, 109.

## Licence

Copyright (c) 2026 Omar El Attar. No open-source licence has been applied yet,
so all rights are reserved for now.
