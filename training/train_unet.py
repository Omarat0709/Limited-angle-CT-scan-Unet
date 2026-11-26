"""
Train the FBPConvNet-style U-Net on the limited-angle CT pilot dataset.
=======================================================================

WHAT THIS FILE DOES, IN ORDER
-----------------------------
1. Load the (FBP-limited, ground-truth) image pairs from disk.
2. Split them deterministically into a training set and a validation set.
3. Measure the FBP baseline on the validation set -- the "floor" the network
   must beat (the quality you already have, with no learning).
4. Build the U-Net and train it: show it batches of inputs, compare its output
   to the ground truth with a loss function, and nudge the weights to reduce
   that loss (this is gradient descent, run by the optimiser).
5. After every epoch (one full pass over the training set), evaluate on the
   validation set, save checkpoints, append to a CSV log, and redraw the
   learning-curve figure.

WHY A TRAIN/VAL SPLIT
---------------------
The training set is what the network learns from. The validation set is held
back and never trained on, so its PSNR/SSIM tell us how well the model does on
images it has NOT seen -- the only honest measure of whether it has learned the
artifact pattern rather than memorised specific pictures.

HOW TO RUN (from the repository root, inside the `tigre` conda env)
-------------------------------------------------------------------
    python training/train_unet.py --epochs 100 --batch-size 16 --amp

Outputs go to training/runs/<run-name>/ :
    best_model.pt        best-by-validation-PSNR checkpoint
    last_model.pt        final-epoch checkpoint (to resume or continue)
    metrics.csv          per-epoch train loss, val loss, val PSNR, val SSIM
    learning_curves.png  the three plots, redrawn every epoch
    config.json          the exact settings this run used
"""

import argparse          # parses the --flags you pass on the command line
import csv               # writes the per-epoch metrics log
import json              # writes config.json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader   # batches + shuffles the dataset

# Sibling modules in this training/ folder. (This is why you run the script as
# `python training/train_unet.py`, not `python -m training.train_unet`.)
from unet_model import FBPConvNet, count_parameters
from ct_dataset import build_datasets
from metrics import batch_psnr_ssim

# Paths are resolved relative to THIS file, so the script works no matter what
# directory you launch it from (as long as data/pilot_v1 sits in the repository root).
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "data", "pilot_v1")
DEFAULT_RUNS = os.path.join(HERE, "runs")


def set_seed(seed):
    """Fix all the random number generators so a run is reproducible: the same
    seed gives the same split, the same weight initialisation, the same shuffle.
    Essential for fair comparisons (e.g. residual vs no-residual ablation)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()   # disables gradient tracking -- evaluation never learns, and
                   # this saves memory and time
def evaluate(model, loader, device, criterion):
    """Run the model over a whole loader and return mean (loss, PSNR, SSIM)."""
    model.eval()   # switches BatchNorm/Dropout to inference behaviour
    losses, psnrs, ssims, counts = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)   # move the batch to GPU (or CPU)
        pred = model(x)
        loss = criterion(pred, y)
        bs = x.size(0)                       # this batch's size
        # Accumulate weighted by batch size so the final mean is correct even if
        # the last batch is smaller than the rest.
        losses += loss.item() * bs
        p, s = batch_psnr_ssim(pred, y)
        psnrs += p * bs
        ssims += s * bs
        counts += bs
    return losses / counts, psnrs / counts, ssims / counts


def baseline_val_metrics(val_loader):
    """PSNR/SSIM of the RAW FBP inputs vs ground truth, on the validation set.

    This is the floor: the quality you have before any deep learning. The whole
    point of the project is to beat it. We compute it on the real data rather
    than hardcoding a number, so the comparison is always honest for this exact
    split (it lands in the mid-teens dB on the random-ellipse pilot)."""
    psnrs, ssims, counts = 0.0, 0.0, 0
    for x, y in val_loader:
        bs = x.size(0)
        p, s = batch_psnr_ssim(x, y)   # note: x (the input) vs y, no model
        psnrs += p * bs
        ssims += s * bs
        counts += bs
    return psnrs / counts, ssims / counts


def plot_curves(metrics_rows, out_path, baseline_psnr=None):
    """Redraw the three diagnostic plots from the metrics collected so far:
    (1) train vs val loss, (2) val PSNR with reference lines, (3) val SSIM."""
    import matplotlib
    matplotlib.use("Agg")          # 'Agg' = write to file, no GUI window needed
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in metrics_rows]
    tr = [r["train_loss"] for r in metrics_rows]
    vl = [r["val_loss"] for r in metrics_rows]
    vp = [r["val_psnr"] for r in metrics_rows]
    vs = [r["val_ssim"] for r in metrics_rows]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # Plot 1: loss. Log scale because loss falls fast then slowly; a log axis
    # keeps the later, smaller improvements visible. train and val tracking each
    # other = healthy; val rising while train falls = overfitting.
    ax[0].plot(ep, tr, label="train")
    ax[0].plot(ep, vl, label="val")
    ax[0].set_title("MSE loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[0].set_yscale("log")

    # Plot 2: validation PSNR, with the FBP floor and the OS-SART bar drawn in
    # so you can read at a glance how far above the floor / past the classical
    # method the network has climbed.
    ax[1].plot(ep, vp, color="tab:green")
    if baseline_psnr is not None:
        ax[1].axhline(baseline_psnr, ls="--", color="gray",
                      label=f"FBP floor ({baseline_psnr:.1f} dB)")
        ax[1].axhline(25.1, ls=":", color="tab:red",
                      label="OS-SART bar (25.1 dB)")
        ax[1].legend()
    ax[1].set_title("Val PSNR (dB)"); ax[1].set_xlabel("epoch")

    # Plot 3: validation SSIM (structural similarity, 0..1).
    ax[2].plot(ep, vs, color="tab:purple")
    ax[2].set_title("Val SSIM"); ax[2].set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)                 # free the figure so memory doesn't pile up


def main():
    # ---- Command-line options ----
    # Every training choice is a flag with a sensible default, so you can sweep
    # hyperparameters without editing the file.
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA, help="dataset dir")
    p.add_argument("--runs-dir", default=DEFAULT_RUNS)
    p.add_argument("--run-name", default=None,
                   help="subfolder name (default: timestamp)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="learning rate: how big each weight update step is")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="L2 regularisation strength (anti-overfitting)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of data held out for validation")
    p.add_argument("--base-channels", type=int, default=64,
                   help="U-Net width; drop to 32 if GPU memory is tight")
    p.add_argument("--no-residual", action="store_true",
                   help="train a plain U-Net (ablation, disables residual skip)")
    p.add_argument("--init-from", default=None,
                   help="checkpoint (.pt) to initialise weights from, for "
                        "fine-tuning (e.g. training/runs/<run>/best_model.pt)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="parallel data-loading processes (use 2 on Windows if it hangs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true",
                   help="mixed precision: faster + less VRAM on the RTX 4090")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    args = p.parse_args()

    # ---- Set up this run's output folder ----
    set_seed(args.seed)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)   # record exactly what we ran

    device = torch.device(args.device)
    print(f"Device: {device} | run dir: {run_dir}")

    # ---- Data ----
    train_ds, val_ds, train_pairs, val_pairs = build_datasets(
        args.data, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    pin = (device.type == "cuda")    # pinned memory speeds up CPU->GPU transfer
    # The DataLoader batches samples, shuffles the training order each epoch, and
    # loads batches in parallel background workers so the GPU is never waiting.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin)

    base_psnr, base_ssim = baseline_val_metrics(val_loader)
    print(f"FBP baseline on val: {base_psnr:.2f} dB / {base_ssim:.3f} SSIM "
          f"(the floor to beat)")

    # ---- Model, loss, optimiser, scheduler ----
    model = FBPConvNet(base_channels=args.base_channels,
                       residual=not args.no_residual).to(device)
    print(f"Model: FBPConvNet | residual={not args.no_residual} | "
          f"{count_parameters(model):,} params")

    # Fine-tuning: start from a previously trained checkpoint instead of
    # random initialisation. The architecture flags above must match the
    # checkpoint's (same base_channels / residual), or loading will fail.
    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=device)
        model.load_state_dict(init_ckpt["model_state"])
        print(f"Initialised weights from {args.init_from} "
              f"(epoch {init_ckpt.get('epoch', '?')}, "
              f"val PSNR {init_ckpt.get('val_psnr', float('nan')):.2f} dB) "
              f"-- fine-tuning")

    # MSELoss = mean squared error between prediction and ground truth, pixel by
    # pixel. Minimising MSE is equivalent to maximising PSNR, which is why we
    # train on it. Adam is the optimiser: it does the actual gradient-descent
    # weight updates, adapting the step size per parameter.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    # If val PSNR stops improving for `patience` epochs, halve the learning rate.
    # mode="max" because higher PSNR is better. This squeezes out extra accuracy
    # late in training without you babysitting the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8)
    # GradScaler supports mixed-precision (AMP) training: it scales the loss to
    # stop tiny float16 gradients from underflowing to zero. (torch.amp is the
    # current API; the old torch.cuda.amp form is deprecated.)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # ---- The training loop ----
    metrics_rows = []
    best_psnr = -1.0
    csv_path = os.path.join(run_dir, "metrics.csv")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()                # BatchNorm in training mode; grads tracked
        running, seen = 0.0, 0
        for x, y in train_loader:    # one iteration = one batch
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)   # clear last step's gradients
            # autocast runs the forward pass in float16 where safe (faster on the
            # 4090's tensor cores). With --amp off it is a no-op.
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = model(x)                      # forward pass
                loss = criterion(pred, y)            # how wrong we were
            scaler.scale(loss).backward()            # backprop: compute gradients
            scaler.step(optimizer)                   # update the weights
            scaler.update()                          # adjust the loss scale
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / seen

        # --- Validation at the end of the epoch ---
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, device,
                                                criterion)
        scheduler.step(val_psnr)                     # maybe drop the LR
        lr_now = optimizer.param_groups[0]["lr"]

        # --- Log this epoch's numbers to memory and to metrics.csv ---
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_psnr": val_psnr, "val_ssim": val_ssim, "lr": lr_now}
        metrics_rows.append(row)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader()
            w.writerows(metrics_rows)

        gain = val_psnr - base_psnr   # how far above the FBP floor we are
        print(f"[{epoch:3d}/{args.epochs}] train {train_loss:.5f} | "
              f"val {val_loss:.5f} | PSNR {val_psnr:.2f} dB "
              f"(+{gain:.2f} vs FBP) | SSIM {val_ssim:.3f} | lr {lr_now:.2e}")

        # --- Save checkpoints ---
        # A checkpoint bundles the weights AND the config/metrics, so it is fully
        # self-describing when you reload it later for evaluation.
        ckpt = {"model_state": model.state_dict(),
                "config": vars(args),
                "epoch": epoch,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "baseline_psnr": base_psnr,
                "baseline_ssim": base_ssim}
        torch.save(ckpt, os.path.join(run_dir, "last_model.pt"))
        if val_psnr > best_psnr:      # keep the best-so-far separately
            best_psnr = val_psnr
            torch.save(ckpt, os.path.join(run_dir, "best_model.pt"))

        # Redraw the curves every epoch so you can watch progress live.
        plot_curves(metrics_rows, os.path.join(run_dir, "learning_curves.png"),
                    baseline_psnr=base_psnr)

    dt = time.time() - t_start
    print(f"\nDone in {dt/60:.1f} min. Best val PSNR: {best_psnr:.2f} dB "
          f"(FBP floor {base_psnr:.2f}, OS-SART bar 25.1).")
    print(f"Artifacts in: {run_dir}")


if __name__ == "__main__":
    # The __main__ guard is REQUIRED on Windows: DataLoader worker processes
    # re-import this file, and without the guard they would each re-run training.
    main()
