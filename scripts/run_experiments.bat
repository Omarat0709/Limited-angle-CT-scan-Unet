@echo off
REM ============================================================================
REM  run_experiments.bat  --  unattended experiment queue E1-E5
REM  Deep Learning for Limited-Angle CT  (Omar El-Attar, MSc thesis)
REM
REM  WHAT IT RUNS, IN ORDER:
REM    E1  Shepp-Logan held-out test + OS-SART on val set  (baseline run)
REM    E2  Residual ablation           1 training run
REM    E3  Noise robustness            1 dataset + 1 training run
REM    E4  Severity sweep 90/150 deg   2 datasets + 2 training runs
REM    E5  Data-size sweep 200/400     2 datasets + 2 training runs
REM  Each training run is followed by its evaluation automatically.
REM
REM  HOW TO RUN: double-click this file, or from an Anaconda Prompt:
REM      scripts\run_experiments.bat   (from the repository root)
REM  Total runtime: roughly 4.5-5 hours on the RTX 4090. Safe to walk away.
REM  Progress: check logs\queue_status.txt  (one line per finished step).
REM  Already-generated datasets are skipped, so re-running resumes cheaply.
REM ============================================================================

cd /d "%~dp0.."
if not exist logs mkdir logs

REM ---- Activate the tigre conda env (no-op if launched from Anaconda Prompt) --
call conda activate tigre 2>nul

REM ---- Sanity check: env + GPU + TIGRE must be available -----------------------
python -c "import torch, tigre; assert torch.cuda.is_available(), 'no CUDA'" >logs\env_check.log 2>&1
if errorlevel 1 (
    echo [ERROR] Environment check failed - see logs\env_check.log
    echo Open an Anaconda Prompt, then run:
    echo     conda activate tigre
    echo     cd /d "%~dp0.."
    echo     scripts\run_experiments.bat
    pause
    exit /b 1
)

echo ============================================== >> logs\queue_status.txt
echo Queue started  %date% %time% >> logs\queue_status.txt

REM ============================ E1 ==============================================
echo [E1] Shepp-Logan test + OS-SART on val set...
python training\evaluate_e1.py --run training\runs\20260622_154318 >logs\E1.log 2>&1
if errorlevel 1 (echo E1 FAILED - see logs\E1.log >> logs\queue_status.txt) else (echo E1 done  %time% >> logs\queue_status.txt)

REM ============================ E2: residual ablation ===========================
echo [E2] Training no-residual ablation - about 45 min...
python training\train_unet.py --run-name e2_no_residual --epochs 100 --batch-size 16 --amp --no-residual >logs\E2_train.log 2>&1
if errorlevel 1 (echo E2 train FAILED - see logs\E2_train.log >> logs\queue_status.txt & goto e3)
python training\evaluate_unet.py --run training\runs\e2_no_residual >logs\E2_eval.log 2>&1
echo E2 done  %time% >> logs\queue_status.txt

:e3
REM ============================ E3: noise robustness ============================
echo [E3] Noise dataset + training - about 1 hour...
if not exist data\pilot_noise_1e5\meta.json (
    python generate_dataset.py --n 800 --out data/pilot_noise_1e5 --noise-i0 100000 >logs\E3_gen.log 2>&1
)
python training\train_unet.py --data data/pilot_noise_1e5 --run-name e3_noise_1e5 --epochs 100 --batch-size 16 --amp >logs\E3_train.log 2>&1
if errorlevel 1 (echo E3 train FAILED - see logs\E3_train.log >> logs\queue_status.txt & goto e4)
python training\evaluate_unet.py --run training\runs\e3_noise_1e5 --data data/pilot_noise_1e5 >logs\E3_eval.log 2>&1
echo E3 done  %time% >> logs\queue_status.txt

:e4
REM ============================ E4: severity sweep ==============================
echo [E4] 90-deg and 150-deg severity sweep - about 2 hours...
if not exist data\pilot_90deg\meta.json (
    python generate_dataset.py --n 800 --out data/pilot_90deg --limited-deg 90 --n-proj 45 >logs\E4_gen90.log 2>&1
)
python training\train_unet.py --data data/pilot_90deg --run-name e4_90deg --epochs 100 --batch-size 16 --amp >logs\E4_train90.log 2>&1
if errorlevel 1 (echo E4-90 train FAILED - see logs\E4_train90.log >> logs\queue_status.txt & goto e4b)
python training\evaluate_unet.py --run training\runs\e4_90deg --data data/pilot_90deg >logs\E4_eval90.log 2>&1
echo E4-90deg done  %time% >> logs\queue_status.txt

:e4b
if not exist data\pilot_150deg\meta.json (
    python generate_dataset.py --n 800 --out data/pilot_150deg --limited-deg 150 --n-proj 75 >logs\E4_gen150.log 2>&1
)
python training\train_unet.py --data data/pilot_150deg --run-name e4_150deg --epochs 100 --batch-size 16 --amp >logs\E4_train150.log 2>&1
if errorlevel 1 (echo E4-150 train FAILED - see logs\E4_train150.log >> logs\queue_status.txt & goto e5)
python training\evaluate_unet.py --run training\runs\e4_150deg --data data/pilot_150deg >logs\E4_eval150.log 2>&1
echo E4-150deg done  %time% >> logs\queue_status.txt

:e5
REM ============================ E5: data-size sweep =============================
echo [E5] 200-pair and 400-pair training-set sweep - about 1.5 hours...
if not exist data\pilot_200\meta.json (
    python generate_dataset.py --n 200 --out data/pilot_200 >logs\E5_gen200.log 2>&1
)
python training\train_unet.py --data data/pilot_200 --run-name e5_n200 --epochs 100 --batch-size 16 --amp >logs\E5_train200.log 2>&1
if errorlevel 1 (echo E5-200 train FAILED - see logs\E5_train200.log >> logs\queue_status.txt & goto e5b)
python training\evaluate_unet.py --run training\runs\e5_n200 --data data/pilot_200 >logs\E5_eval200.log 2>&1
echo E5-200 done  %time% >> logs\queue_status.txt

:e5b
if not exist data\pilot_400\meta.json (
    python generate_dataset.py --n 400 --out data/pilot_400 >logs\E5_gen400.log 2>&1
)
python training\train_unet.py --data data/pilot_400 --run-name e5_n400 --epochs 100 --batch-size 16 --amp >logs\E5_train400.log 2>&1
if errorlevel 1 (echo E5-400 train FAILED - see logs\E5_train400.log >> logs\queue_status.txt & goto summary)
python training\evaluate_unet.py --run training\runs\e5_n400 --data data/pilot_400 >logs\E5_eval400.log 2>&1
echo E5-400 done  %time% >> logs\queue_status.txt

:summary
REM ============================ Collect all results =============================
python -c "import json,glob,os; rows=[(os.path.basename(os.path.dirname(p)), json.load(open(p))) for p in sorted(glob.glob('training/runs/*/eval_summary.json'))]; print(f\"{'run':<22}{'FBP dB':>8}{'UNet dB':>9}{'gain':>7}{'FBP SSIM':>10}{'UNet SSIM':>11}\"); [print(f\"{n:<22}{d['fbp_psnr']:>8.2f}{d['unet_psnr']:>9.2f}{d['delta_psnr']:>+7.2f}{d['fbp_ssim']:>10.3f}{d['unet_ssim']:>11.3f}\") for n,d in rows]" >logs\all_results.txt 2>&1

echo Queue finished %date% %time% >> logs\queue_status.txt
echo.
echo ==============================================================
echo  ALL DONE. Results:
echo    - per-run folders:   training\runs\e2_... e3_... e4_... e5_...
echo    - E1 outputs:        training\runs\20260622_154318\e1_summary.json + e1_shepp_logan.png
echo    - results table:     logs\all_results.txt
echo    - step log:          logs\queue_status.txt
echo ==============================================================
type logs\all_results.txt
pause
