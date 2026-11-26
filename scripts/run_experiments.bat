@echo off
REM ============================================================================
REM  run_experiments.bat  --  unattended queue for the first five experiments
REM  Deep Learning for Limited-Angle CT  (Omar El-Attar, MSc thesis)
REM
REM  WHAT IT RUNS, IN ORDER:
REM    1.  Shepp-Logan held-out test + OS-SART on val set  (baseline run)
REM    2.  Residual ablation           1 training run
REM    3.  Noise robustness            1 dataset + 1 training run
REM    4.  Severity sweep 90/150 deg   2 datasets + 2 training runs
REM    5.  Data-size sweep 200/400     2 datasets + 2 training runs
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

REM ============================ classical comparison ==============================================
echo [classical] Shepp-Logan test + OS-SART on val set...
python training\evaluate_classical.py --run training\runs\20260622_154318 >logs\classical.log 2>&1
if errorlevel 1 (echo classical comparison FAILED - see logs\classical.log >> logs\queue_status.txt) else (echo classical comparison done  %time% >> logs\queue_status.txt)

REM ============================ residual ablation ===========================
echo [ablation] Training no-residual ablation - about 45 min...
python training\train_unet.py --run-name ablation_no_residual --epochs 100 --batch-size 16 --amp --no-residual >logs\ablation_train.log 2>&1
if errorlevel 1 (echo residual ablation train FAILED - see logs\ablation_train.log >> logs\queue_status.txt & goto noise_step)
python training\evaluate_unet.py --run training\runs\ablation_no_residual >logs\ablation_eval.log 2>&1
echo residual ablation done  %time% >> logs\queue_status.txt

:noise_step
REM ============================ noise robustness ============================
echo [noise-v1] Noise dataset + training - about 1 hour...
if not exist data\pilot_noise_1e5\meta.json (
    python generate_dataset.py --n 800 --out data/pilot_noise_1e5 --noise-i0 100000 >logs\noise_unscaled_gen.log 2>&1
)
python training\train_unet.py --data data/pilot_noise_1e5 --run-name noise_unscaled_i0_1e5 --epochs 100 --batch-size 16 --amp >logs\noise_unscaled_train.log 2>&1
if errorlevel 1 (echo first noise run train FAILED - see logs\noise_unscaled_train.log >> logs\queue_status.txt & goto sev90_step)
python training\evaluate_unet.py --run training\runs\noise_unscaled_i0_1e5 --data data/pilot_noise_1e5 >logs\noise_unscaled_eval.log 2>&1
echo first noise run done  %time% >> logs\queue_status.txt

:sev90_step
REM ============================ severity sweep ==============================
echo [severity] 90-deg and 150-deg severity sweep - about 2 hours...
if not exist data\pilot_90deg\meta.json (
    python generate_dataset.py --n 800 --out data/pilot_90deg --limited-deg 90 --n-proj 45 >logs\severity_gen90.log 2>&1
)
python training\train_unet.py --data data/pilot_90deg --run-name severity_90deg --epochs 100 --batch-size 16 --amp >logs\severity_train90.log 2>&1
if errorlevel 1 (echo severity-90 train FAILED - see logs\severity_train90.log >> logs\queue_status.txt & goto sev150_step)
python training\evaluate_unet.py --run training\runs\severity_90deg --data data/pilot_90deg >logs\severity_eval90.log 2>&1
echo severity-90deg done  %time% >> logs\queue_status.txt

:sev150_step
if not exist data\pilot_150deg\meta.json (
    python generate_dataset.py --n 800 --out data/pilot_150deg --limited-deg 150 --n-proj 75 >logs\severity_gen150.log 2>&1
)
python training\train_unet.py --data data/pilot_150deg --run-name severity_150deg --epochs 100 --batch-size 16 --amp >logs\severity_train150.log 2>&1
if errorlevel 1 (echo severity-150 train FAILED - see logs\severity_train150.log >> logs\queue_status.txt & goto n200_step)
python training\evaluate_unet.py --run training\runs\severity_150deg --data data/pilot_150deg >logs\severity_eval150.log 2>&1
echo severity-150deg done  %time% >> logs\queue_status.txt

:n200_step
REM ============================ data-size sweep =============================
echo [datasize] 200-pair and 400-pair training-set sweep - about 1.5 hours...
if not exist data\pilot_200\meta.json (
    python generate_dataset.py --n 200 --out data/pilot_200 >logs\datasize_gen200.log 2>&1
)
python training\train_unet.py --data data/pilot_200 --run-name datasize_n200 --epochs 100 --batch-size 16 --amp >logs\datasize_train200.log 2>&1
if errorlevel 1 (echo datasize-200 train FAILED - see logs\datasize_train200.log >> logs\queue_status.txt & goto n400_step)
python training\evaluate_unet.py --run training\runs\datasize_n200 --data data/pilot_200 >logs\datasize_eval200.log 2>&1
echo datasize-200 done  %time% >> logs\queue_status.txt

:n400_step
if not exist data\pilot_400\meta.json (
    python generate_dataset.py --n 400 --out data/pilot_400 >logs\datasize_gen400.log 2>&1
)
python training\train_unet.py --data data/pilot_400 --run-name datasize_n400 --epochs 100 --batch-size 16 --amp >logs\datasize_train400.log 2>&1
if errorlevel 1 (echo datasize-400 train FAILED - see logs\datasize_train400.log >> logs\queue_status.txt & goto summary)
python training\evaluate_unet.py --run training\runs\datasize_n400 --data data/pilot_400 >logs\datasize_eval400.log 2>&1
echo datasize-400 done  %time% >> logs\queue_status.txt

:summary
REM ============================ Collect all results =============================
python -c "import json,glob,os; rows=[(os.path.basename(os.path.dirname(p)), json.load(open(p))) for p in sorted(glob.glob('training/runs/*/eval_summary.json'))]; print(f\"{'run':<22}{'FBP dB':>8}{'UNet dB':>9}{'gain':>7}{'FBP SSIM':>10}{'UNet SSIM':>11}\"); [print(f\"{n:<22}{d['fbp_psnr']:>8.2f}{d['unet_psnr']:>9.2f}{d['delta_psnr']:>+7.2f}{d['fbp_ssim']:>10.3f}{d['unet_ssim']:>11.3f}\") for n,d in rows]" >logs\all_results.txt 2>&1

echo Queue finished %date% %time% >> logs\queue_status.txt
echo.
echo ==============================================================
echo  ALL DONE. Results:
echo    - per-run folders:   training\runs\ablation_... noise_unscaled_... severity_... datasize_...
echo    - classical outputs:     training\runs\20260622_154318\classical_summary.json + classical_shepp_logan.png
echo    - results table:     logs\all_results.txt
echo    - step log:          logs\queue_status.txt
echo ==============================================================
type logs\all_results.txt
pause
