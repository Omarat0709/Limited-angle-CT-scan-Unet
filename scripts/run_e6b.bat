@echo off
REM ============================================================================
REM  run_e6b.bat  --  E6b: domain-matched training on real LoDoPaB-CT slices
REM
REM  WHAT IT RUNS, IN ORDER:
REM    1. Downloads ground_truth_validation.zip (1.6 GB, Zenodo) if needed
REM       -- the VALIDATION split: its patients are disjoint from the test
REM       split used in E6, so evaluation stays honest.
REM    2. Builds an 800-pair real-anatomy training dataset (needs TIGRE)
REM    3. Run A "e6b_finetune": fine-tunes the ellipse-trained baseline
REM       (starts from its weights, lower learning rate 1e-4)
REM    4. Run B "e6b_scratch":  trains an identical network from random
REM       initialisation on the same real data (lr 1e-3)
REM    5. Evaluates BOTH on the same 300 E6 test slices as the baseline
REM       (target: beat OS-SART's 27.23 dB)
REM    6. Forgetting check: evaluates the fine-tuned model back on the
REM       synthetic pilot validation set
REM
REM  HOW TO RUN: double-click, or from an Anaconda Prompt in the repository root (scripts\<name>.bat).
REM  Total: download + ~25 min dataset build + 2 x ~45 min training + evals.
REM  Progress: logs\e6b_status.txt (one line per finished step).
REM ============================================================================

cd /d "%~dp0.."
if not exist logs mkdir logs

call conda activate tigre 2>nul
python -c "import torch, tigre, h5py, skimage; assert torch.cuda.is_available()" >logs\e6b_env_check.log 2>&1
if errorlevel 1 (
    echo [ERROR] Environment check failed - see logs\e6b_env_check.log
    echo Open an Anaconda Prompt, conda activate tigre, and re-run this file.
    pause
    exit /b 1
)

if not exist data\e6_lodopab_test\meta.json (
    echo [ERROR] The E6 test dataset is missing - run run_e6.bat first.
    pause
    exit /b 1
)

set RAW=data\lodopab_raw
set ZIP=%RAW%\ground_truth_validation.zip
set URL=https://zenodo.org/records/3384092/files/ground_truth_validation.zip?download=1

if not exist %RAW% mkdir %RAW%
echo ============================================== >> logs\e6b_status.txt
echo E6b started  %date% %time% >> logs\e6b_status.txt

REM ---- 1. Download the validation split (resumable: re-run if it drops) ----
if exist %RAW%\ground_truth_validation_027.hdf5 goto build

echo [E6b] Downloading ground_truth_validation.zip - 1.6 GB from Zenodo...
echo       (resumable: if it fails or stalls, just re-run this file)
curl -L -C - -o %ZIP% "%URL%"

for %%A in (%ZIP%) do set SIZE=%%~zA
if %SIZE% LSS 1500000000 (
    echo [ERROR] Download incomplete: %SIZE% bytes, expected about 1.6 GB.
    echo         Re-run this file to resume the download.
    pause
    exit /b 1
)
echo [E6b] Verifying md5 checksum - takes a few seconds...
certutil -hashfile %ZIP% MD5 | findstr /i "666c36f403734842f14ca8811a63b8f7" >nul
if errorlevel 1 (
    echo [ERROR] md5 mismatch - the file is corrupt. Delete %ZIP% and re-run.
    pause
    exit /b 1
)
echo [E6b] Checksum OK. Extracting...
tar -xf %ZIP% -C %RAW%
if not exist %RAW%\ground_truth_validation_027.hdf5 (
    echo [ERROR] Extraction incomplete - extract %ZIP% into %RAW% manually.
    pause
    exit /b 1
)
echo download+extract done  %time% >> logs\e6b_status.txt

:build
REM ---- 2. Build the 800-pair real training dataset ----
if not exist data\e6b_lodopab_train\meta.json (
    echo [E6b] Building 800-slice real training dataset - about 25 minutes...
    python -u build_e6_dataset.py --h5-dir %RAW% --part validation --n 800 --out data/e6b_lodopab_train >logs\E6b_build.log 2>&1
    if errorlevel 1 (
        echo [ERROR] Dataset build failed - see logs\E6b_build.log
        pause
        exit /b 1
    )
)
echo dataset built  %time% >> logs\e6b_status.txt

REM ---- 3. Run A: fine-tune the ellipse-trained baseline on real data ----
echo [E6b] Run A: fine-tuning the baseline - about 45 min...
python -u training\train_unet.py --data data/e6b_lodopab_train --run-name e6b_finetune --init-from training/runs/20260622_154318/best_model.pt --lr 1e-4 --epochs 100 --batch-size 16 --amp >logs\E6b_finetune.log 2>&1
if errorlevel 1 (echo finetune FAILED - see logs\E6b_finetune.log >> logs\e6b_status.txt & goto scratch)
echo finetune done  %time% >> logs\e6b_status.txt

:scratch
REM ---- 4. Run B: identical network from scratch on the same real data ----
echo [E6b] Run B: training from scratch - about 45 min...
python -u training\train_unet.py --data data/e6b_lodopab_train --run-name e6b_scratch --epochs 100 --batch-size 16 --amp >logs\E6b_scratch.log 2>&1
if errorlevel 1 (echo scratch FAILED - see logs\E6b_scratch.log >> logs\e6b_status.txt & goto eval)
echo scratch done  %time% >> logs\e6b_status.txt

:eval
REM ---- 5. Evaluate both on the SAME 300 test slices as E6 ----
echo [E6b] Evaluating both models on the 300 E6 test slices...
python -u training\evaluate_e6.py --run training\runs\e6b_finetune >logs\E6b_eval_finetune.log 2>&1
python -u training\evaluate_e6.py --run training\runs\e6b_scratch >logs\E6b_eval_scratch.log 2>&1

REM ---- 6. Forgetting check: fine-tuned model back on synthetic pilot ----
python -u training\evaluate_unet.py --run training\runs\e6b_finetune --data data\pilot_v1 >logs\E6b_forgetting.log 2>&1
echo evals done  %time% >> logs\e6b_status.txt
echo E6b finished %date% %time% >> logs\e6b_status.txt

echo.
echo ==============================================================
echo  E6b DONE. Compare (all on the same 300 real test slices):
echo    baseline (ellipse-trained):  training\runs\20260622_154318\e6_summary.json  (15.03 dB)
echo    fine-tuned:                  training\runs\e6b_finetune\e6_summary.json
echo    from scratch:                training\runs\e6b_scratch\e6_summary.json
echo    classical bar to beat: OS-SART 27.23 dB
echo    forgetting check:            logs\E6b_forgetting.log  (synthetic val)
echo ==============================================================
findstr /C:"===" /C:"UNET" /C:"U-Net" logs\E6b_eval_finetune.log
findstr /C:"===" /C:"UNET" /C:"U-Net" logs\E6b_eval_scratch.log
pause
