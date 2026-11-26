@echo off
REM ============================================================================
REM  run_w2_C_angle_patterns.bat -- Block C: the six-variant angle study (~8 h)
REM
REM  Datasets (needs TIGRE, ~2-3 min each) then one training+eval per variant
REM  (~75 min each). Projection density held at 0.5 proj/deg throughout:
REM    gap30        : 180-deg scan, ONE missing 30-deg chunk  (150-deg coverage
REM                   -> compare against severity_150deg, 40.79 dB)
REM    gap60        : ONE missing 60-deg chunk (120-deg coverage
REM                   -> compare against the baseline, 36.45 dB)
REM    multigap3x20 : THREE missing 20-deg chunks (120-deg coverage)
REM    multigap6x10 : SIX missing 10-deg chunks   (120-deg coverage)
REM    offset45/90  : the contiguous 120-deg wedge rotated 45 / 90 deg
REM  Skip-if-done: safe to stop overnight and relaunch (an interrupted
REM  training restarts from scratch; finished runs are skipped).
REM  Progress: logs\week2_status.txt
REM ============================================================================
cd /d "%~dp0.."
if not exist logs mkdir logs
call conda activate tigre 2>nul
python -c "import torch, tigre; assert torch.cuda.is_available()" >logs\week2_env_check.log 2>&1
if errorlevel 1 (echo [ERROR] Environment check failed - see logs\week2_env_check.log & if not defined W2_MASTER pause & exit /b 1)

echo ============================================== >> logs\week2_status.txt
echo Block C (angle patterns) started %date% %time% >> logs\week2_status.txt

REM ---- datasets ----
call :gen pilot_gap30        "--angle-mode gap --gap-deg 30"
call :gen pilot_gap60        "--angle-mode gap --gap-deg 60"
call :gen pilot_multigap3x20 "--angle-mode multigap --n-gaps 3 --gap-deg 20"
call :gen pilot_multigap6x10 "--angle-mode multigap --n-gaps 6 --gap-deg 10"
call :gen pilot_offset45     "--offset-deg 45"
call :gen pilot_offset90     "--offset-deg 90"
echo angle-pattern datasets built %time% >> logs\week2_status.txt

REM ---- training: one model per pattern ----
call :train_eval data/pilot_gap60        gap60        42
call :train_eval data/pilot_multigap3x20 multigap3x20 42
call :train_eval data/pilot_offset90     offset90     42
call :train_eval data/pilot_gap30        gap30        42
call :train_eval data/pilot_multigap6x10 multigap6x10 42
call :train_eval data/pilot_offset45     offset45     42

echo Block C done %date% %time% >> logs\week2_status.txt
python -u collect_week2_results.py >logs\week2_results.txt 2>&1
echo [C] Done. See logs\week2_results.txt
if not defined W2_MASTER pause
exit /b 0

:gen
REM %1 = dataset folder name, %2 = quoted extra flags for generate_dataset.py
if exist data\%1\meta.json (
    echo [C] dataset %1 already exists - skipping.
    exit /b 0
)
echo [C] dataset: %1...
python -u generate_dataset.py --n 800 --out data/%1 %~2 >logs\W2_gen_%1.log 2>&1
if errorlevel 1 (echo %1 dataset FAILED >> logs\week2_status.txt)
exit /b 0

:train_eval
REM %1 = dataset dir, %2 = run name, %3 = seed (train AND eval must match)
if exist training\runs\%2\best_model.pt (
    echo [C] %2 already trained - skipping training.
) else (
    echo [C] training %2 - about 75 min...
    python -u training\train_unet.py --data %1 --run-name %2 --seed %3 --epochs 100 --batch-size 16 --amp >logs\W2_train_%2.log 2>&1
    if errorlevel 1 (echo %2 train FAILED >> logs\week2_status.txt & exit /b 0)
)
if exist training\runs\%2\eval_summary.json (
    echo [C] %2 already evaluated - skipping eval.
) else (
    python -u training\evaluate_unet.py --run training\runs\%2 --data %1 --seed %3 >logs\W2_eval_%2.log 2>&1
    if errorlevel 1 (echo %2 eval FAILED >> logs\week2_status.txt & exit /b 0)
)
echo %2 done %time% >> logs\week2_status.txt
exit /b 0
