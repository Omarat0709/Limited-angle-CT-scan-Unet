@echo off
REM ============================================================================
REM  run_w2_B_seeds.bat -- Block B: seed-robustness repeats (~2.5 h)
REM
REM  Retrains the baseline configuration on pilot_v1 with seeds 43 and 44
REM  (~75 min each, per the noise study's measured actuals). Turns the single-seed 36.45 dB
REM  claim into mean +/- std. Evaluation reuses the SAME seed as training so
REM  the validation split reproduces.
REM  Skip-if-done. Progress: logs\week2_status.txt
REM ============================================================================
cd /d "%~dp0.."
if not exist logs mkdir logs
call conda activate tigre 2>nul
python -c "import torch; assert torch.cuda.is_available()" >logs\week2_env_check.log 2>&1
if errorlevel 1 (echo [ERROR] Environment check failed - see logs\week2_env_check.log & if not defined W2_MASTER pause & exit /b 1)

echo ============================================== >> logs\week2_status.txt
echo Block B (seeds) started %date% %time% >> logs\week2_status.txt

call :train_eval data/pilot_v1 baseline_seed43 43
call :train_eval data/pilot_v1 baseline_seed44 44

echo Block B done %date% %time% >> logs\week2_status.txt
python -u collect_week2_results.py >logs\week2_results.txt 2>&1
echo [B] Done. See logs\week2_results.txt
if not defined W2_MASTER pause
exit /b 0

:train_eval
REM %1 = dataset dir, %2 = run name, %3 = seed (train AND eval must match)
if exist training\runs\%2\best_model.pt (
    echo [B] %2 already trained - skipping training.
) else (
    echo [B] training %2 - about 75 min...
    python -u training\train_unet.py --data %1 --run-name %2 --seed %3 --epochs 100 --batch-size 16 --amp >logs\W2_train_%2.log 2>&1
    if errorlevel 1 (echo %2 train FAILED >> logs\week2_status.txt & exit /b 0)
)
if exist training\runs\%2\eval_summary.json (
    echo [B] %2 already evaluated - skipping eval.
) else (
    python -u training\evaluate_unet.py --run training\runs\%2 --data %1 --seed %3 >logs\W2_eval_%2.log 2>&1
    if errorlevel 1 (echo %2 eval FAILED >> logs\week2_status.txt & exit /b 0)
)
echo %2 done %time% >> logs\week2_status.txt
exit /b 0
