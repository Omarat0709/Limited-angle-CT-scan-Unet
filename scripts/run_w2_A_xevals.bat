@echo off
REM ============================================================================
REM  run_w2_A_xevals.bat -- Block A: cross-condition evaluations (~10 min)
REM
REM  Scores EXISTING checkpoints on datasets they were not trained for:
REM    base_on_90deg / base_on_150deg : 120-deg baseline model on 90/150-deg data
REM    clean_on_i0_1e6/1e5/1e4        : e7_clean model on the noisy phys_i0 sets
REM  No training. Checkpoints are COPIED into training\runs\xeval\<name>\ so the
REM  source runs' eval_summary.json is never overwritten; copies are deleted.
REM  Skip-if-done. Progress: logs\week2_status.txt
REM ============================================================================
cd /d "%~dp0.."
if not exist logs mkdir logs
call conda activate tigre 2>nul
python -c "import torch; assert torch.cuda.is_available()" >logs\week2_env_check.log 2>&1
if errorlevel 1 (echo [ERROR] Environment check failed - see logs\week2_env_check.log & if not defined W2_MASTER pause & exit /b 1)

echo ============================================== >> logs\week2_status.txt
echo Block A (xevals) started %date% %time% >> logs\week2_status.txt

call :xeval base_on_90deg   training\runs\20260622_154318\best_model.pt data/pilot_90deg  42
call :xeval base_on_150deg  training\runs\20260622_154318\best_model.pt data/pilot_150deg 42
call :xeval clean_on_i0_1e6 training\runs\e7_clean\best_model.pt        data/phys_i0_1e6  42
call :xeval clean_on_i0_1e5 training\runs\e7_clean\best_model.pt        data/phys_i0_1e5  42
call :xeval clean_on_i0_1e4 training\runs\e7_clean\best_model.pt        data/phys_i0_1e4  42

echo Block A done %date% %time% >> logs\week2_status.txt
python -u collect_week2_results.py >logs\week2_results.txt 2>&1
echo [A] Done. See logs\week2_results.txt
if not defined W2_MASTER pause
exit /b 0

:xeval
REM %1 = name, %2 = source checkpoint, %3 = dataset dir, %4 = split seed
if exist training\runs\xeval\%1\eval_summary.json (
    echo [A] cross-eval %1 already done - skipping.
    exit /b 0
)
if not exist training\runs\xeval\%1 mkdir training\runs\xeval\%1
copy /Y "%~2" training\runs\xeval\%1\best_model.pt >nul
if errorlevel 1 (echo xeval %1 checkpoint copy FAILED >> logs\week2_status.txt & exit /b 0)
echo [A] cross-eval %1...
python -u training\evaluate_unet.py --ckpt training\runs\xeval\%1\best_model.pt --data %3 --seed %4 >logs\W2_xeval_%1.log 2>&1
if errorlevel 1 (echo xeval %1 FAILED >> logs\week2_status.txt) else (echo xeval %1 done %time% >> logs\week2_status.txt)
del training\runs\xeval\%1\best_model.pt >nul 2>&1
exit /b 0
