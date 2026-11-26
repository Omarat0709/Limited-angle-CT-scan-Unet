@echo off
REM ============================================================================
REM  run_w2_D_e5_1600.bat -- Block D: 1600-pair data-efficiency run (~2.5-3 h)
REM
REM  Extends the E5 sweep (200/400/800) with a 1600-pair point: dataset ~5 min
REM  (needs TIGRE), then one training+eval (double the data => ~2x per-epoch
REM  time, ~2.5 h total). Run name: e5_n1600.
REM  Skip-if-done. Progress: logs\week2_status.txt
REM ============================================================================
cd /d "%~dp0.."
if not exist logs mkdir logs
call conda activate tigre 2>nul
python -c "import torch, tigre; assert torch.cuda.is_available()" >logs\week2_env_check.log 2>&1
if errorlevel 1 (echo [ERROR] Environment check failed - see logs\week2_env_check.log & if not defined W2_MASTER pause & exit /b 1)

echo ============================================== >> logs\week2_status.txt
echo Block D (e5_n1600) started %date% %time% >> logs\week2_status.txt

if not exist data\pilot_1600\meta.json (
    echo [D] dataset: pilot_1600 - 1600 pairs...
    python -u generate_dataset.py --n 1600 --out data/pilot_1600 >logs\W2_gen_1600.log 2>&1
    if errorlevel 1 (echo pilot_1600 dataset FAILED >> logs\week2_status.txt & if not defined W2_MASTER pause & exit /b 1)
)

if exist training\runs\e5_n1600\best_model.pt (
    echo [D] e5_n1600 already trained - skipping training.
) else (
    echo [D] training e5_n1600 - about 2.5 h...
    python -u training\train_unet.py --data data/pilot_1600 --run-name e5_n1600 --epochs 100 --batch-size 16 --amp >logs\W2_train_e5_n1600.log 2>&1
    if errorlevel 1 (echo e5_n1600 train FAILED >> logs\week2_status.txt & if not defined W2_MASTER pause & exit /b 1)
)
if exist training\runs\e5_n1600\eval_summary.json (
    echo [D] e5_n1600 already evaluated - skipping eval.
) else (
    python -u training\evaluate_unet.py --run training\runs\e5_n1600 --data data/pilot_1600 >logs\W2_eval_e5_n1600.log 2>&1
    if errorlevel 1 (echo e5_n1600 eval FAILED >> logs\week2_status.txt)
)
echo e5_n1600 done %time% >> logs\week2_status.txt

echo Block D done %date% %time% >> logs\week2_status.txt
python -u collect_week2_results.py >logs\week2_results.txt 2>&1
echo [D] Done. See logs\week2_results.txt
if not defined W2_MASTER pause
exit /b 0
