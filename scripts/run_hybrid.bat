@echo off
REM ============================================================================
REM  run_hybrid.bat -- hybrid reconstruction (v2) + trade-off sweep
REM
REM  ONE comprehensive run, informed by the debugging so far:
REM    - completion grid at 0.25 deg (720 views; the 1-deg default capped every
REM      result at the ~27 dB FBP-inversion ceiling)
REM    - composite inverted with OS-SART (FBP streaks when prior rows and
REM      measured rows disagree, i.e. exactly the out-of-distribution case)
REM    - PLUS the trade-off sweep: warm-started OS-SART on measured data only,
REM      scored at iteration checkpoints 0..30 -- maps the full prior<->data
REM      dial on pilot_v1 (prior is right) and LoDoPaB (prior is wrong)
REM
REM  RUNTIME: roughly 1-1.5 h total, unattended.
REM  Results: training\runs\hybrid\<dataset>\hybrid_summary.json /
REM           tradeoff_sweep.json; console log in logs\hybrid_results.txt
REM ============================================================================

cd /d "%~dp0.."
if not exist logs mkdir logs

call conda activate tigre 2>nul
python -c "import torch, tigre; assert torch.cuda.is_available()" >logs\hybrid_env_check.log 2>&1
if errorlevel 1 (
    echo [ERROR] Environment check failed - see logs\hybrid_env_check.log
    if not defined HYBRID_NOPAUSE pause
    exit /b 1
)

echo ============================================== > logs\hybrid_results.txt
echo Hybrid v2 run  %date% %time% >> logs\hybrid_results.txt

echo [hybrid] hybrid (OS-SART completion, 0.25-deg grid): pilot_v1...
python -u reconstruct_hybrid.py --data data/pilot_v1 --split val >>logs\hybrid_results.txt 2>&1
echo [hybrid] hybrid: pilot_shapes...
python -u reconstruct_hybrid.py --data data/pilot_shapes --split val >>logs\hybrid_results.txt 2>&1
echo [hybrid] hybrid: lodopab_test (300 slices, the long one)...
python -u reconstruct_hybrid.py --data data/lodopab_test --split all >>logs\hybrid_results.txt 2>&1

echo [hybrid] trade-off sweep: pilot_v1 (20 images)...
python -u reconstruct_hybrid.py --data data/pilot_v1 --split val --sweep >>logs\hybrid_results.txt 2>&1
echo [hybrid] trade-off sweep: lodopab_test (20 images)...
python -u reconstruct_hybrid.py --data data/lodopab_test --split all --sweep >>logs\hybrid_results.txt 2>&1

echo Hybrid v2 finished %date% %time% >> logs\hybrid_results.txt
type logs\hybrid_results.txt
echo.
echo ==============================================================
echo  HYBRID v2 DONE.
echo    Hybrid summaries: training\runs\hybrid\^<dataset^>\hybrid_summary.json
echo    Trade-off curves: training\runs\hybrid\^<dataset^>\tradeoff_sweep.json
echo ==============================================================
if not defined HYBRID_NOPAUSE pause
exit /b 0
