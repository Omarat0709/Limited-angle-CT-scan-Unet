@echo off
REM  Deep trade-off sweep: locate the semi-convergence peak of U-Net-initialised
REM  OS-SART on pilot_v1 (in-distribution) and LoDoPaB (real anatomy).
REM  Checkpoints 0..150 iterations, 20 images each. ~15-25 min.
cd /d "%~dp0.."
call conda activate tigre 2>nul
echo ============================================== >> logs\e10_results.txt
echo E10 deep sweep  %date% %time% >> logs\e10_results.txt
python -u reconstruct_hybrid.py --data data/pilot_v1 --split val --sweep --sweep-deep >>logs\e10_results.txt 2>&1
python -u reconstruct_hybrid.py --data data/e6_lodopab_test --split all --sweep --sweep-deep >>logs\e10_results.txt 2>&1
echo Deep sweep finished %date% %time% >> logs\e10_results.txt
type logs\e10_results.txt | findstr /C:"===" /C:"dB" /C:"finished"
pause
