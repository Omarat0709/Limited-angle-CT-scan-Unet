@echo off
REM ============================================================================
REM  run_week2.bat -- MASTER: runs all four Week-2 blocks back to back (~13 h)
REM
REM  Each block is its own script and can be run on its own instead:
REM    run_w2_A_xevals.bat      cross-condition evaluations        ~10 min
REM    run_w2_B_seeds.bat       baseline seed repeats (43, 44)     ~2.5 h
REM    run_w2_C_e8_angles.bat   E8: six-variant angle study        ~8 h
REM    run_w2_D_e5_1600.bat     1600-pair data-efficiency run      ~2.5-3 h
REM
REM  Everything is skip-if-done, so this master can also be stopped and
REM  relaunched: finished work is skipped, only a training interrupted
REM  mid-run restarts from scratch.
REM  Progress: logs\week2_status.txt   Results: logs\week2_results.txt
REM ============================================================================
cd /d "%~dp0.."
set W2_MASTER=1

call scripts\run_w2_A_xevals.bat
call scripts\run_w2_B_seeds.bat
call scripts\run_w2_C_e8_angles.bat
call scripts\run_w2_D_e5_1600.bat

set W2_MASTER=
echo.
echo ==============================================================
echo  WEEK-2 QUEUE DONE.
echo    Results table:      logs\week2_results.txt
echo    Cross-evals:        training\runs\xeval\^<name^>\eval_summary.json
echo    Seed repeats:       training\runs\baseline_seed43 ^& _seed44
echo    E8 angle study:     training\runs\gap30, gap60, multigap3x20,
echo                        multigap6x10, offset45, offset90
echo    Data efficiency:    training\runs\e5_n1600
echo ==============================================================
type logs\week2_results.txt
pause
exit /b 0
