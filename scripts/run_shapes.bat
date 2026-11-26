@echo off
REM ============================================================================
REM  run_shapes.bat -- out-of-distribution geometry test (+ seed43 redo)
REM
REM  THE QUESTION (supervisor's suggestion): the baseline network has only ever
REM  seen elliptical structures. Put squares, rectangles and triangles inside
REM  the phantom -- does it reconstruct them, or does its learned prior
REM  'ellipsify' them (round corners, bend edges, erase shapes)? Sharp corners
REM  need exactly the directional frequencies the missing wedge destroys, so
REM  this probes hallucination directly.
REM
REM  WHAT IT RUNS, IN ORDER:
REM    1. data\pilot_shapes: 800 pairs with mixed-shape internal structures
REM       (generate_dataset.py --shapes; same skull/intensity conventions as
REM       pilot_v1, so results are comparable like-for-like)     ~3 min
REM    2. (a) the ellipse-trained BASELINE evaluated on the shape phantoms
REM       -> training\runs\xeval\base_on_shapes\ (comparison.png is the money
REM       figure: look at what happens to the squares)           ~2 min
REM    3. (b) identical network retrained ON the mixed shapes    ~75 min
REM       -> training\runs\shapes800\ (in-domain eval)
REM    4. Cross-check: the shapes-trained model back on pilot_v1 -- does shape
REM       diversity cost any ellipse performance?                 ~2 min
REM    5. Results table -> logs\shapes_results.txt
REM    6. Chains run_w2_B_seeds.bat: skips the finished seed44 and redoes the
REM       crashed baseline_seed43                                 ~75 min
REM
REM  TOTAL: ~2.5-3 h unattended. Skip-if-done throughout.
REM  Progress: logs\shapes_status.txt
REM ============================================================================

cd /d "%~dp0.."
if not exist logs mkdir logs

call conda activate tigre 2>nul
python -c "import torch, tigre; assert torch.cuda.is_available()" >logs\shapes_env_check.log 2>&1
if errorlevel 1 (
    echo [ERROR] Environment check failed - see logs\shapes_env_check.log
    pause
    exit /b 1
)

echo ============================================== >> logs\shapes_status.txt
echo Shapes test started  %date% %time% >> logs\shapes_status.txt

REM ---- 1. Mixed-shape dataset ------------------------------------------------
if not exist data\pilot_shapes\meta.json (
    echo [shapes] dataset: pilot_shapes - mixed shape family...
    python -u generate_dataset.py --n 800 --out data/pilot_shapes --shapes >logs\shapes_gen.log 2>&1
    if errorlevel 1 (echo pilot_shapes dataset FAILED >> logs\shapes_status.txt & pause & exit /b 1)
)
echo dataset built %time% >> logs\shapes_status.txt

REM ---- 2. (a) ellipse-trained baseline vs unseen shapes ---------------------
call :xeval base_on_shapes training\runs\20260622_154318\best_model.pt data/pilot_shapes 42

REM ---- 3. (b) retrain on the mixed shapes -----------------------------------
if exist training\runs\shapes800\best_model.pt (
    echo [shapes] shapes800 already trained - skipping training.
) else (
    echo [shapes] training shapes800 - about 75 min...
    python -u training\train_unet.py --data data/pilot_shapes --run-name shapes800 --epochs 100 --batch-size 16 --amp >logs\shapes_train.log 2>&1
    if errorlevel 1 (echo shapes800 train FAILED >> logs\shapes_status.txt & pause & exit /b 1)
)
if not exist training\runs\shapes800\eval_summary.json (
    python -u training\evaluate_unet.py --run training\runs\shapes800 --data data/pilot_shapes >logs\shapes_eval.log 2>&1
    if errorlevel 1 (echo shapes800 eval FAILED >> logs\shapes_status.txt)
)
echo shapes800 done %time% >> logs\shapes_status.txt

REM ---- 4. Cross-check: shapes-trained model on the ellipse set ---------------
call :xeval shapes_on_pilot training\runs\shapes800\best_model.pt data/pilot_v1 42

REM ---- 5. Results table ------------------------------------------------------
python -u collect_shapes_results.py >logs\shapes_results.txt 2>&1
type logs\shapes_results.txt
echo Shapes test finished %date% %time% >> logs\shapes_status.txt

REM ---- 6. Chain the seed43 redo ---------------------------------------------
echo.
echo [shapes] Now chaining run_w2_B_seeds.bat to redo baseline_seed43...
set W2_MASTER=1
call scripts\run_w2_B_seeds.bat
set W2_MASTER=

echo.
echo ==============================================================
echo  SHAPES TEST + SEED REDO DONE.
echo    Results table:   logs\shapes_results.txt
echo    Money figure:    training\runs\xeval\base_on_shapes\comparison.png
echo    Shapes model:    training\runs\shapes800\
echo    Seed table:      logs\week2_results.txt
echo ==============================================================
pause
exit /b 0

REM ============================================================================
:xeval
REM %1 = name, %2 = source checkpoint, %3 = dataset dir, %4 = split seed
if exist training\runs\xeval\%1\eval_summary.json (
    echo [shapes] cross-eval %1 already done - skipping.
    exit /b 0
)
if not exist training\runs\xeval\%1 mkdir training\runs\xeval\%1
copy /Y "%~2" training\runs\xeval\%1\best_model.pt >nul
if errorlevel 1 (echo xeval %1 checkpoint copy FAILED >> logs\shapes_status.txt & exit /b 0)
echo [shapes] cross-eval %1...
python -u training\evaluate_unet.py --ckpt training\runs\xeval\%1\best_model.pt --data %3 --seed %4 >logs\shapes_xeval_%1.log 2>&1
if errorlevel 1 (echo xeval %1 FAILED >> logs\shapes_status.txt) else (echo xeval %1 done %time% >> logs\shapes_status.txt)
del training\runs\xeval\%1\best_model.pt >nul 2>&1
exit /b 0
