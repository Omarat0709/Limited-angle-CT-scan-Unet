@echo off
REM ============================================================================
REM  run_noise.bat  --  noise study, redone with a physically valid model
REM
REM  WHY: in the first noise run the phantom values (0-1) were used directly as attenuation in
REM  mm^-1. With 1 mm voxels the line integrals reached ~200, so I0*exp(-p)
REM  underflowed and every interior ray was photon-starved: the interior was
REM  destroyed before the network saw it. That is why the reconstructions
REM  fogged in the middle. See diagnose_noise.py for the evidence.
REM
REM  WHAT IT RUNS:
REM    0. diagnose_noise.py                    -> analysis\noise_diagnostic.png
REM    1. datasets at mu_water = 0.02 mm^-1 and I0 = 1e4, 1e5, 1e6 (+ noiseless
REM       control at the same scaling, to separate the scaling change from the
REM       noise itself)
REM    2. one training run per dataset, each evaluated
REM
REM  RUNTIME: ~20 min of dataset generation + 4 x ~45 min training. Unattended.
REM  Progress: logs\noise_status.txt
REM ============================================================================

cd /d "%~dp0.."
if not exist logs mkdir logs
if not exist analysis mkdir analysis

call conda activate tigre 2>nul
python -c "import torch, tigre; assert torch.cuda.is_available()" >logs\noise_env_check.log 2>&1
if errorlevel 1 (
    echo [ERROR] Environment check failed - see logs\noise_env_check.log
    pause
    exit /b 1
)

echo ============================================== >> logs\noise_status.txt
echo Noise study started  %date% %time% >> logs\noise_status.txt

REM ---- 0. Diagnostic: prove the cause of the first noise run's fogging -----------------------
echo [noise] Running noise diagnostic...
python -u diagnose_noise.py --mu-water 0.02 --i0 1e4 1e5 1e6 1e7 >logs\noise_diagnostic.log 2>&1
if errorlevel 1 (echo diagnostic FAILED >> logs\noise_status.txt) else (echo diagnostic done %time% >> logs\noise_status.txt)
type logs\noise_diagnostic.log

REM ---- 1. Datasets: physically scaled, three dose levels + control ------------
if not exist data\phys_noiseless\meta.json (
    echo [noise] dataset: physically scaled, noiseless control...
    python -u generate_dataset.py --n 800 --out data/phys_noiseless --mu-water 0.02 >logs\noise_gen_clean.log 2>&1
)
if not exist data\phys_i0_1e6\meta.json (
    echo [noise] dataset: I0 = 1e6 - mild noise...
    python -u generate_dataset.py --n 800 --out data/phys_i0_1e6 --mu-water 0.02 --noise-i0 1000000 >logs\noise_gen_1e6.log 2>&1
)
if not exist data\phys_i0_1e5\meta.json (
    echo [noise] dataset: I0 = 1e5 - moderate...
    python -u generate_dataset.py --n 800 --out data/phys_i0_1e5 --mu-water 0.02 --noise-i0 100000 >logs\noise_gen_1e5.log 2>&1
)
if not exist data\phys_i0_1e4\meta.json (
    echo [noise] dataset: I0 = 1e4 - low dose...
    python -u generate_dataset.py --n 800 --out data/phys_i0_1e4 --mu-water 0.02 --noise-i0 10000 >logs\noise_gen_1e4.log 2>&1
)
echo datasets built %time% >> logs\noise_status.txt

REM ---- 2. Train + evaluate one model per dose level ---------------------------
call :run phys_noiseless noise_noiseless
call :run phys_i0_1e6    noise_i0_1e6
call :run phys_i0_1e5    noise_i0_1e5
call :run phys_i0_1e4    noise_i0_1e4

echo Noise study finished %date% %time% >> logs\noise_status.txt
echo.
echo ==============================================================
echo  NOISE STUDY DONE. Compare val PSNR/SSIM across dose levels:
echo    training\runs\noise_noiseless\eval_summary.json      (control)
echo    training\runs\noise_i0_1e6\eval_summary.json     (mild)
echo    training\runs\noise_i0_1e5\eval_summary.json     (moderate)
echo    training\runs\noise_i0_1e4\eval_summary.json     (low dose)
echo  Diagnostic figure: analysis\noise_diagnostic.png
echo ==============================================================
pause
exit /b 0

:run
REM %1 = dataset folder name, %2 = run name
echo [noise] training on %1 - about 45 min...
python -u training\train_unet.py --data data/%1 --run-name %2 --epochs 100 --batch-size 16 --amp >logs\train_%2.log 2>&1
if errorlevel 1 (echo %2 train FAILED >> logs\noise_status.txt & exit /b 0)
python -u training\evaluate_unet.py --run training\runs\%2 --data data/%1 >logs\eval_%2.log 2>&1
echo %2 done %time% >> logs\noise_status.txt
exit /b 0
