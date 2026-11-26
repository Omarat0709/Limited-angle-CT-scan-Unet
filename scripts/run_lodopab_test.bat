@echo off
REM ============================================================================
REM  run_lodopab_test.bat  --  real-data evaluation (LoDoPaB-CT), end to end
REM
REM  1. Downloads ground_truth_test.zip (1.6 GB) from Zenodo, if not present
REM  2. Extracts the HDF5 files
REM  3. Builds the 300-slice limited-angle dataset (needs TIGRE, ~10 min)
REM  4. Evaluates the trained baseline model on it (three-way comparison)
REM
REM  HOW TO RUN: double-click, or from an Anaconda Prompt in the repository root (scripts\<name>.bat).
REM  Download time dominates (depends on your connection); everything after
REM  is roughly 10-15 minutes. Safe to leave unattended.
REM  Results land in training\runs\20260622_154318\ as lodopab_summary.json,
REM  lodopab_per_image.csv and lodopab_comparison.png.
REM ============================================================================

cd /d "%~dp0.."
if not exist logs mkdir logs

call conda activate tigre 2>nul
python -c "import torch, tigre, h5py, skimage" >logs\lodopab_test_env_check.log 2>&1
if errorlevel 1 (
    echo [ERROR] Environment check failed - see logs\lodopab_test_env_check.log
    echo Open an Anaconda Prompt, conda activate tigre, and re-run this file.
    pause
    exit /b 1
)

set RAW=data\lodopab_raw
set ZIP=%RAW%\ground_truth_test.zip
set URL=https://zenodo.org/records/3384092/files/ground_truth_test.zip?download=1

if not exist %RAW% mkdir %RAW%

REM ---- 1. Download (resumable: re-run this bat if the connection drops) ----
REM Extraction is considered complete only if the LAST file is present.
if exist %RAW%\ground_truth_test_027.hdf5 goto build

echo [lodopab-test] Downloading ground_truth_test.zip - 1.6 GB from Zenodo...
echo      (resumable: if it fails or stalls, just re-run this file)
curl -L -C - -o %ZIP% "%URL%"

REM ---- 1a. Verify the download BEFORE trusting it ----
for %%A in (%ZIP%) do set SIZE=%%~zA
if %SIZE% LSS 1500000000 (
    echo [ERROR] Download incomplete: %SIZE% bytes, expected about 1.6 GB.
    echo         Re-run this file to resume the download.
    pause
    exit /b 1
)
echo [lodopab-test] Verifying md5 checksum - takes a few seconds...
certutil -hashfile %ZIP% MD5 | findstr /i "ecc655767fbe3d40908ca823921f4c7f" >nul
if errorlevel 1 (
    echo [ERROR] md5 mismatch - the file is corrupt. Delete %ZIP% and re-run.
    pause
    exit /b 1
)
echo [lodopab-test] Checksum OK.

REM ---- 2. Extract and verify count (expect 28 files: 000-027) ----
echo [lodopab-test] Extracting HDF5 files...
tar -xf %ZIP% -C %RAW%
if not exist %RAW%\ground_truth_test_027.hdf5 (
    echo [ERROR] Extraction incomplete - extract %ZIP% into %RAW% manually.
    pause
    exit /b 1
)

:build
REM ---- 3. Build the limited-angle dataset from 300 real slices ----
if not exist data\lodopab_test\meta.json (
    echo [lodopab-test] Building 300-slice dataset - about 10 minutes...
    python -u build_lodopab_dataset.py --h5-dir %RAW% --part test --n 300 >logs\lodopab_test_build.log 2>&1
    if errorlevel 1 (
        echo [ERROR] Dataset build failed - see logs\lodopab_test_build.log
        pause
        exit /b 1
    )
)

REM ---- 4. Evaluate the trained baseline model ----
echo [lodopab-test] Evaluating trained model on real anatomy...
python -u training\evaluate_lodopab.py --run training\runs\20260622_154318 >logs\lodopab_test_eval.log 2>&1
if errorlevel 1 (
    echo [ERROR] Evaluation failed - see logs\lodopab_test_eval.log
