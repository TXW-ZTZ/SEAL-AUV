@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Train SEAL...
python train_csac.py --seed 1 --stages 1,2,3 --episodes 2200 --frames 2200000 --device auto --no-current-curriculum
if errorlevel 1 exit /b 1
echo [2/3] Evaluate...
python evaluate.py --trajectory all --seed 1 --episodes 10 --device auto --train-tag single_traj_curriculum
if errorlevel 1 exit /b 1
echo [3/3] Plot results...
python plot_results.py --trajectory all --seed 1
if errorlevel 1 exit /b 1
echo Done.
