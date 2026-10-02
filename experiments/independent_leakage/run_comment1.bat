@echo off
set PY=C:\Python39\python.exe
cd /d "%~dp0"

%PY% independent_leakage.py train-attacker --source chest --cpu
if errorlevel 1 goto :err
%PY% independent_leakage.py train-attacker --source retinal --cpu
if errorlevel 1 goto :err
%PY% independent_leakage.py train-attacker --source split --cpu
if errorlevel 1 goto :err

%PY% independent_leakage.py evaluate --source all --cpu --budgets 0.25 0.50 0.75 --bootstrap 2000 --permutations 10000 --include-full-mask
if errorlevel 1 goto :err

echo.
echo COMMENT-1 EXPERIMENT COMPLETED.
echo Send the summary and paired-test CSV files back for manuscript revision.
pause
exit /b 0

:err
echo.
echo ERROR: experiment stopped. Copy the traceback and send it for correction.
pause
exit /b 1
