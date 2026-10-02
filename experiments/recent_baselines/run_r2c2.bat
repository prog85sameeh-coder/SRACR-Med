@echo off
REM Edit the two directories below if needed.
set ATTACKER_DIR=..\R1C1_IndependentLeakage\attacker_models
set PRIMARY_DIR=..\SRACR_Med_R1C11_Model_Checkpoints
set DATA_ROOT=F:\نهى ترقية\SRACR-Med

C:\Python39\python.exe r2c2_recent_baselines.py --source all --cpu --dataset-root "%DATA_ROOT%" --attacker-dir "%ATTACKER_DIR%" --primary-model-dir "%PRIMARY_DIR%" --budgets 0.25 0.50 0.75 --bootstrap 2000 --permutations 10000
pause
