@echo off
rem Detached launcher for the long revision-1 jobs (run via Task Scheduler so the
rem jobs do not belong to an interactive session's process tree). Everything is
rem resumable: training resumes from ckpt_latest.pt, sweeps/evals from their CSVs.
cd /d "%~dp0\..\.."
set GPU=D:\aiharp-gpu\Scripts\python.exe
set CPU=%USERPROFILE%\miniconda3\python.exe
if not exist logs\revision1 mkdir logs\revision1
start "laneA" /b cmd /c "%GPU% -u -m experiments.revision1.train_queue --only ref_s1 gcn_s1 --workers 5 >> logs\revision1\train_laneA.log 2>&1"
start "laneB" /b cmd /c "%GPU% -u -m experiments.revision1.train_queue --only ref_s2 mlp_s1 --workers 5 >> logs\revision1\train_laneB.log 2>&1"
start "laneC" /b cmd /c "%GPU% -u -m experiments.revision1.train_queue --only star_s1 --wait-for ref_s1 ref_s2 --workers 4 >> logs\revision1\train_laneC.log 2>&1"
start "c4eval" /b cmd /c "%GPU% -u -m experiments.revision1.agent_eval --checkpoint checkpoints\campaign\ref\ckpt_final.pt --name ref_s0_gen --cells generalization --taus 0.5 --modes sampled --jobs 3 >> logs\revision1\eval_c4.log 2>&1"
start "sweeps" /b cmd /c "%CPU% -u -m experiments.revision1.sweeps gen --jobs 3 >> logs\revision1\sweeps4.log 2>&1 && %CPU% -u -m experiments.revision1.sweeps grid24 --jobs 3 >> logs\revision1\sweeps4.log 2>&1"
