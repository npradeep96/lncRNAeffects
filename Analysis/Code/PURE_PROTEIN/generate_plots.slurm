#!/bin/bash
#SBATCH -J MakePlots
#SBATCH -p sched_mit_arupc
#SBATCH --mem-per-cpu 1000

cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
source activate fipyenv
python Analysis-lncRNA-and-activity.py
conda deactivate
}

run_program
