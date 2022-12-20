#!/bin/bash
#SBATCH -J PhaseField
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc_long
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu 1000 
cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
    conda activate fipyenv
    python pure-protein.py
    conda deactivate
}

run_program

