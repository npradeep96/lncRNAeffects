#!/bin/bash
#SBATCH -J MakeMovies
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu 2000 
cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
 source activate fipyenv
 python generate_dynamical_trajectories_without_activity.py
 conda deactivate
}

run_program