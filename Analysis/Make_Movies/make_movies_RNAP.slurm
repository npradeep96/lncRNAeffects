#!/bin/bash
#SBATCH -J MakeMovies
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu 10000 
cd "$SLURM_SUBMIT_DIR"
echo $PWD

run_program()
{
 source activate fipyenv
 python make_movies_RNAP.py --i $input_directory --f $hdf5_file --mf $mesh_file
 conda deactivate
}

run_program
