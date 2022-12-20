#!/bin/bash
#SBATCH -J PhaseField
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu 1000 
cd "$SLURM_SUBMIT_DIR"
echo $PWD

stage_env()
{
    cp Input/input_params_bursting.txt input_parameters_$SLURM_JOBID.txt
    cp Input/param_list_bursting.txt param_list_$SLURM_JOBID.txt
}

run_program()
{
    source activate fipyenv
    input_file=input_parameters_$SLURM_JOBID.txt
    # out_folder=TWO_RNA_SOURCES/VARYING_DISTANCE
    param_file=param_list_$SLURM_JOBID.txt
    # program_file=phase_field-twosources.py
    python $program_file --i $input_file --o $out_folder --p $param_file --pN $pN
    conda deactivate
}

cleanup_files()
{
rm input_parameters_$SLURM_JOBID.txt
rm param_list_$SLURM_JOBID.txt
echo "DONE"
}

stage_env
run_program
cleanup_files
