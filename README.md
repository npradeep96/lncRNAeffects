# lncRNAeffects
Stable code to perform phase field simulations with lncRNAs and transcriptional condensates. This code was written to support this publication in the Biophysical Journal: https://doi.org/10.1016/j.bpj.2023.05.032

1. The scripts/ directory contains the different scripts that you can run to perform different kinds of phase field simulations. The main script used to run simulations for the paper is ``lncRNAeffect.py``
2. The use the following command to run code: 
``` python lncRNAeffect.py --i <path_to_input_parameter_file> --o <storage_folder_for_simulation_data> --p <file_containing_parameter_name_and_values_to_sweep_across> --pN <parameter_number_in_parameter_file>```
3. The Inputs/ directory contains example input parameter files with descriptions that different scripts in scripts/ take in to run simulations.
4. The Analysis/ directory contains code to make movies and perform other kinds of data analysis on the data files generated from scripts and generate plots.
