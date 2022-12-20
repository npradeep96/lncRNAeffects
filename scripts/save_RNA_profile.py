#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function that is called to generate RNA concentration profiles at equilibirum/steady state
"""
import numpy as np
import numpy.linalg as la
import os
import re
from utils.lncrna_analysis_helper import *
import h5py
import pickle as pkl
import fipy as fp

### Saving the lncRNA profiles as separate files

def save_lncRNA_profile(phi_r_list, l_list, target_directory, name_pattern, 
                        param_file='input_params.txt', 
                        spatial_variables_file='spatial_variables.hdf5', nx=300, dx=0.1):
    
    """
    Function to read through directories that contain lncRNA concentration data and store the concentration profiles 
    as pickle files in their corresponding directory
    """    

    for i in range(len(phi_r_list)):
        
        for j in range(len(l_list)):
    
            pattern = name_pattern.format(phi_r=phi_r_list[i],l=l_list[j])

            for root, dirs, files in os.walk(target_directory):

                regex = re.compile(pattern)
                match = re.search(regex, root)

                if match != None:

                    # Read values of input parameters from param_file
                    input_param_file = root + '/' + param_file
                    input_parameters = lh.input_parse(input_param_file)
                    if input_parameters['nx'] != nx or input_parameters['dx'] != dx:
                        nx = input_parameters['nx']
                        dx = input_parameters['dx']

                    mesh = lh.create_circular_mesh(radius=float(nx)*dx/2,cellSize=dx*1.5)
                    spatial_variable_hdf5_file_name = root + '/' + spatial_variables_file
                    with h5py.File(spatial_variable_hdf5_file_name) as sp_f:
                        counter = 1
                        while np.sum(sp_f['phi_r'][-counter]) == 0.0:
                            counter = counter + 1
                        phi_r = np.array(sp_f['phi_r'][-counter])
                        with open(root + '/phi_r_ss.pkl','wb') as f:
                            pkl.dump(phi_r, f)

if __name__ == "__main":

	# change these parameters for each run
	phi_r_list = [0.001, 0.002, 0.005, 0.01]
	l_list = [2.0, 6.0, 8.0, 10.0, 15.0]
	target_directory = '/nfs/arupclab001/npradeep96/PhaseField/RNA_EQBM_PROFILE/K_0.2/'
	name_pattern = "phi_p0_.*_L_P_{l}_.*_phi_r_0{phi_r}$"
	
	save_lncRNA_profile(phi_r_list=phi_r_list, l_list=l_list, target_directory=target_directory, 
	                    name_pattern=name_pattern)