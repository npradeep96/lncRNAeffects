#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import fipy as fp
from fipy import Gmsh2D
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import re
import pdb
import h5py
import pdb


# In[2]:


def create_circular_mesh(radius,cellSize):
    """
    Function creates circular 2D mesh

    **Input**

    -   radius   =   Radius of mesh
    -   cellSize    =   Size of unit cell

    *Note* : No support for 3D meshes currently and **requires GMSH**
    """

    mesh = Gmsh2D('''
                     cellSize = %g;
                     radius = %g;
                     Point(1) = {0, 0, 0, cellSize};
                     Point(2) = {-radius, 0, 0, cellSize};
                     Point(3) = {0, radius, 0, cellSize};
                     Point(4) = {radius, 0, 0, cellSize};
                     Point(5) = {0, -radius, 0, cellSize};
                     Circle(6) = {2, 1, 3};


                    Circle(7) = {3, 1, 4};
                    Circle(8) = {4, 1, 5};
                    Circle(9) = {5, 1, 2};
                    Line Loop(10) = {6, 7, 8, 9};
                    Plane Surface(11) = {10};

       '''%(cellSize,radius)) # doctest: +GMSH


    return(mesh)


# In[3]:


nx = 600
dx = 0.2
mesh = create_circular_mesh(radius=float(nx)*dx/2,cellSize=dx*1.5)


# In[4]:


import ast

def input_parse(filename):
    """
    Parses input files (filename is path to input parameters or param_list file)

    params_flag toggles between input_params vs param_list
    """

    input_parameters  ={}
    with open(filename, 'r') as f:
        count = 0

        for line in f:
            line=line.strip()
            if line:
                if line.find('#') == -1:
                    var_name,var_value = line.split(',')[0],",".join(line.split(',')[1:]) # handle lines with more than 1 comma
                    if var_name != 'output_dir':
                        try:
                            input_parameters[var_name] = float(var_value)
                        except ValueError: # This occurs when python cannot convert list into a float.
                            # Evaluate the python expression as a list
                            input_parameters[var_name] = ast.literal_eval(var_value)
    return input_parameters


# In[7]:


def generate_dynamical_trajectories(target_directory, file_name_pattern, condensate_concentration_cutoff = 0.2):

    dynamical_trajectory_file_name = 'dynamical_trajectories.hdf5'
    dynamical_variables = ['steps', 't', 'average_protein_in_condensate', 'average_lncrna_in_condensate', 
                                   'average_mrna_in_condensate', 'molar_rate_of_mrna_production', 'net_rate_of_mrna_increase', 
                                   'condensate_area', 'total_amount_of_mrna']
    
    for root, dirs, files in os.walk(target_directory):

        regex = re.compile(file_name_pattern)
        match = re.search(regex, root)

        if match != None:
            
            if dynamical_trajectory_file_name in files:
                continue
            
            stats_file_name = root + '/stats.txt'
            df_stats = pd.read_csv(stats_file_name, '\t')

            times = df_stats['t']
            steps = df_stats['step']

            param_file = root + '/input_params.txt'
            input_parameters = input_parse(param_file) 

            with h5py.File(os.path.join(root + '/' + dynamical_trajectory_file_name), 'w') as dt_f:
                for dv in dynamical_variables:
                    dt_f.create_dataset(dv, shape=(len(steps),1))

            spatial_variable_hdf5_file_name = root + '/spatial_variables.hdf5'

            with h5py.File(spatial_variable_hdf5_file_name) as sp_f:      
                with h5py.File(os.path.join(root + '/' + dynamical_trajectory_file_name), 'a') as dt_f:

                    dt_f['steps'][:,0] = steps
                    dt_f['t'][:,0] = times

                    for s in range(len(steps)):

                        phi_p = fp.CellVariable(mesh=mesh, value = sp_f['phi_p'][s])
                        phi_m = fp.CellVariable(mesh=mesh, value = sp_f['phi_m'][s])
                        phi_r = fp.CellVariable(mesh=mesh, value = sp_f['phi_r'][s])

                        indices = phi_p.value > condensate_concentration_cutoff   
                        indices_enhancer_region = (mesh.x-0.5*input_parameters['rna_nucleus_distance_to_protein'])**2 + mesh.y**2 < 4*input_parameters['sigma']**2

                        if np.any(indices):
                            dt_f['average_protein_in_condensate'][s] = np.sum(phi_p.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                            dt_f['average_lncrna_in_condensate'][s] = np.sum(phi_r.value[indices_enhancer_region]*mesh.cellVolumes[indices_enhancer_region])/np.sum(mesh.cellVolumes[indices_enhancer_region])
                            dt_f['average_mrna_in_condensate'][s] = np.sum(phi_m.value[indices_enhancer_region]*mesh.cellVolumes[indices_enhancer_region])/np.sum(mesh.cellVolumes[indices_enhancer_region])
                            if dt_f['average_lncrna_in_condensate'][s] < 0.0:
                                dt_f['average_lncrna_in_condensate'][s] = 0.0
                            dt_f['condensate_area'][s] = np.sum(mesh.cellVolumes[indices])
                        else:
                            dt_f['average_protein_in_condensate'][s] = 0.0
                            dt_f['average_lncrna_in_condensate'][s] = 0.0
                            dt_f['average_mrna_in_condensate'][s] = 0.0
                            dt_f['condensate_area'][s] = 0.0

                        dt_f['total_amount_of_mrna'][s] = np.sum(phi_m.value*mesh.cellVolumes)

                        if input_parameters['reaction_rate'] == 0.0:
                            dt_f['molar_rate_of_mrna_production'][s] =  np.sum((input_parameters['k_p_max']*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value*mesh.cellVolumes)
                            dt_f['net_rate_of_mrna_increase'][s] =  np.sum((input_parameters['k_p_max']*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)
                        elif input_parameters['reaction_rate'] == 1.0:
                            dt_f['molar_rate_of_mrna_production'][s] =  np.sum((input_parameters['k_p_max']*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value*mesh.cellVolumes)
                            dt_f['net_rate_of_mrna_increase'][s] =  np.sum((input_parameters['k_p_max']*(phi_p > input_parameters['protein_threshold_mRNA_production'])-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)
                        elif input_parameters['reaction_rate'] == 2.0:
                            kpx = input_parameters['k_p_max']*np.exp(-((mesh.cellCenters[0]+0.5*input_parameters['rna_nucleus_distance_to_protein'])**2 + (mesh.cellCenters[1])**2)/input_parameters['spread_kp']**2)
                            dt_f['molar_rate_of_mrna_production'][s] =  np.sum(kpx.value*phi_p.value*mesh.cellVolumes)
                            dt_f['net_rate_of_mrna_increase'][s] =  np.sum((kpx*phi_p-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)


# In[ ]:


directory = "/nobackup1c/users/npradeep96/PhaseField/LNCRNA_AND_ACTIVITY/LNCRNA_RESCUES_CONDENSATE"
file_name_pattern = r'.*K_1.0$'

generate_dynamical_trajectories(directory, file_name_pattern, condensate_concentration_cutoff = 0.2)

