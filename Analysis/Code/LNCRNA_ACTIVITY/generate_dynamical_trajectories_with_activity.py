import numpy as np
import pandas as pd
import fipy as fp
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import re
import h5py
import lncrna_analysis_helper as lh

class LncrnaDataAnalysis:
    
    dynamical_variables = ['steps', 't', 
                               'average_protein_in_enhancer_well', 'average_lncrna_in_enhancer_well', 
                               'average_mrna_in_enhancer_well',
                               'rate_average_protein_in_enhancer_well', 'rate_average_lncrna_in_enhancer_well', 
                               'rate_average_mrna_in_enhancer_well',
                               'flux_of_protein_into_enhancer_well', 'flux_of_lncrna_into_enhancer_well', 
                               'flux_of_mrna_into_enhancer_well',
                               'flux_of_protein_into_enhancer_well_absum',
                               'flux_of_protein_into_enhancer_well_DW', 'flux_of_protein_into_enhancer_well_well_att', 
                               'flux_of_protein_into_enhancer_well_elec_att', 'flux_of_protein_into_enhancer_well_reent_rep',
                               'flux_of_lncrna_into_enhancer_well_absum',
                               'flux_of_lncrna_into_enhancer_well_elec_att', 'flux_of_lncrna_into_enhancer_well_well_att', 
                               'flux_of_lncrna_into_enhancer_well_reent_rep', 'flux_of_lncrna_into_enhancer_well_rna_rep',
                               'flux_of_protein_into_enhancer_well_DW_frac', 'flux_of_protein_into_enhancer_well_well_att_frac', 
                               'flux_of_protein_into_enhancer_well_elec_att_frac', 'flux_of_protein_into_enhancer_well_reent_rep_frac',
                               'flux_of_lncrna_into_enhancer_well_elec_att_frac', 'flux_of_lncrna_into_enhancer_well_well_att_frac', 
                               'flux_of_lncrna_into_enhancer_well_reent_rep_frac', 'flux_of_lncrna_into_enhancer_well_rna_rep_frac',
                               'average_protein_in_lncrna_well', 'average_lncrna_in_lncrna_well', 
                               'average_mrna_in_lncrna_well',
                               'rate_average_protein_in_lncrna_well', 'rate_average_lncrna_in_lncrna_well', 
                               'rate_average_mrna_in_lncrna_well',
                               'flux_of_protein_into_lncrna_well', 'flux_of_lncrna_into_lncrna_well', 
                               'flux_of_mrna_into_lncrna_well',
                               'flux_of_protein_into_lncrna_well_absum',
                               'flux_of_protein_into_lncrna_well_DW', 'flux_of_protein_into_lncrna_well_well_att', 
                               'flux_of_protein_into_lncrna_well_elec_att', 'flux_of_protein_into_lncrna_well_reent_rep',
                               'flux_of_lncrna_into_lncrna_well_absum',
                               'flux_of_lncrna_into_lncrna_well_elec_att', 'flux_of_lncrna_into_lncrna_well_well_att', 
                               'flux_of_lncrna_into_lncrna_well_reent_rep', 'flux_of_lncrna_into_lncrna_well_rna_rep',
                               'flux_of_protein_into_lncrna_well_DW_frac', 'flux_of_protein_into_lncrna_well_well_att_frac',
                               'flux_of_protein_into_lncrna_well_elec_att_frac', 'flux_of_protein_into_lncrna_well_reent_rep_frac',
                               'flux_of_lncrna_into_lncrna_well_elec_att_frac', 'flux_of_lncrna_into_lncrna_well_well_att_frac', 
                               'flux_of_lncrna_into_lncrna_well_reent_rep_frac', 'flux_of_lncrna_into_lncrna_well_rna_rep_frac',
                               'total_amount_of_mrna', 'molar_rate_of_mrna_production', 'net_rate_of_mrna_increase',
                               'molar_rate_of_mrna_production_in_enhancer_well', 
                               'condensate_area', 'rate_condensate_area']
        
    steady_state_variables = ['average_protein_in_condensate', 'average_lncrna_in_condensate', 
                                  'average_mrna_in_condensate',
                                  'average_protein_in_lncrna_well', 'average_lncrna_in_lncrna_well', 
                                  'average_mrna_in_lncrna_well',
                                  'total_amount_of_mrna', 'molar_rate_of_mrna_production', 
                                  'molar_rate_of_mrna_production_in_condensate',
                                  'condensate_area']
    
    dynamical_trajectory_file_name = 'dynamical_trajectories.hdf5'
    
    def __init__(self):
        
        return
    
    def calculate_dynamical_trajectories(self, target_directory, directory_name_pattern, condensate_concentration_cutoff=0.25,
                                         stats_file='stats.txt', param_file='input_params.txt', 
                                         spatial_variables_file='spatial_variables.hdf5', overwrite=False):
        
        nx = 100
        dx = 0.1
        
        for root, dirs, files in os.walk(target_directory):

            regex = re.compile(directory_name_pattern)
            match = re.search(regex, root)

            if match != None:
                
                if self.dynamical_trajectory_file_name not in files or overwrite:
                
                    # Read steps and corresponding simulation times from stats file
                    stats_file_name = root + '/' + stats_file
                    df_stats = pd.read_csv(stats_file_name, '\t')
                    times = df_stats['t']
                    steps = df_stats['step']

                    # Read values of input parameters from param_file
                    input_param_file = root + '/' + param_file
                    input_parameters = lh.input_parse(input_param_file)
                    
                    if input_parameters['nx'] != nx or input_parameters['dx'] != dx:
                        nx = input_parameters['nx']
                        dx = input_parameters['dx']
                        mesh = lh.create_circular_mesh(radius=float(nx)*dx/2,cellSize=dx*1.5)

                    # Create hdf5 file to store dynamical trajectories
                    with h5py.File(os.path.join(root + '/' + self.dynamical_trajectory_file_name), 'w') as dt_f:
                        for dv in self.dynamical_variables:
                            dt_f.create_dataset(dv, shape=(len(steps),1))

                    # Compute values of dynamical variables by looping through the spatial variables file
                    spatial_variable_hdf5_file_name = root + '/' + spatial_variables_file

                    with h5py.File(spatial_variable_hdf5_file_name) as sp_f:      
                        with h5py.File(os.path.join(root + '/' + self.dynamical_trajectory_file_name), 'a') as dt_f:

                            print('Generating ' + root + '/' + self.dynamical_trajectory_file_name + ' ...')
                            
                            dt_f['steps'][:,0] = steps
                            dt_f['t'][:,0] = times

                            indices_enhancer_well = (mesh.x+0.5*input_parameters['rna_nucleus_distance_to_protein'])**2 + mesh.y**2 - 1.0*input_parameters['sigma']**2 < 0
                            indices_lncrna_well = (mesh.x-0.5*input_parameters['rna_nucleus_distance_to_protein'])**2 + mesh.y**2 - 1.0*input_parameters['L_R']**2 < 0

                            for s in range(len(steps)):

                                phi_p = fp.CellVariable(mesh=mesh, value = sp_f['phi_p'][s])
                                if 'phi_m' in sp_f.keys():
                                    phi_m = fp.CellVariable(mesh=mesh, value = sp_f['phi_m'][s])
                                else:
                                    phi_m = fp.CellVariable(mesh=mesh, value = 0.0)  
                                    
                                phi_r = fp.CellVariable(mesh=mesh, value = sp_f['phi_r'][s])
                                
                                mu_p = fp.CellVariable(mesh=mesh, value = sp_f['mu_p'][s])
                                # mu_m = fp.CellVariable(mesh=mesh, value = sp_f['mu_m'][s])
                                mu_r = fp.CellVariable(mesh=mesh, value = sp_f['mu_r'][s])

                                dt_f['average_protein_in_lncrna_well'][s] = np.sum(phi_p.value[indices_lncrna_well]*mesh.cellVolumes[indices_lncrna_well])/np.sum(mesh.cellVolumes[indices_lncrna_well])
                                dt_f['average_mrna_in_lncrna_well'][s] = np.sum(phi_m.value[indices_lncrna_well]*mesh.cellVolumes[indices_lncrna_well])/np.sum(mesh.cellVolumes[indices_lncrna_well])
                                dt_f['average_lncrna_in_lncrna_well'][s] = np.sum(phi_r.value[indices_lncrna_well]*mesh.cellVolumes[indices_lncrna_well])/np.sum(mesh.cellVolumes[indices_lncrna_well])
                                                                
                                dt_f['average_protein_in_enhancer_well'][s] = np.sum(phi_p.value[indices_enhancer_well]*mesh.cellVolumes[indices_enhancer_well])/np.sum(mesh.cellVolumes[indices_enhancer_well])
                                dt_f['average_mrna_in_enhancer_well'][s] = np.sum(phi_m.value[indices_enhancer_well]*mesh.cellVolumes[indices_enhancer_well])/np.sum(mesh.cellVolumes[indices_enhancer_well])
                                dt_f['average_lncrna_in_enhancer_well'][s] = np.sum(phi_r.value[indices_enhancer_well]*mesh.cellVolumes[indices_enhancer_well])/np.sum(mesh.cellVolumes[indices_enhancer_well])
                                
                                dt_f['total_amount_of_mrna'][s] = np.sum(phi_m.value*mesh.cellVolumes)
                                
                                # pdb.set_trace()
                                
                                mu_p_T1 = 4*input_parameters['rho_s']*(phi_p-input_parameters['c_alpha'])*(phi_p-input_parameters['c_beta'])*(phi_p-0.5*(input_parameters['c_alpha']+input_parameters['c_beta']))
                                dt_f['flux_of_protein_into_enhancer_well_DW'][s] = np.sum(input_parameters['M_protein']*mu_p_T1.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                mu_p_T2 = phi_p - phi_p - 1.0*input_parameters['c_max']*np.exp(-((mesh.x+0.5*input_parameters['rna_nucleus_distance_to_protein'])**2 + mesh.y**2)/input_parameters['sigma']**2)
                                dt_f['flux_of_protein_into_enhancer_well_well_att'][s] = np.sum(input_parameters['M_protein']*mu_p_T2.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                mu_p_T3 = -1.0*input_parameters['chi_r']*phi_r -1.0*input_parameters['chi_m']*phi_m
                                dt_f['flux_of_protein_into_enhancer_well_elec_att'][s] = np.sum(input_parameters['M_protein']*mu_p_T3.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                mu_p_T4 = 2.0*input_parameters['c']*phi_p*(phi_r+phi_m)**2
                                dt_f['flux_of_protein_into_enhancer_well_reent_rep'][s] = np.sum(input_parameters['M_protein']*mu_p_T4.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                
                                mu_r_T1 = -1.0*input_parameters['chi_m']*phi_p
                                dt_f['flux_of_lncrna_into_enhancer_well_elec_att'][s] = np.sum(input_parameters['M_rna']*mu_r_T1.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                mu_r_T2 = 2.0*input_parameters['c']*(phi_r+phi_m)*(phi_p)**2
                                dt_f['flux_of_lncrna_into_enhancer_well_reent_rep'][s] = np.sum(input_parameters['M_rna']*mu_r_T2.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                mu_r_T3 = 2.0*(input_parameters['rho_r']*phi_r+input_parameters['rho_c']*phi_m)
                                dt_f['flux_of_lncrna_into_enhancer_well_rna_rep'][s] = np.sum(input_parameters['M_rna']*mu_r_T3.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                mu_r_T4 = phi_p - phi_p - 1.0*input_parameters['K']*np.exp(-((mesh.x-0.5*input_parameters['rna_nucleus_distance_to_protein'])**2 + mesh.y**2)/input_parameters['L_R']**2)
                                dt_f['flux_of_lncrna_into_enhancer_well_well_att'][s] = np.sum(input_parameters['M_rna']*mu_r_T4.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                
                                dt_f['flux_of_protein_into_enhancer_well'][s] = np.sum(input_parameters['M_protein']*mu_p.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                dt_f['flux_of_lncrna_into_enhancer_well'][s] = dt_f['flux_of_lncrna_into_enhancer_well_elec_att'][s] + dt_f['flux_of_lncrna_into_enhancer_well_reent_rep'][s] + dt_f['flux_of_lncrna_into_enhancer_well_rna_rep'][s] + dt_f['flux_of_lncrna_into_enhancer_well_well_att'][s]
                                dt_f['flux_of_mrna_into_enhancer_well'][s] = np.sum(input_parameters['M_mrna']*phi_m.faceGrad.divergence[indices_enhancer_well].value*mesh.cellVolumes[indices_enhancer_well])
                                
                                dt_f['flux_of_protein_into_enhancer_well_absum'][s] = np.abs(dt_f['flux_of_protein_into_enhancer_well_DW'][s]) + np.abs(dt_f['flux_of_protein_into_enhancer_well_well_att'][s]) + np.abs(dt_f['flux_of_protein_into_enhancer_well_elec_att'][s]) + np.abs(dt_f['flux_of_protein_into_enhancer_well_reent_rep'][s])
                                dt_f['flux_of_protein_into_enhancer_well_DW_frac'][s] = dt_f['flux_of_protein_into_enhancer_well_DW'][s]/dt_f['flux_of_protein_into_enhancer_well_absum'][s]
                                dt_f['flux_of_protein_into_enhancer_well_well_att_frac'][s] = dt_f['flux_of_protein_into_enhancer_well_well_att'][s]/dt_f['flux_of_protein_into_enhancer_well_absum'][s]
                                dt_f['flux_of_protein_into_enhancer_well_elec_att_frac'][s] = dt_f['flux_of_protein_into_enhancer_well_elec_att'][s]/dt_f['flux_of_protein_into_enhancer_well_absum'][s]
                                dt_f['flux_of_protein_into_enhancer_well_reent_rep_frac'][s] = dt_f['flux_of_protein_into_enhancer_well_reent_rep'][s]/dt_f['flux_of_protein_into_enhancer_well_absum'][s]
                                
                                dt_f['flux_of_lncrna_into_enhancer_well_absum'][s] = np.abs(dt_f['flux_of_lncrna_into_enhancer_well_elec_att'][s]) + np.abs(dt_f['flux_of_lncrna_into_enhancer_well_reent_rep'][s]) + np.abs(dt_f['flux_of_lncrna_into_enhancer_well_rna_rep'][s]) + np.abs(dt_f['flux_of_lncrna_into_enhancer_well_well_att'][s])
                                dt_f['flux_of_lncrna_into_enhancer_well_elec_att_frac'][s] = dt_f['flux_of_lncrna_into_enhancer_well_elec_att'][s]/dt_f['flux_of_lncrna_into_enhancer_well_absum'][s]
                                dt_f['flux_of_lncrna_into_enhancer_well_reent_rep_frac'][s] = dt_f['flux_of_lncrna_into_enhancer_well_reent_rep'][s]/dt_f['flux_of_lncrna_into_enhancer_well_absum'][s]
                                dt_f['flux_of_lncrna_into_enhancer_well_rna_rep_frac'][s] = dt_f['flux_of_lncrna_into_enhancer_well_rna_rep'][s]/dt_f['flux_of_lncrna_into_enhancer_well_absum'][s]
                                dt_f['flux_of_lncrna_into_enhancer_well_well_att_frac'][s] = dt_f['flux_of_lncrna_into_enhancer_well_well_att'][s]/dt_f['flux_of_lncrna_into_enhancer_well_absum'][s]
                                

                                dt_f['flux_of_protein_into_lncrna_well_DW'][s] = np.sum(input_parameters['M_protein']*mu_p_T1.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                dt_f['flux_of_protein_into_lncrna_well_well_att'][s] = np.sum(input_parameters['M_protein']*mu_p_T2.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                dt_f['flux_of_protein_into_lncrna_well_elec_att'][s] = np.sum(input_parameters['M_protein']*mu_p_T3.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                dt_f['flux_of_protein_into_lncrna_well_reent_rep'][s] = np.sum(input_parameters['M_protein']*mu_p_T4.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                
                                dt_f['flux_of_lncrna_into_lncrna_well_elec_att'][s] = np.sum(input_parameters['M_rna']*mu_r_T1.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                dt_f['flux_of_lncrna_into_lncrna_well_reent_rep'][s] = np.sum(input_parameters['M_rna']*mu_r_T2.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                dt_f['flux_of_lncrna_into_lncrna_well_rna_rep'][s] = np.sum(input_parameters['M_rna']*mu_r_T3.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                dt_f['flux_of_lncrna_into_lncrna_well_well_att'][s] = np.sum(input_parameters['M_rna']*mu_r_T4.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                
                                dt_f['flux_of_protein_into_lncrna_well'][s] = np.sum(input_parameters['M_protein']*mu_p.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                dt_f['flux_of_lncrna_into_lncrna_well'][s] = dt_f['flux_of_lncrna_into_lncrna_well_elec_att'][s] + dt_f['flux_of_lncrna_into_lncrna_well_reent_rep'][s] + dt_f['flux_of_lncrna_into_lncrna_well_rna_rep'][s] + dt_f['flux_of_lncrna_into_lncrna_well_well_att'][s]
                                dt_f['flux_of_mrna_into_lncrna_well'][s] = np.sum(input_parameters['M_mrna']*phi_m.faceGrad.divergence[indices_lncrna_well].value*mesh.cellVolumes[indices_lncrna_well])
                                
                                dt_f['flux_of_protein_into_lncrna_well_absum'][s] = np.abs(dt_f['flux_of_protein_into_lncrna_well_DW'][s]) + np.abs(dt_f['flux_of_protein_into_lncrna_well_well_att'][s]) + np.abs(dt_f['flux_of_protein_into_lncrna_well_elec_att'][s]) + np.abs(dt_f['flux_of_protein_into_lncrna_well_reent_rep'][s])
                                dt_f['flux_of_protein_into_lncrna_well_DW_frac'][s] = dt_f['flux_of_protein_into_lncrna_well_DW'][s]/dt_f['flux_of_protein_into_lncrna_well_absum'][s]
                                dt_f['flux_of_protein_into_lncrna_well_well_att_frac'][s] = dt_f['flux_of_protein_into_lncrna_well_well_att'][s]/dt_f['flux_of_protein_into_lncrna_well_absum'][s]
                                dt_f['flux_of_protein_into_lncrna_well_elec_att_frac'][s] = dt_f['flux_of_protein_into_lncrna_well_elec_att'][s]/dt_f['flux_of_protein_into_lncrna_well_absum'][s]
                                dt_f['flux_of_protein_into_lncrna_well_reent_rep_frac'][s] = dt_f['flux_of_protein_into_lncrna_well_reent_rep'][s]/dt_f['flux_of_protein_into_lncrna_well_absum'][s]
                                
                                dt_f['flux_of_lncrna_into_lncrna_well_absum'][s] = np.abs(dt_f['flux_of_lncrna_into_lncrna_well_elec_att'][s]) + np.abs(dt_f['flux_of_lncrna_into_lncrna_well_reent_rep'][s]) + np.abs(dt_f['flux_of_lncrna_into_lncrna_well_rna_rep'][s]) + np.abs(dt_f['flux_of_lncrna_into_lncrna_well_well_att'][s]) 
                                dt_f['flux_of_lncrna_into_lncrna_well_elec_att_frac'][s] = dt_f['flux_of_lncrna_into_lncrna_well_elec_att'][s]/dt_f['flux_of_lncrna_into_lncrna_well_absum'][s]
                                dt_f['flux_of_lncrna_into_lncrna_well_reent_rep_frac'][s] = dt_f['flux_of_lncrna_into_lncrna_well_reent_rep'][s]/dt_f['flux_of_lncrna_into_lncrna_well_absum'][s]
                                dt_f['flux_of_lncrna_into_lncrna_well_rna_rep_frac'][s] = dt_f['flux_of_lncrna_into_lncrna_well_rna_rep'][s]/dt_f['flux_of_lncrna_into_lncrna_well_absum'][s]
                                dt_f['flux_of_lncrna_into_lncrna_well_well_att_frac'][s] = dt_f['flux_of_lncrna_into_lncrna_well_well_att'][s]/dt_f['flux_of_lncrna_into_lncrna_well_absum'][s]
                                
                                
                                indices_protein_rich = phi_p.value > condensate_concentration_cutoff                    

                                if np.any(indices_protein_rich):                                    
                                    dt_f['condensate_area'][s] = np.sum(mesh.cellVolumes[indices_protein_rich])
                                else:
                                    dt_f['condensate_area'][s] = 0.0
                                    
                                dt_f['rate_condensate_area'][:,0] = np.gradient(dt_f['condensate_area'][:,0], dt_f['t'][:,0])

                                if input_parameters['reaction_rate'] == 0.0:
                                    dt_f['molar_rate_of_mrna_production'][s] =  np.sum((input_parameters['k_p_max']*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value*mesh.cellVolumes)
                                    dt_f['molar_rate_of_mrna_production_in_enhancer_well'][s] =  np.sum((input_parameters['k_p_max']*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value[indices_enhancer_well]*mesh.cellVolumes[indices_enhancer_well])
                                    dt_f['net_rate_of_mrna_increase'][s] =  np.sum((input_parameters['k_p_max']*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)
                                elif input_parameters['reaction_rate'] == 1.0:
                                    dt_f['molar_rate_of_mrna_production'][s] =  np.sum((input_parameters['k_p_max']*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value*mesh.cellVolumes)
                                    dt_f['molar_rate_of_mrna_production_in_enhancer_well'][s] =  np.sum((input_parameters['k_p_max']*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value[indices_enhancer_well]*mesh.cellVolumes[indices_enhancer_well])
                                    dt_f['net_rate_of_mrna_increase'][s] =  np.sum((input_parameters['k_p_max']*(phi_p > input_parameters['protein_threshold_mRNA_production'])-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)
                                elif input_parameters['reaction_rate'] == 2.0:
                                    kpx = input_parameters['k_p_max']*np.exp(-((mesh.cellCenters[0]+0.5*input_parameters['rna_nucleus_distance_to_protein'])**2 + (mesh.cellCenters[1])**2)/input_parameters['spread_kp']**2)
                                    dt_f['molar_rate_of_mrna_production'][s] =  np.sum(kpx.value*(phi_p.value - input_parameters['protein_threshold_mRNA_production'])*(phi_p.value > input_parameters['protein_threshold_mRNA_production'])*mesh.cellVolumes)
                                    dt_f['molar_rate_of_mrna_production_in_enhancer_well'][s] =  np.sum(kpx.value[indices_enhancer_well]*(phi_p.value[indices_enhancer_well] - input_parameters['protein_threshold_mRNA_production'])*(phi_p.value[indices_enhancer_well] > input_parameters['protein_threshold_mRNA_production'])*mesh.cellVolumes[indices_enhancer_well])
                                    dt_f['net_rate_of_mrna_increase'][s] =  np.sum((kpx*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)
                            
                            # pdb.set_trace()
                            
                            dt_f['rate_average_protein_in_lncrna_well'][:,0] = np.gradient(dt_f['average_protein_in_lncrna_well'][:,0], dt_f['t'][:,0])
                            dt_f['rate_average_mrna_in_lncrna_well'][:,0] = np.gradient(dt_f['average_mrna_in_lncrna_well'][:,0], dt_f['t'][:,0])
                            dt_f['rate_average_lncrna_in_lncrna_well'][:,0] = np.gradient(dt_f['average_lncrna_in_lncrna_well'][:,0], dt_f['t'][:,0])
                            
                            dt_f['rate_average_protein_in_enhancer_well'][:,0] = np.gradient(dt_f['average_protein_in_enhancer_well'][:,0], dt_f['t'][:,0])
                            dt_f['rate_average_mrna_in_enhancer_well'][:,0] = np.gradient(dt_f['average_mrna_in_enhancer_well'][:,0], dt_f['t'][:,0])
                            dt_f['rate_average_lncrna_in_enhancer_well'][:,0] = np.gradient(dt_f['average_lncrna_in_enhancer_well'][:,0], dt_f['t'][:,0])
                            


if __name__ == '__main__':
    
    target_directory = "/nobackup1c/users/npradeep96/PhaseField/LNCRNA_AND_ACTIVITY/LNCRNA_COUPLED_WITH_ACTIVITY/ACTIVE_MRNA_LNCRNA/"
    directory_name_pattern = r'.*_M_rna_0.01$'

    analysis_obj = LncrnaDataAnalysis()

    analysis_obj.calculate_dynamical_trajectories(target_directory, directory_name_pattern, condensate_concentration_cutoff=0.25, overwrite=True)

#   target_directory = "/nobackup1c/users/npradeep96/PhaseField/LNCRNA_AND_ACTIVITY/LNCRNA_COUPLED_WITH_ACTIVITY/LOCALIZED_MRNA_PRODUCTION/"
#   directory_name_pattern = r'.*_M_rna_2.0$'

#   analysis_obj = LncrnaDataAnalysis()

#   analysis_obj.calculate_dynamical_trajectories(target_directory, directory_name_pattern, condensate_concentration_cutoff=0.2, overwrite=True)
