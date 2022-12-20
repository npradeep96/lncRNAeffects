#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function that is called to initialize and run phase-field dynamics
"""
from __future__ import print_function
import fipy as fp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os

import pickle as pkl
import h5py

from utils.input_parse import *
from utils.graphics import *
from utils.write_files import *
from utils.sim_tools import *
from utils.free_energy import *


def run_CH(args):
	"""
	Function takes in path to input params, output_folder, and optionally params file that
	are passed while running code. With these parameters, this function initializes and runs
	phase-field simulations while writing output to files.

	**Input variables**

	-   args.i = Path to input params files (required)
	-   args.o = path to output folder (required)
	-   args.p = path to parameter file (optional)
	-   args.pN     =   Nth parameter to use from input (optional)

	"""

	# Read parameters from the input_parameters.txt file:
	input_parameters = input_parse(args.i)

	# Read the parameters in the param_list.txt file and choose the appropriate parameter to change:
	assert args.p is not None, "You need to supply a parameter file: param_list.txt!"
	params = input_parse(args.p, params_flag=True)
	par_name = str(list(params.keys())[0])
	par_values = params[par_name]   

	assert args.pN is not None, "You need to pick which parameter in param_list.txt needs to be used for this simulation!"
	par_values = par_values[int(args.pN)-1]
	input_parameters[par_name] = par_values 

	# Define the mesh:

	nx = input_parameters['nx']
	dx = input_parameters['dx']

	if input_parameters['dimension']==2:
		if int(input_parameters['circ_flag']):
			mesh = create_circular_mesh(radius=float(nx)*dx/2,cellSize=dx*1.5)
		else:
			mesh = fp.Grid2D(nx=nx, ny=nx, dx=dx, dy=dx)
			mesh = mesh-float(nx)*dx*0.5

	# Defining the cellVariables that store the concentrations of lncRNA and Protein within the small control volumes:

	phi_p = fp.CellVariable(mesh=mesh, name=r'$\phi_{prot}$', hasOld=True,value = input_parameters['phi_p_0'])
	phi_r = fp.CellVariable(mesh=mesh, name=r'$\phi_{lncRNA}$', hasOld=True,value = input_parameters['phi_r_0'])
	if input_parameters['mrna_flag']:
		phi_m = fp.CellVariable(mesh=mesh, name=r'$\phi_{mRNA}$', hasOld=True,value = input_parameters['phi_mrna_0'])

	# Defining cellVariables that store the positions of centers of the control volume

	X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
	Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

	# Define the free energy class object:

	rna_nucleus_location = (0.5*input_parameters['rna_nucleus_distance_to_protein'],0)
	protein_nucleus_location = (-0.5*input_parameters['rna_nucleus_distance_to_protein'],0)

	if input_parameters['free_energy_type'] == 1:
		FE = free_energy(c_alpha=input_parameters['c_alpha'],
						 c_beta=input_parameters['c_beta'],
						 rho_s=input_parameters['rho_s'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 chi=input_parameters['chi'],
						 kappa=input_parameters['kappa'],
						 c=input_parameters['c'],
						 r0 =rna_nucleus_location)
	elif input_parameters['free_energy_type'] == 2:
		FE = free_energy_protein_localized(c_alpha=input_parameters['c_alpha'],
						 c_beta=input_parameters['c_beta'],
						 rho_s=input_parameters['rho_s'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 chi=input_parameters['chi'],
						 kappa=input_parameters['kappa'],
						 c=input_parameters['c'],
						 r0 =rna_nucleus_location,
						 rE = protein_nucleus_location,
						 sigma = input_parameters['sigma'],
						 c_max = input_parameters['c_max'])
	elif input_parameters['free_energy_type'] == 3:
		FE = free_energy_gaussian_gaussian(c_alpha=input_parameters['c_alpha'],
						 c_beta=input_parameters['c_beta'],
						 rho_s=input_parameters['rho_s'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 chi=input_parameters['chi'],
						 kappa=input_parameters['kappa'],
						 c=input_parameters['c'],
						 r0 =rna_nucleus_location,
						 rE = protein_nucleus_location,
						 sigma = input_parameters['sigma'],
						 c_max = input_parameters['c_max'],
						 rho_r = input_parameters['rho_r'])
	elif input_parameters['free_energy_type'] == 4:
		FE = free_energy_protein_parabolic_lncrna_gaussian_mRNA(c_alpha=input_parameters['c_alpha'],
						 c_beta=input_parameters['c_beta'],
						 rho_s=input_parameters['rho_s'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 chi=input_parameters['chi'],
						 kappa=input_parameters['kappa'],
						 c=input_parameters['c'],
						 rho_m=input_parameters['rho_r'],
						 r0 =rna_nucleus_location,
						 rE = protein_nucleus_location,
						 sigma = input_parameters['sigma'],
						 c_max = input_parameters['c_max'])
	elif input_parameters['free_energy_type'] == 5:
		FE = free_energy_protein_nuc_RNA_gaussian_mRNA(c_alpha=input_parameters['c_alpha'],
						 c_beta=input_parameters['c_beta'],
						 rho_s=input_parameters['rho_s'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 chi=input_parameters['chi'],
						 kappa=input_parameters['kappa'],
						 c=input_parameters['c'],
						 rho_r=input_parameters['rho_r'],
						 rho_m=input_parameters['rho_m'],
						 rho_c=input_parameters['rho_c'],
						 r0 =rna_nucleus_location,
						 rE = protein_nucleus_location,
						 sigma = input_parameters['sigma'],
						 c_max = input_parameters['c_max'])
	elif input_parameters['free_energy_type'] == 6:
		FE = free_energy_protein_nuc_RNA_gaussian_mRNA_no_lr_int(c_alpha=input_parameters['c_alpha'],
						 c_beta=input_parameters['c_beta'],
						 rho_s=input_parameters['rho_s'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 chi=input_parameters['chi'],
						 kappa=input_parameters['kappa'],
						 c=input_parameters['c'],
						 rho_r=input_parameters['rho_r'],
						 r0 =rna_nucleus_location,
						 rE = protein_nucleus_location,
						 sigma = input_parameters['sigma'],
						 c_max = input_parameters['c_max'])
	elif input_parameters['free_energy_type'] == 7:
		FE = free_energy_protein_nuc_RNA_gaussian_mRNA_walled(c_alpha=input_parameters['c_alpha'],
						 c_beta=input_parameters['c_beta'],
						 rho_s=input_parameters['rho_s'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 chi_r=input_parameters['chi_r'],
						 chi_m=input_parameters['chi_m'],
						 kappa=input_parameters['kappa'],
						 c=input_parameters['c'],
						 rho_r=input_parameters['rho_r'],
						 rho_m=input_parameters['rho_m'],
						 rho_c=input_parameters['rho_c'],
						 r0 =rna_nucleus_location,
						 rE = protein_nucleus_location,
						 sigma = input_parameters['sigma'],
						 c_max = input_parameters['c_max'],
						 wall_k = input_parameters['wall_k'],
						 neg_max = input_parameters['neg_max'])

	
	# Nucleating the lncRNA and Protein sites appropriately:

	nucleate_seed(mesh,phi_p,
				phia_value=input_parameters['phi_p_nuc'],
				nucleus_size=input_parameters['nucleus_size'],
				dimension=input_parameters['dimension'], 
				location=protein_nucleus_location)

	if input_parameters['set_rna_profile'] and input_parameters['free_energy_type'] in [1,2,3,4]: 
		FE.set_initial_rna_profile(phi_r, mesh, input_parameters['rna_nucleus_phi'])
	elif input_parameters['set_rna_profile'] and input_parameters['free_energy_type'] in [5,6,7] and input_parameters['circ_flag']:
		FE.set_initial_rna_profile(phi_r, mesh, input_parameters['rna_nucleus_phi'], input_parameters['nx']*input_parameters['dx']*0.5, input_parameters['circ_flag'])
	else:
		nucleate_seed(mesh,phi_r,
				phia_value=input_parameters['rna_nucleus_phi'],
				nucleus_size=input_parameters['L_R'],
				dimension=input_parameters['dimension'], 
				location=rna_nucleus_location)

	# Define parameters associated with the numerical method to solve the PDEs:

	t = fp.Variable(0.0)
	dt = float(input_parameters['dt'])
	dt_max = float(input_parameters['dt_max'])
	dt_min = float(input_parameters['dt_min'])
	tolerance = float(input_parameters['tolerance'])
	max_change = float(input_parameters['max_change'])
	total_steps = int(input_parameters['total_steps'])
	duration = input_parameters['duration'];
	time_step = fp.Variable(dt)

	# Define the form of the PDEs:

	M_protein = float(input_parameters['M_protein'])
	
	if input_parameters['free_energy_type'] == 2 or input_parameters['free_energy_type'] == 4:
		M_rna = float(input_parameters['M_rna_times_K'])/float(input_parameters['K'])
	else:
		M_rna = float(input_parameters['M_rna'])

	if 'kp_kd_ratio' in input_parameters.keys():
		input_parameters['k_degradation'] = input_parameters['k_p_max']/input_parameters['kp_kd_ratio']

	if input_parameters['mrna_flag']:
		M_mrna = float(input_parameters['M_mrna'])
		reactions = RNA_reactions(mesh=mesh, k_p_max=input_parameters['k_p_max'], k_degradation=input_parameters['k_degradation'],
									spread=input_parameters['spread_kp'], center=protein_nucleus_location, phi_p_threshold = input_parameters['protein_threshold_mRNA_production'])

	mu_r_multiplier = FE.mu_r_multiplier(mesh) # fp.CellVariable(mesh = mesh, name = '$M_R K((r-r_0)^2/L^2_R+1)$',value=(M_rna*K/L_R**2)*((mesh.cellCenters[0]-rna_nucleus_location[0])**2 + (mesh.cellCenters[1]-rna_nucleus_location[1])**2))

	mu_p_multiplier = FE.mu_p_multiplier(mesh) # fp.CellVariable(mesh = mesh, name = '$M_P c_max exp(-(r-r_e)^2/L^2_e+1)$',value=M_protein*K/L_R**2)*((mesh.cellCenters[0]-rna_nucleus_location[0])**2 + (mesh.cellCenters[1]-rna_nucleus_location[1])**2))

	if input_parameters['mrna_flag'] and input_parameters['free_energy_type'] == 7:

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) - M_protein*heaviside_limit_flux(phi_p)*1.0*mu_p_multiplier.faceGrad.divergence

		eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - M_rna*1.0*mu_r_multiplier.faceGrad.divergence

		if input_parameters['reaction_rate'] == 0: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_flat_in_space(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 1: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_thresholded(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 2:   
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate(phi_p) - reactions.degradation_rate(phi_m)	

	if input_parameters['mrna_flag'] and input_parameters['free_energy_type'] == 6:

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) - M_protein*heaviside_limit_flux(phi_p)*1.0*mu_p_multiplier.faceGrad.divergence

		eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) - M_rna*heaviside_limit_flux(phi_r)*1.0*mu_r_multiplier.faceGrad.divergence

		if input_parameters['reaction_rate'] == 0: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_flat_in_space(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 1: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_thresholded(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 2:   
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate(phi_p) - reactions.degradation_rate(phi_m)	

	elif input_parameters['mrna_flag'] and input_parameters['free_energy_type'] == 5:

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) - M_protein*heaviside_limit_flux(phi_p)*1.0*mu_p_multiplier.faceGrad.divergence

		eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - M_rna*heaviside_limit_flux(phi_r)*1.0*mu_r_multiplier.faceGrad.divergence

		if input_parameters['reaction_rate'] == 0: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_flat_in_space(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 1: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_thresholded(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 2:   
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate(phi_p) - reactions.degradation_rate(phi_m)		

	elif input_parameters['mrna_flag'] and input_parameters['free_energy_type'] == 4:

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m,X_CV,Y_CV), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r), var=phi_r) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_m(phi_p,phi_m), var=phi_m) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) + fp.PowerLawConvectionTerm(coeff=M_protein*mu_p_multiplier.grad, var=phi_p)

		eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_r(phi_p,phi_r,X_CV,Y_CV), var=phi_r) + fp.PowerLawConvectionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*mu_r_multiplier.grad, var=phi_r)

		if input_parameters['reaction_rate'] == 0: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_flat_in_space(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 1: 
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_thresholded(phi_p) - reactions.degradation_rate(phi_m)	
		elif input_parameters['reaction_rate'] == 2:   
			eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate(phi_p) - reactions.degradation_rate(phi_m)		
		
	elif input_parameters['mrna_flag'] == 0 and input_parameters['free_energy_type'] in [2,3]:		

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,X_CV,Y_CV), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r), var=phi_r) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) + fp.PowerLawConvectionTerm(coeff=M_protein*mu_p_multiplier.grad, var=phi_p)

		eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_r(phi_p,phi_r,X_CV,Y_CV), var=phi_r) + fp.PowerLawConvectionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*mu_r_multiplier.grad, var=phi_r)

	elif input_parameters['mrna_flag'] == 0 and input_parameters['free_energy_type'] == 1:

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,X_CV,Y_CV), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r), var=phi_r) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p)

		eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_r(phi_p,phi_r,X_CV,Y_CV), var=phi_r) + fp.PowerLawConvectionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*mu_r_multiplier.grad, var=phi_r)		

	# Loop over time and solve the PDE:

	max_sweeps = int(input_parameters['max_sweeps'])
	phi_r_max = float(input_parameters['phi_r_max'])
	
	output_dir = input_parameters['output_dir'] + args.o + "/phi_r_" + str(input_parameters['rna_nucleus_phi']) + "_L_P_" + str(input_parameters['rna_nucleus_distance_to_protein']) + "_k_p_" + str(input_parameters['k_p_max'])  + "_reactiontype_" + str(input_parameters['reaction_rate']) + "_thresh_" + str(input_parameters['protein_threshold_mRNA_production']) + "_FE_" + str(input_parameters['free_energy_type']) + "_rhoc_" + str(input_parameters['rho_c']) + "_MP_" + str(input_parameters['M_protein']) +  "_Mmrna_" + str(input_parameters['M_mrna']) + "/"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	write_input_params(output_dir + '/input_params.txt',input_parameters)

	elapsed = 0.0
	steps = 0 
	
	phi_p.updateOld()
	phi_r.updateOld()

	if input_parameters['mrna_flag']:
		phi_m.updateOld()
		
	while (elapsed <= duration) and (steps <= total_steps) and (dt > dt_min):
				
		assert max(phi_r+phi_m) < phi_r_max, "Phi_r value surpassed 1.0. Aborting due to inaccurate approximations"  
		
		sweeps = 0
		
		while sweeps < max_sweeps:
			res1 = eqn0.sweep(dt=dt)
			res2 = eqn1.sweep(dt=dt)
			if input_parameters['mrna_flag']:
				res3 = eqn2.sweep(dt=dt)
			sweeps += 1		
			
		if input_parameters['mrna_flag']:
			delta_state = np.max([np.abs(np.max((phi_p-phi_p.old).value)),np.abs(np.max((phi_r-phi_r.old).value)),np.abs(np.max((phi_m-phi_m.old).value))])
		else:
			delta_state = np.max([np.abs(np.max((phi_p-phi_p.old).value)),np.abs(np.max((phi_r-phi_r.old).value))])

		# Write out simulation data to text files
		
		if steps % input_parameters['text_log'] == 0:
			
			# Write some simulation statistics for every "text_log" time steps to a text file
			if input_parameters['mrna_flag']:
				write_stats_dynamics(t=t.value, dt=dt, recorded_step=steps, phi_p=phi_p, phi_r=phi_r, phi_m=phi_m, X_CV=X_CV, Y_CV=Y_CV, mesh=mesh, FE=FE, res = (res1+res2+res3)/3, delta_s= delta_state, input_parameters=input_parameters, output_dir=output_dir)
			else:
				write_stats_dynamics(t=t.value, dt=dt, recorded_step=steps, phi_p=phi_p, phi_r=phi_r, phi_m=None, X_CV=X_CV, Y_CV=Y_CV, mesh=mesh, FE=FE, res = (res1+res2)/2, delta_s= delta_state, input_parameters=input_parameters, output_dir=output_dir)
				
		steps += 1
		elapsed += dt
		t.value = t.value+dt
		
		if delta_state > max_change:
			dt *= 0.8
		else:
			dt *= 1.1
			dt = min(dt, dt_max)

		# if (delta_state/dt) < tolerance:
		# 	break
		
		time_step.value = dt;
		phi_p.updateOld()
		phi_r.updateOld()
		if input_parameters['mrna_flag']:
			phi_m.updateOld()



if __name__ == "__main__":
	"""
		Function is called when python code is run on command line and calls run_CH
		to initialize the simulation
	"""
	parser = argparse.ArgumentParser(description='Take output filename to run CH simulations')
	parser.add_argument('--i',help="Name of input params", required = True)
	parser.add_argument('--p',help="Name of parameter file", required = False)
	parser.add_argument('--pN',help="Parameter number from file (indexed from 1)", required = False)

	parser.add_argument('--o',help="Name of output folder", required = True)
	args = parser.parse_args()

	run_CH(args)
