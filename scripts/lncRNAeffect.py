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

# import pickle as pkl
# import h5py

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
	
	if input_parameters['free_energy_type'] == 5:
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
	elif input_parameters['free_energy_type'] == 8:
		FE = free_energy_modified_FH(N_P=input_parameters['N_P'],
						 N_R=input_parameters['N_R'],
						 N_M=input_parameters['N_M'], 
						 chi_p=input_parameters['chi_p'],
						 chi_r=input_parameters['chi_r'],
						 chi_m=input_parameters	['chi_m'],
						 K=input_parameters['K'],
						 L_R = input_parameters['L_R'],
						 kappa=input_parameters['kappa'],
						 a=input_parameters['a'],
						 b=input_parameters['b'],
						 c=input_parameters['c'],
						 rho_r=input_parameters['rho_r'],
						 r0 =rna_nucleus_location,
						 rE = protein_nucleus_location,
						 sigma = input_parameters['sigma'],
						 c_max = input_parameters['c_max'])


	
	# Nucleating the lncRNA and Protein sites appropriately:

	nucleate_seed(mesh,phi_p,
				phia_value=input_parameters['phi_p_nuc'],
				nucleus_size=input_parameters['nucleus_size'],
				dimension=input_parameters['dimension'], 
				location=protein_nucleus_location)

	if input_parameters['set_rna_profile_from_file']:
		with open(input_parameters['rna_profile_filename'].format(L_P=input_parameters['rna_nucleus_distance_to_protein'],phi_r=input_parameters['phi_r_0']), 'rb') as f:
			phi_r_array = pkl.load(f)
			phi_r.value = phi_r_array

	if input_parameters['set_rna_profile'] and input_parameters['free_energy_type'] in [5,7,8] and input_parameters['circ_flag']:
		FE.set_initial_rna_profile(phi_r, mesh, input_parameters['rna_nucleus_phi'], input_parameters['nx']*input_parameters['dx']*0.5, input_parameters['circ_flag'])
	elif input_parameters['nucleate_seed_RNA']:
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
	
	M_rna = float(input_parameters['M_rna'])

	if input_parameters['mrna_flag']:
		M_mrna = float(input_parameters['M_mrna'])
		reactions = RNA_reactions(mesh=mesh, k_p_max=input_parameters['k_p_max'], k_degradation=input_parameters['k_degradation'],
									spread=input_parameters['spread_kp'], center=protein_nucleus_location, phi_threshold = input_parameters['protein_threshold_mRNA_production'], n_hill=input_parameters['n_hill'])

	if input_parameters['lncRNA_reactions_flag']:

		lncRNA_reactions = RNA_reactions(mesh=mesh, k_p_max=input_parameters['k_p_max_lncRNA'], k_degradation=input_parameters['k_degradation_lncRNA'],
									spread=input_parameters['spread_kp_lncRNA'], center=rna_nucleus_location, phi_threshold = input_parameters['protein_threshold_lncRNA_production'], n_hill=input_parameters['lncRNA_n_hill'])

	mu_r_multiplier = FE.mu_r_multiplier(mesh) # fp.CellVariable(mesh = mesh, name = '$M_R K((r-r_0)^2/L^2_R+1)$',value=(M_rna*K/L_R**2)*((mesh.cellCenters[0]-rna_nucleus_location[0])**2 + (mesh.cellCenters[1]-rna_nucleus_location[1])**2))

	mu_p_multiplier = FE.mu_p_multiplier(mesh) # fp.CellVariable(mesh = mesh, name = '$M_P c_max exp(-(r-r_e)^2/L^2_e+1)$',value=M_protein*K/L_R**2)*((mesh.cellCenters[0]-rna_nucleus_location[0])**2 + (mesh.cellCenters[1]-rna_nucleus_location[1])**2))

	if input_parameters['mrna_flag'] and input_parameters['free_energy_type'] == 8:

		# D_protein = M_protein
		# eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=D_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=D_protein*mu_p_multiplier, var=phi_p) - D_protein*(mu_p_multiplier*phi_p).faceGrad.divergence + fp.DiffusionTerm(coeff=D_protein*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=D_protein*FE.dmu_p_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - fp.DiffusionTerm(coeff=(D_protein*phi_p,FE.kappa),var=phi_p)
		Mp1 = M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m)
		Mp2 = M_protein*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m)
		Mp3 = M_protein*FE.dmu_p_dphi_m(phi_p,phi_r,phi_m)
		M_conv = M_protein*mu_p_multiplier.grad
		kappa = fp.CellVariable(mesh=mesh, value=FE.kappa, rank=0)
		
		# Mp1.constrain(0.0, mesh.exteriorFaces) 
		# Mp2.constrain(0.0, mesh.exteriorFaces) 
		# Mp3.constrain(0.0, mesh.exteriorFaces) 
		# M_conv.constrain(0.0, mesh.exteriorFaces) 
		# kappa.constrain(0.0, mesh.exteriorFaces)

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=Mp1, var=phi_p) + fp.DiffusionTerm(coeff=Mp2, var=phi_r) + fp.DiffusionTerm(coeff=Mp3, var=phi_m) - fp.PowerLawConvectionTerm(coeff=M_conv, var=phi_p) - fp.DiffusionTerm(coeff=(M_protein*phi_p,kappa),var=phi_p) # + (mesh.exteriorFaces * 0.0).divergence

		if input_parameters['lncRNA_reactions_flag']:
			if input_parameters['lncRNA_reaction_rate'] == 2: 
				eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna, var=phi_r) + lncRNA_reactions.production_rate(phi_p) - lncRNA_reactions.degradation_rate(phi_r)	
			elif input_parameters['lncRNA_reaction_rate'] == 3:
				eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna, var=phi_r) + lncRNA_reactions.production_rate_hill_gaussian(phi_p) - lncRNA_reactions.degradation_rate(phi_r)
			elif input_parameters['lncRNA_reaction_rate'] == 4:
				eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna, var=phi_r) + lncRNA_reactions.production_rate_no_concentration_dependence() - lncRNA_reactions.degradation_rate(phi_r)
		else:
			# eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=D_rna*FE.dmu_r_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=D_rna*mu_r_multiplier, var=phi_r) - D_rna*(mu_r_multiplier*phi_r).faceGrad.divergence + fp.DiffusionTerm(coeff=D_rna*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=D_rna*FE.dmu_r_dphi_m(phi_p,phi_r,phi_m), var=phi_m)
			eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - fp.PowerLawConvectionTerm(coeff=M_rna*mu_r_multiplier.grad, var=phi_r)

		if input_parameters['mrna_flag']:
			if input_parameters['reaction_rate'] == 0: 
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_flat_in_space(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 1: 
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_thresholded(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 2:   
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 3:  
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_hill_gaussian(phi_p) - reactions.degradation_rate(phi_m)	


	if input_parameters['mrna_flag'] and input_parameters['free_energy_type'] == 7:

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) - M_protein*heaviside_limit_flux(phi_p)*mu_p_multiplier.faceGrad.divergence # ((mu_p_multiplier.faceGrad - mesh.exteriorFaces*mu_p_multiplier.faceGrad).divergence)

		if input_parameters['lncRNA_reactions_flag']:
			if input_parameters['lncRNA_reaction_rate'] == 2: 
				eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna, var=phi_r) + lncRNA_reactions.production_rate(phi_p) - lncRNA_reactions.degradation_rate(phi_r)
			elif input_parameters['lncRNA_reaction_rate'] == 3:
				eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna, var=phi_r) + lncRNA_reactions.production_rate_hill_gaussian(phi_p) - lncRNA_reactions.degradation_rate(phi_r)
			elif input_parameters['lncRNA_reaction_rate'] == 4:
				eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna, var=phi_r) + lncRNA_reactions.production_rate_no_concentration_dependence() - lncRNA_reactions.degradation_rate(phi_r)
		else:
			eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_rna*FE.dmu_r_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - M_rna*heaviside_limit_flux(phi_r)*mu_r_multiplier.faceGrad.divergence # ((mu_r_multiplier.faceGrad - mesh.exteriorFaces*mu_r_multiplier.faceGrad).divergence)

		if input_parameters['mrna_flag']:
			if input_parameters['reaction_rate'] == 0: 
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_flat_in_space(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 1: 
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_thresholded(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 2:   
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 3:  
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_hill_gaussian(phi_p) - reactions.degradation_rate(phi_m)	

	elif input_parameters['mrna_flag'] and input_parameters['free_energy_type'] == 5:

		eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_protein*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) - M_protein*heaviside_limit_flux(phi_p)*((mu_p_multiplier.faceGrad - mesh.exteriorFaces*mu_p_multiplier.faceGrad).divergence)

		if input_parameters['lncRNA_reactions_flag']:
			eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_m(phi_p,phi_r,phi_m), var=phi_m) + lncRNA_reactions.production_rate_no_concentration_dependence() - lncRNA_reactions.degradation_rate(phi_r)
		else:
			eqn1 = fp.TransientTerm(coeff=1.,var=phi_r) == fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_p_dphi_r(phi_p,phi_r,phi_m), var=phi_p) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_r(phi_p,phi_r,phi_m), var=phi_r) + fp.DiffusionTerm(coeff=M_rna*heaviside_limit_flux(phi_r)*FE.dmu_r_dphi_m(phi_p,phi_r,phi_m), var=phi_m) - M_rna*heaviside_limit_flux(phi_r)*((mu_r_multiplier.faceGrad - mesh.exteriorFaces*mu_r_multiplier.faceGrad).divergence)

		if input_parameters['mrna_flag']:
			if input_parameters['reaction_rate'] == 0: 
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_flat_in_space(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 1: 
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_thresholded(phi_p) - reactions.degradation_rate(phi_m)	
			elif input_parameters['reaction_rate'] == 2:   
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate(phi_p) - reactions.degradation_rate(phi_m)		
			elif input_parameters['reaction_rate'] == 3:  
				eqn2 = fp.TransientTerm(coeff=1.,var=phi_m) == fp.DiffusionTerm(coeff=M_mrna, var=phi_m) + reactions.production_rate_hill_gaussian(phi_p) - reactions.degradation_rate(phi_m)	
	
	# Enforce a Dirichlet boundary condition for mRNA if applicable

	if input_parameters['mrna_flag'] and input_parameters['dirichlet_BC_mRNA']:
		if 'r_boundary' in input_parameters.keys():
			X,Y = mesh.faceCenters
			mask = ((X**2 + Y**2) < input_parameters['r_boundary']**2)
			phi_m.constrain(input_parameters['phi_m_boundary'], where=mesh.exteriorFaces | ~mask)
		else:
			phi_m.constrain(input_parameters['phi_m_boundary'], where=mesh.exteriorFaces)
			phi_r.constrain(input_parameters['phi_m_boundary'], where=mesh.exteriorFaces)

		# FE.mu_p(phi_p, phi_r, phi_m, X_CV, Y_CV).faceGrad.constrain(0.0, where=mesh.exteriorFaces)
		# FE.mu_r(phi_p, phi_r, phi_m, X_CV, Y_CV).faceGrad.constrain(0.0, where=mesh.exteriorFaces)

	# Loop over time and solve the PDE:

	max_sweeps = int(input_parameters['max_sweeps'])
	phi_r_max = float(input_parameters['phi_r_max'])
	
	# uncomment below when we have active lncRNA and mRNA production
	output_dir = input_parameters['output_dir'] + args.o + "/phi_p0_" + str(input_parameters['phi_p_0']) + "_k_p_lncRNA_" + str(input_parameters['k_p_max_lncRNA']) + "_L_P_" + str(input_parameters['rna_nucleus_distance_to_protein']) + "_k_p_" + str(input_parameters['k_p_max'])  + "_reactiontype_" + str(input_parameters['reaction_rate']) + "_thresh_" + str(input_parameters['protein_threshold_mRNA_production']) + "_FE_" + str(input_parameters['free_energy_type']) + "_MP_" + str(input_parameters['M_protein']) +  "_Mmrna_" + str(input_parameters['M_mrna']) + "_M_rna_" + str(input_parameters['M_rna']) + "_c_max_" + str(input_parameters['c_max']) + '_phi_r_0' + str(input_parameters['phi_r_0']) +  "/"# input_parameters['output_dir'] + args.o + "k_p_" + str(input_parameters['k_p_max']) + "_reactiontype_" + str(input_parameters['reaction_rate']) + "_thresh_" + str(input_parameters['protein_threshold_mRNA_production']) + "_FE_" + str(input_parameters['free_energy_type']) + "_MP_" + str(input_parameters['M_protein']) +  "_Mmrna_" + str(input_parameters['M_mrna']) + "/"
	# output_dir = input_parameters['output_dir'] + args.o + "/phi_r_" + str(input_parameters['rna_nucleus_phi']) + "_k_p_lncRNA_" + str(input_parameters['k_p_max_lncRNA']) + "_L_P_" + str(input_parameters['rna_nucleus_distance_to_protein']) + "_k_p_" + str(input_parameters['k_p_max'])  + "_reactiontype_" + str(input_parameters['reaction_rate']) + "_thresh_" + str(input_parameters['protein_threshold_mRNA_production']) + "_FE_" + str(input_parameters['free_energy_type']) + "_rhoc_" + str(input_parameters['rho_c']) + "_MP_" + str(input_parameters['M_protein']) +  "_Mmrna_" + str(input_parameters['M_mrna']) + "_M_rna_" + str(input_parameters['M_rna']) + "/"

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
				
		# assert max(phi_r+phi_m) < phi_r_max, "Phi_r value surpassed 1.0. Aborting due to inaccurate approximations"  
		
		if steps == 0:
			
			# Make a directory to store that contains text files with concentration profile and their gradient cellVariables()
			if not os.path.exists(output_dir + 'Mesh/'):
				os.makedirs(output_dir + 'Mesh/')
				
			# Make a directory to store the image files of concentration profiles
			if not os.path.exists(output_dir + 'Images/'):
				os.makedirs(output_dir + 'Images/')    

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

		
		# Check if the concnetration variables are changing crazily
		if delta_state > max_change: 
			dt *= 0.8
			continue
		
		else:

			# Write out simulation data to text files
			
			if steps % input_parameters['text_log'] == 0:
				
				# Write the concentration profile and their gradient cellVariables() to a text file
				if input_parameters['mrna_flag']:
					fp.TSVViewer(vars=[phi_p,phi_r,phi_m,phi_p.grad,phi_r.grad,phi_m.grad]).plot(filename=output_dir + "Mesh/mesh_{step}.txt".format(step=steps))
				else:
					fp.TSVViewer(vars=[phi_p,phi_r,phi_p.grad,phi_r.grad]).plot(filename=output_dir + "Mesh/mesh_{step}.txt".format(step=steps))
				
				# Write some simulation statistics for every "text_log" time steps to a text file
				if input_parameters['mrna_flag']:
					write_stats(t=t.value, dt=dt, steps=steps, phi_p=phi_p, phi_r=phi_r, phi_m=phi_m, X_CV=X_CV, Y_CV=Y_CV, mesh=mesh, FE=FE, res = (res1+res2+res3)/3, delta_s= delta_state,  output_dir=output_dir)
				else:
					write_stats(t=t.value, dt=dt, steps=steps, phi_p=phi_p, phi_r=phi_r, phi_m=None, X_CV=X_CV, Y_CV=Y_CV, mesh=mesh, FE=FE, res = (res1+res2)/2, delta_s= delta_state,  output_dir=output_dir)
				
			# Making figures and storing simulation data relevant to making figures
			
			if steps % input_parameters['image_checkpoint'] == 0:
				
				# Create image files containing concentration profiles of the species
				if (input_parameters['dimension']==2) and (int(input_parameters['plot_flag'])):
					plot_spatial_variables(mesh=mesh, spatial_variable=phi_p, variable_name='Phi_P', steps=steps,
										   colormap="Blues", value_range=[0.0,float(input_parameters['phi_max_plot'])],
										   output_dir=output_dir)
					plot_spatial_variables(mesh=mesh, spatial_variable=phi_r, variable_name='Phi_R', steps=steps,
										   colormap="Reds", value_range=[0.0,float(input_parameters['phi_r_max_plot'])],
										   output_dir=output_dir)
					if input_parameters['mrna_flag']:
						plot_spatial_variables(mesh=mesh, spatial_variable=phi_m, variable_name='Phi_M', steps=steps,
										   colormap="Reds", value_range=[0.0,float(input_parameters['phi_r_max_plot'])],
										   output_dir=output_dir)

				# Write spatial variables into a HDF5 file
				if input_parameters['mrna_flag']:
					list_of_variables = write_spatial_vars_to_hdf5_file(phi_p=phi_p, phi_r=phi_r, phi_m = phi_m, 
												X_CV=X_CV, Y_CV=Y_CV, FE=FE, output_dir=output_dir, 
												recorded_step=int(steps)/int(input_parameters['image_checkpoint']), 
												total_recorded_steps=
												int(np.ceil(input_parameters['total_steps']/input_parameters['image_checkpoint']))+1)
				else:
					list_of_variables = write_spatial_vars_to_hdf5_file(phi_p=phi_p, phi_r=phi_r, phi_m = None, 
												X_CV=X_CV, Y_CV=Y_CV, FE=FE, output_dir=output_dir, 
												recorded_step=int(steps)/int(input_parameters['image_checkpoint']), 
												total_recorded_steps=
												int(np.ceil(input_parameters['total_steps']/input_parameters['image_checkpoint']))+1)
				
			steps += 1
			elapsed += dt
			t.value = t.value+dt
			
			dt *= 1.1
			dt = min(dt, dt_max)
			time_step.value = dt;

			phi_p.updateOld()
			phi_r.updateOld()
			if input_parameters['mrna_flag']:
				phi_m.updateOld()

	# with open('phi_r_ss.pkl','wb') as f:
	# 	pkl.dump(phi_r, f)

	write_movie_from_hdf5(output_dir, list_of_variables, mesh)

	# write_movie_from_hdf5_combined(output_dir, list_of_variables, mesh, input_parameters['mrna_flag'])



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
