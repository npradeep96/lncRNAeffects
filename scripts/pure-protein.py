#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fipy as fp
import os
import numpy as np
import matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import mayavi
import pickle as pkl
import h5py
import moviepy.editor as mp


# ## Parameter file to be read by the script

# In[2]:


input_parameters = {

# Parameters
#   Initialization parameters
#   This line will not be read & similarly blank lines
'dimension':2,
'plot_flag':1,
'circ_flag':1,
'phi_max_plot':0.8,
'free_energy_flag':2,
	
# Directory to output data
'output_dir':"/nobackup1c/users/npradeep96/PhaseField/PURE_PROTEIN/linear_phip_test_nucsize_0.0_K_0.2_phip0_0.1_ca_0.2_cb_0.6_LR_10_kappa_10.0/",

#   Define size of grid (nx:number of cells per dimension): dx : length of cell
'nx':300,
'dx':0.2,

#   These represent the concentrations of coexistence for protein alone
#   If Flory-Huggins potential is employed: they have to be symmetric about 0.5
'c_alpha':0.2,
'c_beta':0.6,

#   Define initial concentrations of the protein & RNA and nucleus size
'phi_p_0':0.1,

#   Mobility of protein and RNA
'M_protein':1.0,

#   Simulation parameters
#   dt is initial step size
#   dt_max & min are maximum/minimum step sizes that are allowed
#   tolerance is the residual below which simulation stops
#   total_steps/duration are number of simulation steps/time to run (whichever occurs first)
#   checkpoint is frequency of logging simulation data
'dt':1e-4,
'dt_max':0.5,
'dt_min':1e-8,
'tolerance':1e-5,
'total_steps':1500,
'image_checkpoint':50,
'text_log':50,
'duration':600.0,
'max_sweeps':3,

#   Sets up free-energy related terms
#   fh is a flag for employing Flory-Huggins instead of double-well
#   changing_chi ::2 uses the gaussian form & 1 :: uses double-well LG expression
#   changing_chi ::0 is not changing_chi and there for backwards compatability
#   rho_s/rho_r is height of double-well potential for protein/RNA respectively
#   kappa is surface tension parameter for protein: default : 0.5
#   chi is value of pairwise interaction
#   a:b:c are values of landau-ginzburg like three-way interaction & four way interactions
#   mu_r chooses whether you use D-R (mu_r:0) or chemical potential fo RNA (mu_r:1)
'rho_s':3.0,
'kappa':10.0,
'r0':(0.0,0.0),
'sigma':10.0,
'K':0.2,

#  Parameters associated with nucleated lncRNA and Protein

# Size and location of the protein nucleus
'nucleus_size':0.0,
'phi_p_nuc':0.55
}


# ## Definition for the class object for free energy

# In[3]:


# Defining a class capturing the free energy of interaction for a pure protein system

class free_energy():
	
	def __init__(self, rho_s, c_alpha, c_beta, kappa, r0, sigma, K):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.kappa = kappa
		self.r0 = r0
		self.sigma = sigma
		self.K = K
	
	def f(self, phi_p, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - K*exp(-\frac{|r-r_0|^2}{sigma^2})\phi^2_p + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 -
			   self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.sigma**2)*phi_p**2 + 
			   0.5*self.kappa*(phi_p.grad.mag)**2)
	
	def mu_p(self, phi_p, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha + self.c_beta) - 
				self.kappa*(phi_p.faceGrad.divergence) - 
				2.0*self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.sigma**2)*phi_p)
	

	def dmu_p_dphi_p(self, phi_p, X_CV, Y_CV):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) 
				- 2.0*self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.sigma**2))

	
	def gaussian_multiplier(self, X_CV, Y_CV):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return (2.0*self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.sigma**2))


class free_energy_linear_phip():
	
	def __init__(self, rho_s, c_alpha, c_beta, kappa, r0, sigma, K):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.kappa = kappa
		self.r0 = r0
		self.sigma = sigma
		self.K = K
	
	def f(self, phi_p, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - K*exp(-\frac{|r-r_0|^2}{sigma^2})\phi^2_p + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 -
			   2.0*self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.sigma**2)*phi_p + 
			   0.5*self.kappa*(phi_p.grad.mag)**2)
	
	def mu_p(self, phi_p, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha + self.c_beta) - 
				self.kappa*(phi_p.faceGrad.divergence) - 
				2.0*self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.sigma**2))
	

	def dmu_p_dphi_p(self, phi_p, X_CV, Y_CV):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2))

	
	def gaussian_multiplier(self, X_CV, Y_CV):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return (2.0*self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.sigma**2))

# ## Some other helper functions

# In[4]:


# Function to create a circular mesh using gmsh:

from fipy import Gmsh2D

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


# In[5]:


# Function to nucleate seed of a particular size

def nucleate_seed(mesh,phi_a,phia_value,nucleus_size=5.0,dimension=2, location=(0,0)):
	"""
	Function nucleates spherical nucleus of condensate into mesh

	**Input**

	-   phi_a   =   Phase-field variable
	-   mesh    =   Mesh variable
	-   phia_value  =   Value of dense phase to nucleate
	-   nucleus_size   =   Radius of initial nucleus
	-   dimension   =   Dimension of grid
	"""
	a=(mesh.cellCenters)

	xc = (min(mesh.x) + max(mesh.x))*0.5
	yc = (min(mesh.y) + max(mesh.y))*0.5
	
	# modify xc and yc with location coords
	xc += location[0]
	yc += location[1]
	
	if dimension==3:
		zc = (min(mesh.z) + max(mesh.z))*0.5;

	for i in np.arange(a.shape[1]):
		if dimension==2:
			dist = np.sqrt((a.value[0][i]-xc)**2 + (a.value[1][i]-yc)**2)
		elif dimension==3:
			dist = np.sqrt((a.value[0][i]-xc)**2 + (a.value[1][i]-yc)**2 + (a.value[2][i]-zc)**2)

		if (dist<=nucleus_size):
			phi_a.value[i] = phia_value


# In[6]:


# Function to generate images of the spatial profiles of different cellVariables()

def plot_spatial_variables(mesh, spatial_variable, variable_name, steps, colormap, value_range, output_dir):
	
	fig, ax = plt.subplots()
	cs = ax.tricontourf(
		mesh.x.value, mesh.y.value,
		spatial_variable.value,
		cmap = plt.cm.get_cmap(colormap),
		levels = np.linspace(value_range[0], value_range[1], 256)
		)
	fig.colorbar(cs)
	try:
		ax.set_title(spatial_variable.name)
	except:
		print("No name given for the spatial variable while plotting")
		raise
		
	fig.savefig(fname=output_dir + 'Images/' + variable_name + '_{step}.png'.format(step=steps),dpi=600,format='png')
	#fig.savefig(fname=output_dir + 'Images/' + variable_name + '_{step}.svg'.format(step=steps),dpi=600,format='svg')
	#pkl.dump((fig,ax),file(output_dir + 'Images/' + variable_name +'_{step}.pickle'.format(step=steps),'w'))
	plt.close()


# In[7]:


# Function to dump simulation data into HDF5 files

def write_spatial_vars_to_hdf5_file(phi_p, X_CV, Y_CV, FE, output_dir, recorded_step, total_recorded_steps):

	list_of_spatial_variables = ["phi_p", "mu_p", "free_energy"] 
	
	if recorded_step == 0:
		
		number_of_CVs = len(phi_p)   

		with h5py.File(os.path.join(output_dir + 'spatial_variables.hdf5'),'w') as f:
			for sv in list_of_spatial_variables:
				f.create_dataset(sv, (total_recorded_steps,number_of_CVs))
		
	with h5py.File(output_dir + 'spatial_variables.hdf5', 'a') as f:
		
		f["phi_p"][recorded_step,:] = phi_p.value        
		f["mu_p"][recorded_step,:] = FE.mu_p(phi_p, X_CV, Y_CV)
		f["free_energy"][recorded_step,:] = FE.f(phi_p, X_CV, Y_CV)
		
	return list_of_spatial_variables


# In[8]:


# Function to write statistics associated with the simulation to a text file

def write_stats(t, dt, steps, phi_p, X_CV, Y_CV, mesh, FE, res, delta_s, output_dir):
	
	stats_list = ["step", "t", "dt","phi_p_avg","phi_p_min","phi_p_max","residual","delta_state","free_energy"]
	
	if steps == 0:
		with open(output_dir + "/stats.txt", 'w+') as stats:
			stats.write("\t".join(stats_list) + "\n")
	
	stats_simulation = ["{}".format(int(steps)),
						"{:.8f}".format(t),
						"{:.3e}".format(dt),
						"{:.4e}".format(phi_p.cellVolumeAverage.value),
						"{:.4e}".format(min(phi_p)),
						"{:.4e}".format(max(phi_p)),
						"{:.4e}".format(res),
						"{:.4e}".format(delta_s),
						"{:.4e}".format(np.sum((FE.f(phi_p,X_CV,Y_CV)*mesh.cellVolumes).value))
						]
	
	assert len(stats_list) == len(stats_simulation), "Aborting as the number of fields in the simulation statistics file is different from the number of values supplied!"

	with open(output_dir + "/stats.txt", 'a') as stats:
		stats.write("\t".join(stats_simulation) + "\n")


# In[9]:


# Function to write movies from hdf5 file

def write_movie_from_hdf5(PATH, names, mesh, fps=5):
	
	def key_funct(x):
		return int(x.split('_')[-1].rstrip('.png'))

	# make directory
	os.mkdir(os.path.join(PATH, 'movies'))
	df_total = h5py.File(os.path.join(PATH, "spatial_variables.hdf5"), mode="r")
	
	for name in names:
		
		max_val = df_total[name][:].max()
		min_val = df_total[name][:].min()

		if len(df_total[name][:].shape) == 2:
			df_ = [df_total[name][:]]
		elif len(df_total[name][:].shape) == 3:
			df_ = [df_total[name][:,0,:], df_total[name][:,1,:]]
		
		for idx, df in enumerate(df_):
			
			for i in range(df.shape[0]):
				fig, ax = plt.subplots()
				cs = ax.tricontourf(mesh.x.value,
									mesh.y.value,
									df[i],
									levels=np.linspace(min_val,max_val,256),
									cmap='coolwarm')
				fig.colorbar(cs)
				ax.set_title(name)

				fig.savefig(fname=PATH +'/movies/{n}_step_{step}.png'.format(n=name, step=i),dpi=300,format='png')
				plt.close(fig)

			file_names = sorted(list((fn for fn in os.listdir(os.path.join(PATH, 'movies')) if fn.endswith('.png'))), key=key_funct)

			file_paths = [os.path.join(PATH, 'movies', f) for f in file_names]
			clip = mp.ImageSequenceClip(file_paths, fps=fps)
			clip.write_videofile(os.path.join(PATH, 'movies','{n}_{idx}.mp4'.format(n=name, idx=idx)), fps=fps)
			clip.close()

			# delete individual images
			for f in file_paths:
				os.remove(f)

# Function to write out input parameters

def write_input_params(file_output,input_params):
    """
    write_input_params writes all the input information from
    *input_params* into *file_output* in the same syntax
    """

    with open(file_output,'w+') as f:
        for key in input_params.keys():
            f.write( ''.join(key)+','+str(input_params[key])+'\n')
    f.close()

# ## Script to run simulations

# In[10]:


# Define the mesh

nx = input_parameters['nx']
dx = input_parameters['dx']

if input_parameters['dimension']==2:
	if int(input_parameters['circ_flag']):
		mesh = create_circular_mesh(radius=float(nx)*dx/2,cellSize=dx*1.5)
	else:
		mesh = fp.Grid2D(nx=nx, ny=nx, dx=dx, dy=dx)
		mesh = mesh-float(nx)*dx*0.5


# In[11]:


# Defining the cellVariables that store the concentrations of lncRNA and Protein within the small control volumes

phi_p = fp.CellVariable(mesh=mesh, name=r'$\phi_{prot}$', hasOld=True,value = input_parameters['phi_p_0'])

# Defining cellVariables that store the positions of centers of the control volume

X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])


# In[12]:


# Define the free energy class object:
if input_parameters['free_energy_flag'] == 1:
	FE = free_energy(c_alpha=input_parameters['c_alpha'],
					 c_beta=input_parameters['c_beta'],
					 rho_s=input_parameters['rho_s'],
					 r0=input_parameters['r0'],
					 sigma = input_parameters['sigma'],
					 kappa=input_parameters['kappa'],
					 K=input_parameters['K']
					)
elif input_parameters['free_energy_flag'] == 2:
	FE = free_energy_linear_phip(c_alpha=input_parameters['c_alpha'],
					 c_beta=input_parameters['c_beta'],
					 rho_s=input_parameters['rho_s'],
					 r0=input_parameters['r0'],
					 sigma = input_parameters['sigma'],
					 kappa=input_parameters['kappa'],
					 K=input_parameters['K']
					)


# In[13]:


# Nucleating the lncRNA and Protein sites appropriately

nucleate_seed(mesh,phi_p,
			phia_value=(input_parameters['phi_p_nuc']),
			nucleus_size=input_parameters['nucleus_size'],
			dimension=input_parameters['dimension'], 
			location=input_parameters['r0'])


# In[14]:


# Define parameters associated with the numerical method to solve the PDEs:

t = fp.Variable(0.0)
dt = input_parameters['dt']
dt_max = input_parameters['dt_max']
dt_min = input_parameters['dt_min']
tolerance = input_parameters['tolerance']
total_steps = int(input_parameters['total_steps'])
duration = input_parameters['duration'];
time_step = fp.Variable(dt)


# In[15]:


# Define the form of the PDEs

M_protein = input_parameters['M_protein']
K = input_parameters['K']
sigma = input_parameters['sigma']



if input_parameters['free_energy_flag'] == 1:
	mu_p_multiplier = fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier',value=M_protein*2*K*np.exp(-((mesh.cellCenters[0]-input_parameters['r0'][0])**2 + (mesh.cellCenters[1]-input_parameters['r0'][1])**2)/sigma**2))
	eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,X_CV,Y_CV), var=phi_p) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) - fp.PowerLawConvectionTerm(coeff=mu_p_multiplier.grad, var=phi_p)
elif input_parameters['free_energy_flag'] == 2:
	mu_p_multiplier = fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier',value=M_protein*2*K*np.exp(-((mesh.cellCenters[0]-input_parameters['r0'][0])**2 + (mesh.cellCenters[1]-input_parameters['r0'][1])**2)/sigma**2))
	eqn0 = fp.TransientTerm(coeff=1.,var=phi_p) == fp.DiffusionTerm(coeff=M_protein*FE.dmu_p_dphi_p(phi_p,X_CV,Y_CV), var=phi_p) - fp.DiffusionTerm(coeff=(M_protein,FE.kappa),var=phi_p) - (M_protein * mu_p_multiplier.faceGrad).divergence


# In[16]:


# Step over time and solve the PDE:

max_sweeps = input_parameters['max_sweeps']
output_dir = input_parameters['output_dir']

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

write_input_params(output_dir + '/input_params.txt',input_parameters)

elapsed = 0.0
steps = 0 
phi_p.updateOld()
	
while (elapsed <= duration) and (steps <= total_steps) and (dt > dt_min):
	
	res1 = 1e10
	
	sweeps = 0
	
	while sweeps < max_sweeps:
		res1 = eqn0.sweep(dt=dt)
		# print(res1,res2)
		sweeps += 1
	
	delta_state = np.abs(np.mean((phi_p-phi_p.old).value))

	if steps == 0:
		
		# Make a directory to store that contains text files with concentration profile and their gradient cellVariables()
		if not os.path.exists(output_dir + 'Mesh/'):
			os.makedirs(output_dir + 'Mesh/')
			
		# Make a directory to store the image files of concentration profiles
		if not os.path.exists(output_dir + 'Images/'):
			os.makedirs(output_dir + 'Images/')    
	
	# Write out simulation data to text files
	
	if steps % input_parameters['text_log'] == 0:
		
		# Write the concentration profile and their gradient cellVariables() to a text file
		fp.TSVViewer(vars=[phi_p,phi_p.grad]).plot(filename=output_dir + "Mesh/mesh_{step}.txt".format(step=steps))
		
		# Write some simulation statistics for every "text_log" time steps to a text file
		write_stats(t=t.value, dt=dt, steps=steps, phi_p=phi_p, X_CV=X_CV, Y_CV=Y_CV, mesh=mesh, FE=FE, res=res1, delta_s=delta_state, output_dir=output_dir)
		
	# Making figures and storing simulation data relevant to making figures
	
	if steps % input_parameters['image_checkpoint'] == 0:
		
		# Create image files containing concentration profiles of the species
		if (input_parameters['dimension']==2) and (input_parameters['plot_flag']):
			plot_spatial_variables(mesh=mesh, spatial_variable=phi_p, variable_name='Phi_P', steps=steps,
								   colormap="Blues", value_range=[0.0,input_parameters['phi_max_plot']],
								   output_dir=output_dir)
		# Write spatial variables into a HDF5 file
		list_of_variables = write_spatial_vars_to_hdf5_file(phi_p=phi_p, 
										X_CV=X_CV, Y_CV=Y_CV, FE=FE, output_dir=output_dir, 
										recorded_step=int(steps/input_parameters['image_checkpoint']), 
										total_recorded_steps=
										int(input_parameters['total_steps']/input_parameters['image_checkpoint'])+1)
		
	steps += 1
	elapsed += dt
	t.value = t.value+dt

	if delta_state > tolerance:
		dt *= 0.8
	else:
		dt *= 1.1
		dt = min(dt, dt_max)
	
	time_step.value = dt;
	phi_p.updateOld()
		
write_movie_from_hdf5(output_dir, list_of_variables, mesh)

