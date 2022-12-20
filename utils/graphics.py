import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import subprocess
import h5py
import os
import moviepy.editor as mp
import cv2
import matplotlib.gridspec as gridspec


def plot_spatial_variables(mesh, spatial_variable, variable_name, steps, colormap, value_range, output_dir):
	"""
	Function to generate images of the spatial profiles of different cellVariables()
	"""
	
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
	# fig.savefig(fname=output_dir + 'Images/' + variable_name + '_{step}.svg'.format(step=steps),dpi=600,format='svg')
	# pkl.dump((fig,ax),file(output_dir + 'Images/' + variable_name +'_{step}.pickle'.format(step=steps),'w'))
	plt.close()


def write_movie_from_hdf5(PATH, names, mesh, fps=3):
	"""
	Function to write movies from hdf5 file
	"""

	def key_funct(x):
		return int(x.split('_')[-1].rstrip('.png'))

	# make directory
	try:
		os.mkdir(os.path.join(PATH, 'movies'))
	except:
		print("/movies directory already exists")
	
	with h5py.File(os.path.join(PATH, "spatial_variables.hdf5"), mode="r") as df_total:
	
		for name in names:
			
			max_val = df_total[name][:].max()
			min_val = df_total[name][:].min()

			if min_val < 0.0 and np.abs(min_val/max_val) > 0.3:
				color_map = 'coolwarm'
			else:
				color_map = 'Reds'

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
										cmap=color_map)
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



def write_movie_from_hdf5_combined(PATH, names, mesh, mRNA_flag, fps=3):
	""" 
	Function to combine all the spatial variables time lapse videos into a single movie, 
	so that we can visualize all of them side-by-side
	"""

	def key_funct(x):
		return int(x.split('_')[-1].rstrip('.png'))

	# make directory
	try:
		os.mkdir(os.path.join(PATH, 'movies'))
	except:
		print("/movies directory already exists")
	
	with h5py.File(os.path.join(PATH, "spatial_variables.hdf5"), mode="r") as df_total:
	
		num_rows = 1 #int(np.ceil(len(names)/float(num_cols)))
		
		if mRNA_flag:
			num_cols = 3
			names = ['phi_p','phi_r','phi_m']
		else:
			num_cols = 2
			names = ['phi_p','phi_r']
		# print(names)
		# print(num_rows)

		for i in range(df_total[names[0]].shape[0]):

			if np.all(df_total[names[0]][i] == 0):
				break

			fig = plt.figure(figsize = (5*num_cols,4*num_rows))
			gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols, figure=fig)
			counter = 0

			for name in names:
				
				max_val = df_total[name][:].max()
				min_val = df_total[name][:].min()

				# if len(df_total[name][:].shape) == 2:
				#     df_ = [df_total[name][:]]
				# elif len(df_total[name][:].shape) == 3:
				#     df_ = [df_total[name][:,0,:], df_total[name][:,1,:]]
				
				
				print(int(counter/num_cols))
				print(int(counter%num_cols))
				ax = fig.add_subplot(gs[int(counter/num_cols),int(counter%num_cols)])
				
				cs = ax.tricontourf(mesh.x.value,
									mesh.y.value,
									df_total[name][i],
									levels=np.linspace(min_val,max_val,64),
									cmap='Reds')
				
				fig.colorbar(cs, ax=ax, ticks=np.linspace(min_val,max_val,11))
				ax.set_title(name)   
				counter = counter + 1

			fig.savefig(fname=PATH +'/movies/combined_movie_step_{step}.png'.format(step=i),dpi=600,format='png')
			plt.close('all')  

	file_names = sorted(list((fn for fn in os.listdir(os.path.join(PATH, 'movies')) if fn.endswith('.png'))), key=key_funct)
	file_paths = [os.path.join(PATH, 'movies', f) for f in file_names]
	
	clip = mp.ImageSequenceClip(file_paths, fps = fps)
	clip.write_videofile(os.path.join(PATH, 'movies','combined_movie.mp4'), fps=fps)
	clip.close()

	# delete individual images
	for f in file_paths:
		os.remove(f)


