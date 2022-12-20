import argparse
import re
import pandas as pd
import h5py
import os
import moviepy.editor as mp
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def write_movies(PATH, hdf5file_name, x, y, fps=5):
	
	def key_funct(x):
		return int(x.split('_')[-1].rstrip('.png'))

	# make directory
	try:
		os.mkdir(os.path.join(PATH, 'movies'))
	except:
		print("/movies directory already exists")
		
	df_total = h5py.File(os.path.join(PATH, hdf5file_name), mode="r")
	
	for name in df_total.keys(): # Iterate through all the datasets in the hdf5 file
		
		# find value to plot for max/min
		max_val = df_total[name][:].max()
		min_val = df_total[name][:].min()

		if len(df_total[name][:].shape) == 2:
			df_ = [df_total[name][:]]
		elif len(df_total[name][:].shape) == 3:
			df_ = [df_total[name][:,0,:], df_total[name][:,1,:]]
		
		for idx, df in enumerate(df_):
			for i in range(df.shape[0]):
				# plot and save individuals
				fig, ax = plt.subplots()
				cs = ax.tricontourf(x, y, df[i], levels=np.linspace(min_val,max_val,256), cmap='coolwarm')

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

if __name__ == "__main__":
	"""
		Function is called when python code is run on command line and calls the function that generates the
		movies from hdf5 files
		
		This script assumes that each directory containing a hdf5 file also contains another subdirectory called 
		/Mesh that contains the values of mesh coordinates
	"""
	parser = argparse.ArgumentParser(description='Take directory name to walk through and generate movies')
	parser.add_argument('--i',help="Root directory containing the hdf5 files", required = True)
	parser.add_argument('--f',help="Name of hdf5 file", required = True)
	parser.add_argument('--mf',help="Path to mesh file", required = True)
	# parser.add_argument('--pN',help="Parameter number from file (indexed from 1)", required = False);

	# parser.add_argument('--o',help="Name of output folder", required = True);
	args = parser.parse_args();

	base_path = args.i
	regex = re.compile(r'.*'+str(args.f))
	found_at_least_one = 0

	for root, dirs, files in os.walk(base_path): # Loop through all subdirectories in the directory "base_path"

		for fi in files: # Loop through all files in each subdirectory to see if we have a hdf5 file with the name given in args.f
			
			match = re.search(regex, fi)

			if match != None: 		# Found a hdf5 file!

				found_at_least_one = 1
				
				# Read mesh coordinates:
				df = pd.read_csv(root + '/' + args.mf , '\t')
				x = df.values[:,0]
				y = df.values[:,1]
				write_movies(root, args.f, x, y, fps = 3)

	if not found_at_least_one:
		print('Could not find any hdf5 files in the supplied directory!')

