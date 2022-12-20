import numpy as np
import fipy as fp
from fipy import Gmsh2D
import pandas as pd
from matplotlib import pyplot as plt
import os
import re
import pdb
import h5py

# Simulations done
# - No lncRNA and No activity
# - Only lncRNA and No activity
# - Only activity and No lncRNA
# - Both lncRNA and activity

## Plots to generate

### Dynamics

# - Condensate radius vs. time
# - Average protein concentration in condensate vs. time
# - Average lncRNA concentration in condensate vs. time
# - molar rate of mRNA production rate in system (k\*phi_p\*volume)
# - molar amounts of mRNA in the system vs. time

### Steady state

# - Phase diagrams for qualitative steady state behaviors as a function of parameters
# - Phase diagrams for qualitative behaviors of dynamics as a function of parameters
# - Phase diagrams for the steady state values of protein concentration, lncRNA concentration, and mRNA amounts 

# Function to create a circular mesh using gmsh:

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


### Define the mesh used for simulations

nx = 300
dx = 0.2
mesh = create_circular_mesh(radius=float(nx)*dx/2,cellSize=dx*1.5)

### Generate plots for pure protein simulations

pure_protein_directory = '/nobackup1c/users/npradeep96/PhaseField/PURE_PROTEIN/'
K_list = [0.2,0.3,0.4,0.8]
LR_list = [5,10,15]

phi_ss = np.zeros([len(LR_list),len(K_list)]) 
r_ss = np.zeros([len(LR_list),len(K_list)]) 

condensate_concentration_cutoff = 0.4

for i in range(len(LR_list)):

    LR = LR_list[i]

    fig, axs = plt.subplots(1,2,figsize=(14,6))

    for j in range(len(K_list)):

        K = K_list[j]

        for root, dirs, files in os.walk(pure_protein_directory):

            regex = re.compile(r'.*K_{}_.*_LR_{}_kappa_0.1$'.format(str(K), str(LR)))
            match = re.search(regex, root)

            if match != None:

                stats_file_name = root + '/stats.txt'
                df_stats = pd.read_csv(stats_file_name, '\t')
                times = df_stats['t']
                steps = df_stats['step']
                phi_p_max = df_stats['phi_p_max']
                average_protein_in_condensate = np.zeros(len(steps))
                condensate_radius = np.zeros(len(steps))

                for s in range(len(steps)):
    
                    hdf5_file_name = root + '/spatial_variables.hdf5'.format(steps[s])
        
                    with h5py.File(hdf5_file_name) as f:

                        phi_p = fp.CellVariable(mesh=mesh, value = f['phi_p'][s])

                    indices = phi_p.value > condensate_concentration_cutoff                    

                    if np.any(indices):
                        average_protein_in_condensate[s] = np.sum(phi_p.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                        condensate_radius[s] = (np.sum(mesh.cellVolumes[indices])/np.pi)**0.5
                    else:
                        average_protein_in_condensate[s] = 0.0
                        condensate_radius[s] = 0.0

                phi_ss[i,j] = average_protein_in_condensate[-1]
                r_ss[i,j] = condensate_radius[-1]

                axs[0].plot(times, average_protein_in_condensate, label='c_max = {}'.format(K), marker='o')
                axs[1].plot(times, condensate_radius, label='c_max = {}'.format(K), marker='o')
   

    axs[0].set_xlabel('time')
    axs[1].set_xlabel('time')
    axs[0].set_ylabel(r'$<\phi_p>$ in condensate')
    axs[1].set_ylabel('Condensate radius')
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylim([0.0,0.7])
    axs[1].set_ylim([0.0,10.0])

    plt.savefig('radius_phi_dynamics_plots_{}.png'.format(LR),dpi=600,format='png')
    plt.close()

##### Bar plots of steady state radii and concentrations:

barWidth = 0.25
fig,axs = plt.subplots(1,2,figsize =(14, 6))

print(r_ss,phi_ss)
 
# Set position of bar on X axis
br1 = np.arange(r_ss.shape[1])
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot

axs[0].bar(br1, phi_ss[0,:], color ='silver', width = barWidth,
        edgecolor ='grey', label = r'$L_R$=5')
axs[0].bar(br2, phi_ss[1,:], color ='gray', width = barWidth,
        edgecolor ='grey', label = r'$L_R$=10')
axs[0].bar(br3, phi_ss[2,:], color ='black', width = barWidth,
        edgecolor ='grey', label = r'$L_R$=15')
 
# Adding Xticks
axs[0].set_xlabel('c_max', fontweight ='bold', fontsize = 15)
axs[0].set_ylabel('Steady state $<\phi_p>$ in condensate', fontweight ='bold', fontsize = 15)
axs[0].set_xticks([r + barWidth for r in br1])
axs[0].set_xticklabels(['0.2', '0.3', '0.4', '0.8'])

axs[1].bar(br1, r_ss[0,:], color ='silver', width = barWidth,
        edgecolor ='grey', label = r'$L_R$=5')
axs[1].bar(br2, r_ss[1,:], color ='gray', width = barWidth,
        edgecolor ='grey', label = r'$L_R$=10')
axs[1].bar(br3, r_ss[2,:], color ='black', width = barWidth,
        edgecolor ='grey', label = r'$L_R$=15')
 
# Adding Xticks
axs[1].set_xlabel('c_max', fontweight ='bold', fontsize = 15)
axs[1].set_ylabel('Steady state condensate radius', fontweight ='bold', fontsize = 15)
axs[1].set_xticks([r + barWidth for r in br1])
axs[1].set_xticklabels(['0.2', '0.3', '0.4', '0.8'])
 
axs[0].legend()
axs[1].legend()
plt.savefig('radius_phi_ss_plots.png',dpi=600,format='png')
plt.show()
plt.close()


#         axs.set_xlabel(r'Initial peak concentration of RNA at the well ($\phi^{peak}_R$)',fontsize=15)
#         axs.set_ylabel(r'Average concentration of RNA in condensate',fontsize=15)
#         axs.legend(fontsize=15)
#         axs.set_ylim([0,0.05])

#         if K == '1':
#             axs.set_title('Shallow Well', fontsize = 20)
#             fig.savefig(fname='Shallow Well.png',dpi=600,format='png')
#         else:
#             axs.set_title('Deep Well', fontsize = 20)
#             fig.savefig(fname='Deep Well.png',dpi=600,format='png')           

                # print(K,L,conc,average_protein_in_condensate, average_rna_in_condensate)