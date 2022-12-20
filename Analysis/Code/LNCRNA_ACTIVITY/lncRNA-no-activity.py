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

pure_protein_directory = '/nobackup1c/users/npradeep96/PhaseField/LNCRNA_AND_ACTIVITY/LNCRNA_RESCUES_CONDENSATE/'
phi_list = [5.0,10.0,15.0,20.0,25.0]
LR_list = [3.0,4.0,5.0,6.0,7.0,10.0,12.0]

phi_p_ss = np.zeros([len(phi_list)+1,len(LR_list)]) 
phi_r_ss = np.zeros([len(phi_list)+1,len(LR_list)]) 
r_ss = np.zeros([len(phi_list)+1,len(LR_list)]) 

condensate_concentration_cutoff = 0.4

for i in range(len(phi_list)):

    phi = phi_list[i]

    fig, axs = plt.subplots(1,3,figsize=(20,6))

    for j in range(len(LR_list)):

        LR = LR_list[j]

        for root, dirs, files in os.walk(pure_protein_directory):

            regex = re.compile(r'.*phi_r_{}_L_P_{}_K_1.0$'.format(phi, LR))
            match = re.search(regex, root)

            if match != None:

                stats_file_name = root + '/stats.txt'
                df_stats = pd.read_csv(stats_file_name, '\t')
                times = df_stats['t']
                steps = df_stats['step']
                phi_p_max = df_stats['phi_p_max']

                average_protein_in_condensate = np.zeros(len(steps))
                average_rna_in_condensate = np.zeros(len(steps))
                condensate_radius = np.zeros(len(steps))

                for s in range(len(steps)):
    
                    hdf5_file_name = root + '/spatial_variables.hdf5'.format(steps[s])
        
                    with h5py.File(hdf5_file_name) as f:

                        phi_p = fp.CellVariable(mesh=mesh, value = f['phi_p'][s])
                        phi_r = fp.CellVariable(mesh=mesh, value = f['phi_r'][s])

                    indices = phi_p.value > condensate_concentration_cutoff                    

                    if np.any(indices):
                        average_protein_in_condensate[s] = np.sum(phi_p.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                        average_rna_in_condensate[s] = np.sum(phi_r.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                        condensate_radius[s] = (np.sum(mesh.cellVolumes[indices])/np.pi)**0.5
                    else:
                        average_protein_in_condensate[s] = 0.0
                        average_rna_in_condensate[s] = 0.0
                        condensate_radius[s] = 0.0

                phi_p_ss[i,j] = average_protein_in_condensate[-1]
                phi_r_ss[i,j] = average_rna_in_condensate[-1]
                r_ss[i,j] = condensate_radius[-1]

                axs[0].plot(times, average_protein_in_condensate, label=r'$L_R$ = {}'.format(LR), marker='o')
                axs[1].plot(times, average_rna_in_condensate, label=r'$L_R$ = {}'.format(LR), marker='o')
                axs[2].plot(times, condensate_radius, label=r'$L_R$ = {}'.format(LR), marker='o')
   

    axs[0].set_xlabel('time')
    axs[1].set_xlabel('time')
    axs[2].set_xlabel('time')

    axs[0].set_ylabel(r'$<\phi_p>$ in condensate')
    axs[1].set_ylabel(r'$<\phi_r>$ in condensate')
    axs[2].set_ylabel('Condensate radius')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].set_ylim([0.0,0.7])
    axs[1].set_ylim([0.0,0.1])


    plt.savefig('radius_phi_dynamics_plots_{}.png'.format(phi),dpi=600,format='png')
    plt.close()

##### Bar plots of steady state radii and concentrations:

barWidth = 0.15
fig,axs = plt.subplots(1,3,figsize =(20, 6))
 
# Set position of bar on X axis
br1 = np.arange(r_ss.shape[1])
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]

axs[0].bar(br1, phi_p_ss[5,:], color ='white', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=0')
axs[0].bar(br2, phi_p_ss[0,:], color ='gainsboro', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=5')
axs[0].bar(br3, phi_p_ss[1,:], color ='silver', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=10')
axs[0].bar(br4, phi_p_ss[2,:], color ='darkgray', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=15')
axs[0].bar(br5, phi_p_ss[3,:], color ='dimgray', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=20')
axs[0].bar(br6, phi_p_ss[4,:], color ='black', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=25')

axs[0].set_xlabel('L', fontweight ='bold', fontsize = 15)
axs[0].set_ylabel('Steady state $<\phi_p>$ in condensate', fontweight ='bold', fontsize = 15)
axs[0].set_xticks([r + barWidth for r in br1])
axs[0].set_xticklabels(['3.0', '4.0', '5.0', '6.0','7.0','10.0','12.0'])

axs[1].bar(br1, phi_r_ss[5,:], color ='white', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=0')
axs[1].bar(br2, phi_r_ss[0,:], color ='gainsboro', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=5')
axs[1].bar(br3, phi_r_ss[1,:], color ='silver', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=10')
axs[1].bar(br4, phi_r_ss[2,:], color ='darkgray', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=15')
axs[1].bar(br5, phi_r_ss[3,:], color ='dimgray', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=20')
axs[1].bar(br6, phi_r_ss[4,:], color ='black', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=25')

axs[1].set_xlabel('L', fontweight ='bold', fontsize = 15)
axs[1].set_ylabel('Steady state $<\phi_r>$ in condensate', fontweight ='bold', fontsize = 15)
axs[1].set_xticks([r + barWidth for r in br1])
axs[1].set_xticklabels(['3.0', '4.0', '5.0', '6.0','7.0','10.0','12.0'])

axs[2].bar(br1, r_ss[5,:], color ='white', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=0')
axs[2].bar(br2, r_ss[0,:], color ='gainsboro', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=5')
axs[2].bar(br3, r_ss[1,:], color ='silver', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=10')
axs[2].bar(br4, r_ss[2,:], color ='darkgray', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=15')
axs[2].bar(br5, r_ss[3,:], color ='dimgray', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=20')
axs[2].bar(br6, r_ss[4,:], color ='black', width = barWidth,
        edgecolor ='grey', label = r'$\int \phi_R dV$=25')

axs[2].set_xlabel('L', fontweight ='bold', fontsize = 15)
axs[2].set_ylabel('Steady state condensate radius', fontweight ='bold', fontsize = 15)
axs[2].set_xticks([r + 2*barWidth for r in br1])
axs[2].set_xticklabels(['3.0', '4.0', '5.0', '6.0','7.0','10.0','12.0'])

axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.savefig('lp_radius_phi_ss_plots.png',dpi=600,format='png')
plt.show()
plt.close()

#### Line plots of steady state properties of the condensate vs. L, for different values of total lncRNA

fig,axs = plt.subplots(1,3,figsize =(20, 6))

axs[0].plot(LR_list, phi_p_ss[5,:], color ='sienna', label = r'$\int \phi_R dV$=0.0', marker='o')
axs[0].plot(LR_list, phi_p_ss[0,:], color ='gainsboro', label = r'$\int \phi_R dV$=5.0', marker='o')
axs[0].plot(LR_list, phi_p_ss[1,:], color ='silver', label = r'$\int \phi_R dV$=10.0', marker='o')
axs[0].plot(LR_list, phi_p_ss[2,:], color ='darkgray', label = r'$\int \phi_R dV$=15.0', marker='o')
axs[0].plot(LR_list, phi_p_ss[3,:], color ='dimgray', label = r'$\int \phi_R dV$=20.0', marker='o')
axs[0].plot(LR_list, phi_p_ss[4,:], color ='black', label = r'$\int \phi_R dV$=25.0', marker='o')
axs[0].set_xlabel('L', fontweight ='bold', fontsize = 15)
axs[0].set_ylabel('Steady state $<\phi_p>$ in condensate', fontweight ='bold', fontsize = 15)

axs[1].plot(LR_list, phi_r_ss[5,:], color ='sienna', label = r'$\int \phi_R dV$=0.0', marker='o')
axs[1].plot(LR_list, phi_r_ss[0,:], color ='gainsboro', label = r'$\int \phi_R dV$=5.0', marker='o')
axs[1].plot(LR_list, phi_r_ss[1,:], color ='silver', label = r'$\int \phi_R dV$=10.0', marker='o')
axs[1].plot(LR_list, phi_r_ss[2,:], color ='darkgray', label = r'$\int \phi_R dV$=15.0', marker='o')
axs[1].plot(LR_list, phi_r_ss[3,:], color ='dimgray', label = r'$\int \phi_R dV$=20.0', marker='o')
axs[1].plot(LR_list, phi_r_ss[4,:], color ='black', label = r'$\int \phi_R dV$=25.0', marker='o')
axs[1].set_xlabel('L', fontweight ='bold', fontsize = 15)
axs[1].set_ylabel('Steady state $<\phi_r>$ in condensate', fontweight ='bold', fontsize = 15)

axs[2].plot(LR_list, r_ss[5,:], color ='sienna', label = r'$\int \phi_R dV$=0.0', marker='o')
axs[2].plot(LR_list, r_ss[0,:], color ='gainsboro', label = r'$\int \phi_R dV$=5.0', marker='o')
axs[2].plot(LR_list, r_ss[1,:], color ='silver', label = r'$\int \phi_R dV$=10.0', marker='o')
axs[2].plot(LR_list, r_ss[2,:], color ='darkgray', label = r'$\int \phi_R dV$=15.0', marker='o')
axs[2].plot(LR_list, r_ss[3,:], color ='dimgray', label = r'$\int \phi_R dV$=20.0', marker='o')
axs[2].plot(LR_list, r_ss[4,:], color ='black', label = r'$\int \phi_R dV$=25.0', marker='o')
axs[2].set_xlabel('L', fontweight ='bold', fontsize = 15)
axs[2].set_ylabel('Steady state condensate radius', fontweight ='bold', fontsize = 15)

axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.savefig('lp_radius_phi_ss__vs_L_plots_lines.png',dpi=600,format='png')
plt.show()
plt.close()

#### Line plots of steady state properties of the condensate vs. total lncRNA, for different values of L

fig,axs = plt.subplots(1,3,figsize =(20, 6))

axs[0].plot([0.0] + phi_list, list([phi_p_ss[-1,0]]) + list(phi_p_ss[:-1,0]), color ='sienna', label = r'$L$=3.0', marker='o')
axs[0].plot([0.0] + phi_list, list([phi_p_ss[-1,1]]) + list(phi_p_ss[:-1,1]), color ='rosybrown', label = r'$L$=4.0' , marker='o')
axs[0].plot([0.0] + phi_list, list([phi_p_ss[-1,2]]) + list(phi_p_ss[:-1,2]), color ='gainsboro', label = r'$L$=5.0', marker='o')
axs[0].plot([0.0] + phi_list, list([phi_p_ss[-1,3]]) + list(phi_p_ss[:-1,3]), color ='silver', label = r'$L$=6.0', marker='o')
axs[0].plot([0.0] + phi_list, list([phi_p_ss[-1,4]]) + list(phi_p_ss[:-1,4]), color ='darkgray', label = r'$L$=7.0', marker='o')
axs[0].plot([0.0] + phi_list, list([phi_p_ss[-1,5]]) + list(phi_p_ss[:-1,5]), color ='dimgray', label = r'$L$=10.0', marker='o')
axs[0].plot([0.0] + phi_list, list([phi_p_ss[-1,6]]) + list(phi_p_ss[:-1,6]), color ='black', label = r'$L$=12.0', marker='o')
axs[0].set_xlabel(r'$\int \phi_R dV$', fontweight ='bold', fontsize = 15)
axs[0].set_ylabel('Steady state $<\phi_p>$ in condensate', fontweight ='bold', fontsize = 15)

axs[1].plot([0.0] + phi_list, list([phi_r_ss[-1,0]]) + list(phi_r_ss[:-1,0]), color ='sienna', label = r'$L$=3.0', marker='o')
axs[1].plot([0.0] + phi_list, list([phi_r_ss[-1,1]]) + list(phi_r_ss[:-1,1]), color ='rosybrown', label = r'$L$=4.0' , marker='o')
axs[1].plot([0.0] + phi_list, list([phi_r_ss[-1,2]]) + list(phi_r_ss[:-1,2]), color ='gainsboro', label = r'$L$=5.0', marker='o')
axs[1].plot([0.0] + phi_list, list([phi_r_ss[-1,3]]) + list(phi_r_ss[:-1,3]), color ='silver', label = r'$L$=6.0', marker='o')
axs[1].plot([0.0] + phi_list, list([phi_r_ss[-1,4]]) + list(phi_r_ss[:-1,4]), color ='darkgray', label = r'$L$=7.0', marker='o')
axs[1].plot([0.0] + phi_list, list([phi_r_ss[-1,5]]) + list(phi_r_ss[:-1,5]), color ='dimgray', label = r'$L$=10.0', marker='o')
axs[1].plot([0.0] + phi_list, list([phi_r_ss[-1,6]]) + list(phi_r_ss[:-1,6]), color ='black', label = r'$L$=12.0', marker='o')
axs[1].set_xlabel(r'$\int \phi_R dV$', fontweight ='bold', fontsize = 15)
axs[1].set_ylabel('Steady state $<\phi_r>$ in condensate', fontweight ='bold', fontsize = 15)

axs[2].plot([0.0] + phi_list, list([r_ss[-1,0]]) + list(r_ss[:-1,0]), color ='sienna', label = r'$L$=3.0', marker='o')
axs[2].plot([0.0] + phi_list, list([r_ss[-1,1]]) + list(r_ss[:-1,1]), color ='rosybrown', label = r'$L$=4.0' , marker='o')
axs[2].plot([0.0] + phi_list, list([r_ss[-1,2]]) + list(r_ss[:-1,2]), color ='gainsboro', label = r'$L$=5.0', marker='o')
axs[2].plot([0.0] + phi_list, list([r_ss[-1,3]]) + list(r_ss[:-1,3]), color ='silver', label = r'$L$=6.0', marker='o')
axs[2].plot([0.0] + phi_list, list([r_ss[-1,4]]) + list(r_ss[:-1,4]), color ='darkgray', label = r'$L$=7.0', marker='o')
axs[2].plot([0.0] + phi_list, list([r_ss[-1,5]]) + list(r_ss[:-1,5]), color ='dimgray', label = r'$L$=10.0', marker='o')
axs[2].plot([0.0] + phi_list, list([r_ss[-1,6]]) + list(r_ss[:-1,6]), color ='black', label = r'$L$=12.0', marker='o')
axs[2].set_xlabel(r'$\int \phi_R dV$', fontweight ='bold', fontsize = 15)
axs[2].set_ylabel('Steady state condensate radius', fontweight ='bold', fontsize = 15)

axs[0].legend()
axs[1].legend()
axs[2].legend()

plt.savefig('lp_radius_phi_ss_vs_phi_plots_lines.png',dpi=600,format='png')
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