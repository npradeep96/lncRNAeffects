import fipy as fp
import numpy as np
import scipy.optimize as spo
import scipy.special as sc

class free_energy:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, rho_s, c_alpha, c_beta, chi, c, kappa, K, r0, L_R, a=0.0, b=0.0):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.chi = chi
		self.c = c
		self.kappa = kappa
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.a = a
		self.b = b
	
	def f(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 - self.chi*phi_p*phi_r + 
			   self.a*phi_r**2*phi_p + self.b*phi_r*phi_p**2 + self.c*phi_r**2*phi_p**2 + 
			   0.5*self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2 + 1.0)*phi_r**2 + 
			   0.5*self.kappa*(phi_p.grad.mag)**2)
	
	def mu_p(self, phi_p, phi_r):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha + self.c_beta) - self.chi*phi_r +
				self.a*phi_r*phi_r  + 2*self.b*phi_p*phi_r + 2*self.c*phi_r*phi_r*phi_p + self.kappa*(phi_p.faceGrad.divergence))
	
	def mu_r(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		return (-self.chi*phi_p + 2*self.a*phi_p*phi_r  + self.b*phi_p*phi_p + 2*self.c*phi_r*phi_p*phi_p + 
				self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2 + 1.0)*phi_r)
		
		
	def dmu_p_dphi_p(self, phi_p, phi_r, X_CV, Y_CV):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + 2*self.b*phi_r + 2*self.c*phi_r*phi_r)

	def dmu_p_dphi_r(self, phi_p, phi_r):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*phi_r + 2*self.b*phi_p + 4*self.c*phi_p*phi_r)  
	
	def dmu_r_dphi_r(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2.0 + 1.0) + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)
	
	def parabolic_multiplier(self, X_CV, Y_CV):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return (self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2.0 + 1.0))
	
	def det_J(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
		"""
		return (self.dmu_p_dphi_p(phi_p,phi_r,X_CV,Y_CV)*self.dmu_r_dphi_r(phi_p,phi_r,X_CV,Y_CV)-(self.dmu_p_dphi_r(phi_p,phi_r))**2)

	def mu_r_multiplier(self, mesh):
		return(fp.CellVariable(mesh = mesh, name = '$M_R K((r-r_0)^2/L^2_R+1)$',value=(self.K/self.L_R**2)*((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)))

	def mu_p_multiplier(self, mesh):
		return fp.CellVariable(mesh=mesh, value = 1.0)

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_phi):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
		Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

		C_R = rna_nucleus_phi*self.K
		
		phi_r.value = C_R*(self.parabolic_multiplier(X_CV, Y_CV).value)**(-1)


class free_energy_protein_localized:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, rho_s, c_alpha, c_beta, chi, c, kappa, K, r0, L_R, c_max, sigma, rE, a=0.0, b=0.0):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.chi = chi
		self.c = c
		self.kappa = kappa
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
		self.a = a
		self.b = b
	
	def f(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 - self.chi*phi_p*phi_r + 
			   self.a*phi_r**2*phi_p + self.b*phi_r*phi_p**2 + self.c*phi_r**2*phi_p**2 + 
			   0.5*self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2 + 1.0)*phi_r**2 + 
			   0.5*self.kappa*(phi_p.grad.mag)**2 - 
			   self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p**2)
	
	def mu_p(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha + self.c_beta) - self.chi*phi_r +
				self.a*phi_r*phi_r  + 2*self.b*phi_p*phi_r + 2*self.c*phi_r*phi_r*phi_p + self.kappa*(phi_p.faceGrad.divergence) - 
				2.0*self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p)
	
	def mu_r(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		return (-self.chi*phi_p + 2*self.a*phi_p*phi_r  + self.b*phi_p*phi_p + 2*self.c*phi_r*phi_p*phi_p + 
				self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2 + 1.0)*phi_r)
		
		
	def dmu_p_dphi_p(self, phi_p, phi_r, X_CV, Y_CV):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + 2*self.b*phi_r + 2*self.c*phi_r*phi_r - 2.0*self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2))

	def dmu_p_dphi_r(self, phi_p, phi_r):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*phi_r + 2*self.b*phi_p + 4*self.c*phi_p*phi_r)  
	
	def dmu_r_dphi_r(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2.0 + 1.0) + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)
	
	def parabolic_multiplier(self, X_CV, Y_CV):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return (self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2.0 + 1.0))
	
	def det_J(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
		"""
		return (self.dmu_p_dphi_p(phi_p,phi_r,X_CV,Y_CV)*self.dmu_r_dphi_r(phi_p,phi_r,X_CV,Y_CV)-(self.dmu_p_dphi_r(phi_p,phi_r))**2)

	def mu_r_multiplier(self, mesh):
		return(fp.CellVariable(mesh = mesh, name = '$M_R K((r-r_0)^2/L^2_R+1)$',value=(self.K/self.L_R**2)*((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)))

	def mu_p_multiplier(self, mesh):
		return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_phi):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
		Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

		C_R = rna_nucleus_phi*self.K
		
		phi_r.value = C_R*(self.parabolic_multiplier(X_CV, Y_CV).value)**(-1)


class free_energy_gaussian_gaussian:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, rho_s, c_alpha, c_beta, chi, c, kappa, rho_r, K, r0, L_R, c_max, sigma, rE, a=0.0, b=0.0):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.chi = chi
		self.c = c
		self.kappa = kappa
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.rho_r = rho_r
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
		self.a = a
		self.b = b
	
	def f(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 - self.chi*phi_p*phi_r + 
			   self.a*phi_r**2*phi_p + self.b*phi_r*phi_p**2 + self.c*phi_r**2*phi_p**2 + 
			   self.rho_r*(1.0 - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))*phi_r**2 + 
			   0.5*self.kappa*(phi_p.grad.mag)**2 - 
			   self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p**2)
	
	def mu_p(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha + self.c_beta) - self.chi*phi_r +
				self.a*phi_r*phi_r  + 2*self.b*phi_p*phi_r + 2*self.c*phi_r*phi_r*phi_p + self.kappa*(phi_p.faceGrad.divergence) - 
				2.0*self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p)
	
	def mu_r(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		return (-self.chi*phi_p + 2*self.a*phi_p*phi_r  + self.b*phi_p*phi_p + 2*self.c*phi_r*phi_p*phi_p + 
				2.0*self.rho_r*(1.0 - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))*phi_r)
		
		
	def dmu_p_dphi_p(self, phi_p, phi_r, X_CV, Y_CV):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + 2*self.b*phi_r + 2*self.c*phi_r*phi_r - 2.0*self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2))

	def dmu_p_dphi_r(self, phi_p, phi_r):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*phi_r + 2*self.b*phi_p + 4*self.c*phi_p*phi_r)  
	
	def dmu_r_dphi_r(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (2.0*self.rho_r*(1.0 - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)) + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)
	
	def gaussian_multiplier(self, X_CV, Y_CV):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return (2.0*self.rho_r*(1.0 - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)))
	
	def det_J(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
		"""
		return (self.dmu_p_dphi_p(phi_p,phi_r,X_CV,Y_CV)*self.dmu_r_dphi_r(phi_p,phi_r,X_CV,Y_CV)-(self.dmu_p_dphi_r(phi_p,phi_r))**2)

	def mu_r_multiplier(self, mesh):
		return(fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_RNA',value=2.0*self.rho_r*(1.0 - self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2))))

	def mu_p_multiplier(self, mesh):
		return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_Protein',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_phi_total):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		
		X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
		Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

		gaussian_multiplier_inverse = (self.gaussian_multiplier(X_CV, Y_CV))**(-1)
		A = rna_nucleus_phi_total/(gaussian_multiplier_inverse.cellVolumeAverage.value)

		phi_r.value = A*(gaussian_multiplier_inverse)


class free_energy_protein_parabolic_lncrna_gaussian_mRNA:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, rho_s, c_alpha, c_beta, chi, c, kappa, rho_m, K, r0, L_R, c_max, sigma, rE, a=0.0, b=0.0):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.chi = chi
		self.c = c
		self.kappa = kappa
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
		self.a = a
		self.b = b
		self.rho_r = rho_m
	
	def f(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""

		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 - self.chi*phi_p*phi_r + 
			   self.a*phi_r**2*phi_p + self.b*phi_r*phi_p**2 + self.c*phi_r**2*phi_p**2 + 
			   0.5*self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2 + 1.0)*phi_r**2 + 
			   0.5*self.kappa*(phi_p.grad.mag)**2 - 
			   self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p**2
			   -self.chi*phi_m*phi_p + self.c*phi_m**2*phi_p**2 + self.rho_r*phi_m**2)
	
	def mu_p(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""

		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha + self.c_beta) - self.chi*phi_r +
				self.a*phi_r*phi_r  + 2*self.b*phi_p*phi_r + 2*self.c*phi_r*phi_r*phi_p + self.kappa*(phi_p.faceGrad.divergence) - 
				2.0*self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p
				-self.chi*phi_m + 2.0*self.c*phi_m**2*phi_p)
	
	def mu_r(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""

		return (-self.chi*phi_p + 2*self.a*phi_p*phi_r  + self.b*phi_p*phi_p + 2*self.c*phi_r*phi_p*phi_p + 
				self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2 + 1.0)*phi_r)

	def mu_m(self, phi_p, phi_r, phi_m):

		return (-self.chi*phi_p +  2.0*self.c*phi_m*phi_p**2 + 2.0*self.rho_r*phi_m)
		
		
	def dmu_p_dphi_p(self, phi_p, phi_r, phi_m, X_CV, Y_CV):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""

		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + 2*self.b*phi_r + 2*self.c*phi_r*phi_r - 2.0*self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2) + 2.0*self.c*phi_m**2)

	def dmu_p_dphi_r(self, phi_p, phi_r):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*phi_r + 2*self.b*phi_p + 4*self.c*phi_p*phi_r)  

	def dmu_p_dphi_m(self, phi_p, phi_m):

		return (-self.chi + 4*self.c*phi_p*phi_m)
	
	def dmu_r_dphi_r(self, phi_p, phi_r, X_CV, Y_CV):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2.0 + 1.0) + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)

	def dmu_m_dphi_m(self, phi_p, phi_m):

		return (2.0*phi_p**2 + 2.0*self.rho_r)
	
	def parabolic_multiplier(self, X_CV, Y_CV):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return (self.K*(((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2.0 + 1.0))
	
	def det_J(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
		"""

		return (self.dmu_p_dphi_p(phi_p,phi_r,phi_m,X_CV,Y_CV)*self.dmu_r_dphi_r(phi_p,phi_r,X_CV,Y_CV)*self.dmu_m_dphi_m(phi_p, phi_m) - 
			   (self.dmu_p_dphi_r(phi_p,phi_r))**2*self.dmu_m_dphi_m(phi_p, phi_m) - (self.dmu_p_dphi_m(phi_p,phi_m))**2*self.dmu_m_dphi_m(phi_p, phi_m))

	def mu_r_multiplier(self, mesh):
		return(fp.CellVariable(mesh = mesh, name = '$M_R K((r-r_0)^2/L^2_R+1)$',value=(self.K/self.L_R**2)*((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)))

	def mu_p_multiplier(self, mesh):
		return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_phi):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
		Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

		C_R = rna_nucleus_phi*self.K
		
		phi_r.value = C_R*(self.parabolic_multiplier(X_CV, Y_CV).value)**(-1)


class free_energy_protein_nuc_RNA_gaussian_mRNA:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, rho_s, c_alpha, c_beta, chi, c, kappa, rho_r, rho_m, rho_c, K, r0, L_R, c_max, sigma, rE, a=0.0, b=0.0):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.chi = chi
		self.c = c
		self.kappa = kappa
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.rho_r = rho_r
		self.rho_m = rho_m
		self.rho_c = rho_c
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
		self.a = a
		self.b = b
	
	def f(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 + 0.5*self.kappa*(phi_p.grad.mag)**2 - 
			   self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p - 
			   self.chi*phi_p*(phi_m + phi_r) + self.a*(phi_r + phi_m)**2*phi_p + self.b*(phi_r + phi_m)*phi_p**2 + self.c*(phi_r + phi_m)**2*phi_p**2 + 
			   self.rho_r*phi_r**2 + self.rho_m*phi_m**2 + 2*self.rho_c*phi_r*phi_m - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)*phi_r)
	
	def mu_p(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha - self.c_beta) - self.kappa*(phi_p.faceGrad.divergence) - 
				self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2) - 
				self.chi*(phi_r + phi_m) + self.a*(phi_r + phi_m)**2 + 2*self.b*(phi_r + phi_m)*phi_p + 2*self.c*(phi_r + phi_m)**2*phi_p)
	
	def mu_r(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		return (-self.chi*phi_p + 2*self.a*phi_p*(phi_r + phi_m)  + self.b*phi_p*phi_p + 2*self.c*(phi_r + phi_m)*phi_p**2 + 
				2.0*self.rho_r*phi_r + 2*self.rho_c*phi_m - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_m(self, phi_p, phi_r, phi_m):

		return (-self.chi*phi_p + 2*self.a*phi_p*(phi_r + phi_m)  + self.b*phi_p*phi_p + 2*self.c*(phi_r + phi_m)*phi_p**2 +
				2.0*self.rho_c*phi_r + 2.0*self.rho_m*phi_m)
		
		
	def dmu_p_dphi_p(self, phi_p, phi_r, phi_m):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + 2*self.b*(phi_r+phi_m) + 2*self.c*(phi_r + phi_m)**2)

	def dmu_p_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*(phi_r + phi_m) + 2*self.b*phi_p + 4*self.c*phi_p*(phi_r+phi_m))  
	
	def dmu_p_dphi_m(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*(phi_r + phi_m) + 2*self.b*phi_p + 4*self.c*phi_p*(phi_r+phi_m))  

	def dmu_r_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (2.0*self.rho_r + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)
	
	def dmu_r_dphi_m(self, phi_p, phi_r, phi_m):

		return (2.0*self.rho_c + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)

	def dmu_m_dphi_m(self, phi_p, phi_r, phi_m):

		return (2.0*self.rho_m + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)

	def mu_r_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return(fp.CellVariable(mesh = mesh, name = '$K exp(-(r-r_0)^2/L^2_R)$', value=self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2)))
		# return (self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_p_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		
		return(fp.CellVariable(mesh = mesh, name = '$c_max exp(-(r-r_E)^2/sigma^2)$', value=self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))
		# return (self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2))
	
	def det_J(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
		"""
		return (self.dmu_p_dphi_p(phi_p, phi_r, phi_m)*(self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)**2) -
				self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)) + 
				self.dmu_p_dphi_m(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_r_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)))

	# def mu_r_multiplier(self, mesh):
	# 	return(fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_RNA',value=2.0*self.rho_r*(1.0 - self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2))))

	# def mu_p_multiplier(self, mesh):
	# 	return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_Protein',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_N_total, domain_radius, circ_flag):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		
		if circ_flag: # Since all the calculations below are performed assuming a circular domain

			def solve_for_rmax_ratio(x, *args):
				N_total = args[0]
				rho_r = args[1]
				chi_r = args[2]
				L_R = args[3]

				eqn = -np.exp(-x**2)*x**2 + 1.0 - np.exp(-x) - 2.0*rho_r*N_total/(np.pi*chi_r*L_R**2)
				return eqn

			N_tot_ratio =  2.0*self.rho_r*rna_nucleus_N_total/(np.pi*self.K*self.L_R**2)
			X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
			Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

			if N_tot_ratio < 1.0:   
				ratio = spo.fsolve(solve_for_rmax_ratio, [1.0], args=(rna_nucleus_N_total, self.rho_r, self.K, self.L_R))
				r_max = ratio*self.L_R 
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r)*(1-np.exp(-ratio)))*2.0*self.rho_r/(np.pi*r_max**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)
				phi_r[phi_r < 0.0] = 0.0
			else:
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r))*2.0*self.rho_r/(np.pi*domain_radius**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)


class free_energy_protein_nuc_RNA_gaussian_mRNA_no_lr_int:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, rho_s, c_alpha, c_beta, chi, c, kappa, rho_r, K, r0, L_R, c_max, sigma, rE, a=0.0, b=0.0):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.chi = chi
		self.c = c
		self.kappa = kappa
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.rho_r = rho_r
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
		self.a = a
		self.b = b
	
	def f(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 + 0.5*self.kappa*(phi_p.grad.mag)**2 - 
			   self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p - 
			   self.chi*phi_p*(phi_m + phi_r) + self.a*(phi_r**2 + phi_m**2)*phi_p + self.b*(phi_r + phi_m)*phi_p**2 + self.c*(phi_r**2 + phi_m**2)*phi_p**2 + 
			   self.rho_r*(phi_r**2 + phi_m**2) - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)*phi_r)
	
	def mu_p(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha - self.c_beta) - self.kappa*(phi_p.faceGrad.divergence) - 
				self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2) - 
				self.chi*(phi_r + phi_m) + self.a*(phi_r**2 + phi_m**2) + 2*self.b*(phi_r + phi_m)*phi_p + 2*self.c*(phi_r**2 + phi_m**2)*phi_p)
	
	def mu_r(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		return (-self.chi*phi_p + 2*self.a*phi_p*phi_r  + self.b*phi_p*phi_p + 2*self.c*phi_r*phi_p**2 + 
				2.0*self.rho_r*phi_r - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_m(self, phi_p, phi_r, phi_m):

		return (-self.chi*phi_p + 2*self.a*phi_p*phi_m  + self.b*phi_p*phi_p + 2*self.c*phi_m*phi_p**2 +
				2.0*self.rho_r*phi_m)
		
		
	def dmu_p_dphi_p(self, phi_p, phi_r, phi_m):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + 2*self.b*(phi_r+phi_m) + 2*self.c*(phi_r**2 + phi_m**2))

	def dmu_p_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*phi_r + 2*self.b*phi_p + 4*self.c*phi_p*phi_r)  
	
	def dmu_p_dphi_m(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi + 2*self.a*phi_m + 2*self.b*phi_p + 4*self.c*phi_p*phi_m)  

	def dmu_r_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (2.0*self.rho_r + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)
	
	def dmu_r_dphi_m(self, phi_p, phi_r, phi_m):

		return 0.0

	def dmu_m_dphi_m(self, phi_p, phi_r, phi_m):

		return (2.0*self.rho_r + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)

	def mu_r_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return(fp.CellVariable(mesh = mesh, name = '$K exp(-(r-r_0)^2/L^2_R)$', value=self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2)))
		# return (self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_p_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		
		return(fp.CellVariable(mesh = mesh, name = '$c_max exp(-(r-r_E)^2/sigma^2)$', value=self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))
		# return (self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2))
	
	def det_J(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
		"""
		return (self.dmu_p_dphi_p(phi_p, phi_r, phi_m)*(self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)**2) -
				self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)) + 
				self.dmu_p_dphi_m(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_r_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)))

	# def mu_r_multiplier(self, mesh):
	# 	return(fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_RNA',value=2.0*self.rho_r*(1.0 - self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2))))

	# def mu_p_multiplier(self, mesh):
	# 	return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_Protein',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_N_total, domain_radius, circ_flag):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		
		if circ_flag: # Since all the calculations below are performed assuming a circular domain

			def solve_for_rmax_ratio(x, *args):
				N_total = args[0]
				rho_r = args[1]
				chi_r = args[2]
				L_R = args[3]

				eqn = -np.exp(-x**2)*x**2 + 1.0 - np.exp(-x) - 2.0*rho_r*N_total/(np.pi*chi_r*L_R**2)
				return eqn

			N_tot_ratio =  2.0*self.rho_r*rna_nucleus_N_total/(np.pi*self.K*self.L_R**2)
			X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
			Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

			if N_tot_ratio < 1.0:   
				ratio = spo.fsolve(solve_for_rmax_ratio, [1.0], args=(rna_nucleus_N_total, self.rho_r, self.K, self.L_R))
				r_max = ratio*self.L_R 
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r)*(1-np.exp(-ratio)))*2.0*self.rho_r/(np.pi*r_max**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)
				phi_r[phi_r < 0.0] = 0.0
			else:
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r))*2.0*self.rho_r/(np.pi*domain_radius**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)


class free_energy_protein_nuc_RNA_gaussian_mRNA_walled:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, rho_s, c_alpha, c_beta, chi_r, chi_m, c, kappa, rho_r, rho_m, rho_c, K, r0, L_R, c_max, sigma, rE, wall_k, neg_max, a=0.0, b=0.0):
		self.rho_s = rho_s
		self.c_alpha = c_alpha
		self.c_beta = c_beta
		self.chi_m = chi_m
		self.chi_r = chi_r
		self.c = c
		self.kappa = kappa
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.rho_r = rho_r
		self.rho_m = rho_m
		self.rho_c = rho_c
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
		self.wall_k = wall_k
		self.neg_max = neg_max
		self.a = a
		self.b = b
	
	def f(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""
		return(self.rho_s*(phi_p-self.c_alpha)**2*(phi_p-self.c_beta)**2 + 0.5*self.kappa*(phi_p.grad.mag)**2 - 
			   self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p - 
			   self.chi_m*phi_p*phi_m + self.chi_r*phi_p*phi_r + self.a*(phi_r + phi_m)**2*phi_p + self.b*(phi_r + phi_m)*phi_p**2 + self.c*(phi_r + phi_m)**2*phi_p**2 + 
			   self.rho_r*phi_r**2 + self.wall_k*(phi_r < self.neg_max)*(phi_r-self.neg_max)**4 + self.rho_m*phi_m**2 + 2*self.rho_c*phi_r*phi_m - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)*phi_r)
	
	def mu_p(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return (2*self.rho_s*(phi_p-self.c_alpha)*(phi_p-self.c_beta)*(2*phi_p-self.c_alpha - self.c_beta) - self.kappa*(phi_p.faceGrad.divergence) - 
				self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2) - 
				self.chi_r*phi_r + self.chi_m*phi_m + self.a*(phi_r + phi_m)**2 + 2*self.b*(phi_r + phi_m)*phi_p + 2*self.c*(phi_r + phi_m)**2*phi_p)
	
	def mu_r(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		return (-self.chi_r*phi_p + 2*self.a*phi_p*(phi_r + phi_m)  + self.b*phi_p*phi_p + 2*self.c*(phi_r + phi_m)*phi_p**2 + 
				2.0*self.rho_r*phi_r + 4*self.wall_k*(phi_r < self.neg_max)*(phi_r-self.neg_max)**3 + 2*self.rho_c*phi_m - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_m(self, phi_p, phi_r, phi_m):

		return (-self.chi_m*phi_p + 2*self.a*phi_p*(phi_r + phi_m)  + self.b*phi_p*phi_p + 2*self.c*(phi_r + phi_m)*phi_p**2 +
				2.0*self.rho_c*phi_r + 2.0*self.rho_m*phi_m)
		
		
	def dmu_p_dphi_p(self, phi_p, phi_r, phi_m):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (2*self.rho_s*((self.c_alpha - phi_p)**2 + 4*(self.c_alpha - phi_p)*(self.c_beta - phi_p) + (self.c_beta - phi_p)**2) + 2*self.b*(phi_r+phi_m) + 2*self.c*(phi_r + phi_m)**2)

	def dmu_p_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi_r + 2*self.a*(phi_r + phi_m) + 2*self.b*phi_p + 4*self.c*phi_p*(phi_r+phi_m))  
	
	def dmu_p_dphi_m(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi_m + 2*self.a*(phi_r + phi_m) + 2*self.b*phi_p + 4*self.c*phi_p*(phi_r+phi_m))  

	def dmu_r_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (2.0*self.rho_r + 4*3*self.wall_k*(phi_r < self.neg_max)*(phi_r-self.neg_max)**2 + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)
	
	def dmu_r_dphi_m(self, phi_p, phi_r, phi_m):

		return (2.0*self.rho_c + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)

	def dmu_m_dphi_m(self, phi_p, phi_r, phi_m):

		return (2.0*self.rho_m + 2*self.a*phi_p +  2*self.c*phi_p*phi_p)

	def mu_r_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return(fp.CellVariable(mesh = mesh, name = '$K exp(-(r-r_0)^2/L^2_R)$', value=self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2)))
		# return (self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_p_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		
		return(fp.CellVariable(mesh = mesh, name = '$c_max exp(-(r-r_E)^2/sigma^2)$', value=self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))
		# return (self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2))
	
	def det_J(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
		"""
		return (self.dmu_p_dphi_p(phi_p, phi_r, phi_m)*(self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)**2) -
				self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)) + 
				self.dmu_p_dphi_m(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_r_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)))

	# def mu_r_multiplier(self, mesh):
	# 	return(fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_RNA',value=2.0*self.rho_r*(1.0 - self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2))))

	# def mu_p_multiplier(self, mesh):
	# 	return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_Protein',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_N_total, domain_radius, circ_flag):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		
		if circ_flag: # Since all the calculations below are performed assuming a circular domain

			def solve_for_rmax_ratio(x, *args):
				N_total = args[0]
				rho_r = args[1]
				chi_r = args[2]
				L_R = args[3]

				eqn = -np.exp(-x**2)*x**2 + 1.0 - np.exp(-x) - 2.0*rho_r*N_total/(np.pi*chi_r*L_R**2)
				return eqn

			N_tot_ratio =  2.0*self.rho_r*rna_nucleus_N_total/(np.pi*self.K*self.L_R**2)
			X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
			Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

			if N_tot_ratio < 1.0:   
				ratio = spo.fsolve(solve_for_rmax_ratio, [1.0], args=(rna_nucleus_N_total, self.rho_r, self.K, self.L_R))
				r_max = ratio*self.L_R 
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r)*(1-np.exp(-ratio)))*2.0*self.rho_r/(np.pi*r_max**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)
				phi_r[phi_r < 0.0] = 0.0
			else:
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r))*2.0*self.rho_r/(np.pi*domain_radius**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)



class free_energy_protein_lncRNA_mRNA_FH:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, NP, NR, NM, chi_p, chi_pr, rho_r, kappa, K, r0, L_R, c_max, sigma, rE):
		
		self.NP = NP
		self.NR = NM
		self.NM = NM
		self.chi_p = chi_p
		self.chi_pr = chi_pr
		self.rho_r = rho_r
		self.kappa = kappa 
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
	
	def f(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""

		fe = phi_p*np.log(phi_p)/self.NP + (1.0-phi_p)*np.log(1.0-phi_p) + 0.5*self.kappa*(phi_p.grad.mag)**2 - self.chi_p*phi_p**2 - self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p - self.chi_pr*phi_p*(phi_m+phi_r) + 0.5*(phi_r + phi_m)**2*phi_p + 0.5*(phi_r + phi_m)*phi_p**2 + 0.5*(phi_r + phi_m)**2*phi_p**2 + self.rho_r*(phi_r+phi_m)**2 - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)*phi_r
		
		if np.sum(phi_r.value) > 0.0:

			fe = fe + phi_r*np.log(phi_r)/self.NR
		
		if np.sum(phi_m.value) > 0.0: 

			fe = fe + phi_m*np.log(phi_m)/self.NM

		return fe
		
	def mu_p(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return ((1.0+np.log(phi_p))/self.NP - 1.0 - np.log(1.0-phi_p) - 2*self.chi_p*phi_p - self.kappa*(phi_p.faceGrad.divergence) - 
					self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2) - 
					self.chi_pr*(phi_r + phi_m) + 0.5*(phi_r + phi_m)**2 + (phi_r + phi_m)*phi_p + (phi_r + phi_m)**2*phi_p)

	def mu_r(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		
		mu_r = - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2) - self.chi_pr*phi_p + phi_p*(phi_r + phi_m)  + 0.5*phi_p**2 + (phi_r + phi_m)*phi_p**2 + 2.0*self.rho_r*(phi_r + phi_m)

		if np.sum(phi_r.value) > 0.0:
			mu_r = mu_r + (1.0+np.log(phi_r))/self.NR

		return mu_r

	def mu_m(self, phi_p, phi_r, phi_m):


		mu_m = - self.chi_pr*phi_p + phi_p*(phi_r + phi_m)  + 0.5*phi_p**2 + (phi_r + phi_m)*phi_p**2 + 2.0*self.rho_r*(phi_r + phi_m)

		if np.sum(phi_m.value) > 0.0:

			mu_m = mu_m + (1.0+np.log(phi_m))/self.NM 

		return mu_m
			
		
	def dmu_p_dphi_p(self, phi_p, phi_r, phi_m):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (1/self.NP + phi_p*(1.0-phi_p)**(-1) - 2*self.chi_p*phi_p + 
				   phi_p*(phi_r + phi_m) + phi_p*(phi_r + phi_m)**2)

	def dmu_p_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi_pr + (phi_r + phi_m) + phi_p + 2*phi_p*(phi_r+phi_m))*phi_p  

	
	def dmu_p_dphi_m(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi_pr + (phi_r + phi_m) + phi_p + 2*phi_p*(phi_r+phi_m))*phi_p  

	def dmu_r_dphi_p(self, phi_p, phi_r, phi_m):

		return (-self.chi_pr + (phi_r + phi_m) + phi_p + 2*phi_p*(phi_r+phi_m))*phi_r


	def dmu_r_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (1/self.NR + (2.0*self.rho_r + phi_p +  phi_p**2)*phi_r)
	
	def dmu_r_dphi_m(self, phi_p, phi_r, phi_m):

		return (2.0*self.rho_r + phi_p +  phi_p**2)*phi_r

	# def dmu_m_dphi_m(self, phi_p, phi_r, phi_m):

	# 	dmu = 2.0*self.rho_r + phi_p +  phi_p**2
	# 	if np.sum(phi_m.value) > 0.0:
	# 		dmu = dmu + (self.NR*phi_m)**(-1)

	# 	return dmu

	def mu_r_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return(fp.CellVariable(mesh = mesh, name = '$K exp(-(r-r_0)^2/L^2_R)$', value=self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2)))
		# return (self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_p_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		
		return(fp.CellVariable(mesh = mesh, name = '$c_max exp(-(r-r_E)^2/sigma^2)$', value=self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))
		# return (self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2))
	
	# def det_J(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
	# 	"""
	# 	Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
	# 	"""
	# 	return (self.dmu_p_dphi_p(phi_p, phi_r, phi_m)*(self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)**2) -
	# 			self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)) + 
	# 			self.dmu_p_dphi_m(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_r_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)))

	# def mu_r_multiplier(self, mesh):
	# 	return(fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_RNA',value=2.0*self.rho_r*(1.0 - self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2))))

	# def mu_p_multiplier(self, mesh):
	# 	return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_Protein',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_N_total, domain_radius, circ_flag):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		
		if circ_flag: # Since all the calculations below are performed assuming a circular domain

			def solve_for_rmax_ratio(x, *args):
				N_total = args[0]
				rho_r = args[1]
				chi_r = args[2]
				L_R = args[3]

				eqn = -np.exp(-x**2)*x**2 + 1.0 - np.exp(-x) - 2.0*rho_r*N_total/(np.pi*chi_r*L_R**2)
				return eqn

			N_tot_ratio =  2.0*self.rho_r*rna_nucleus_N_total/(np.pi*self.K*self.L_R**2)
			X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
			Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

			if N_tot_ratio < 1.0:   
				ratio = spo.fsolve(solve_for_rmax_ratio, [1.0], args=(rna_nucleus_N_total, self.rho_r, self.K, self.L_R))
				r_max = ratio*self.L_R 
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r)*(1-np.exp(-ratio)))*2.0*self.rho_r/(np.pi*r_max**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)
				phi_r[phi_r < 0.0] = 1e-8
			else:
				C1 = (rna_nucleus_N_total - self.K*np.pi*self.L_R**2/(2.0*self.rho_r))*2.0*self.rho_r/(np.pi*domain_radius**2)
				phi_r.value = C1/(2.0*self.rho_r) + self.K/(2.0*self.rho_r)*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)

class free_energy_modified_FH:
	"""
	Defining a class capturing the free energy of interaction between lncRNA and protein
	"""
	
	def __init__(self, N_P, N_R, N_M, chi_p, chi_r, chi_m, rho_r, kappa, K, r0, L_R, c_max, sigma, rE, a=0.5, b=0.5, c=0.5):
		
		self.N_P = N_P
		self.N_R = N_R
		self.N_M = N_M
		self.chi_p = chi_p
		self.chi_r = chi_r
		self.chi_m = chi_m
		self.rho_r = rho_r
		self.a = a
		self.b = b
		self.c = c
		self.kappa = kappa 
		self.K = K
		self.r0 = r0
		self.L_R = L_R
		self.c_max = c_max
		self.sigma = sigma
		self.rE = rE
	
	def f(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns overall free-energy per unit volume, including gradient (surface-tension) terms:
		
		.. math::
			f = \rho_s(\phi_p-\alpha)^2(\phi_p-\beta)^2 - \chi\phi_p\phi_r + a\phi_p\phi^2_r + b\phi^2_p\phi_r + c\phi^2_p\phi^2_r + K(\frac{|r-r_0|^2}{L^2_r}+1)\phi^2_r + \kappa/2|\nabla\phi_p|^2
		"""

		fe = phi_p*np.log(phi_p)/self.N_P + (1.0-phi_p)*np.log(1.0-phi_p) + 0.5*self.kappa*(phi_p.grad.mag)**2 - self.chi_p*phi_p**2 - self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2)*phi_p + phi_r*np.log(phi_r)/self.N_R + phi_m*np.log(phi_m)/self.N_M - self.chi_r*phi_p*phi_r - self.chi_m*phi_p*phi_m + self.a*(phi_r + phi_m)**2*phi_p + self.b*(phi_r + phi_m)*phi_p**2 + self.c*(phi_r + phi_m)**2*phi_p**2 - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)*phi_r

		return fe
		
	def mu_p(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns protein chemical potential

		.. math::
			\mu_{p} = \\frac{df}{d \phi_{p}}
		"""
		return ((1.0+np.log(phi_p))/self.N_P - 1.0 - np.log(1.0-phi_p) - 2*self.chi_p*phi_p - self.kappa*(phi_p.faceGrad.divergence) - 
					self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2) - 
					self.chi_r*phi_r -self.chi_m*phi_m + self.a*(phi_r + phi_m)**2 + 2*self.b*(phi_r + phi_m)*phi_p + 2*self.c*(phi_r + phi_m)**2*phi_p)

	def mu_r(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
		"""
		Returns RNA chemical potential

		.. math::
			\mu_{r} = \\frac{df}{d \phi_{r}}
		"""
		
		mu_r = - self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2) - self.chi_r*phi_p + 2*self.a*phi_p*(phi_r + phi_m)  + self.b*phi_p**2 + 2*self.c*(phi_r + phi_m)*phi_p**2 + 2.0*self.rho_r*(phi_r + phi_m)

		if np.sum(phi_r.value) > 0.0:
			mu_r = mu_r + (1.0+np.log(phi_r))/self.N_R 

		return mu_r

	def mu_m(self, phi_p, phi_r, phi_m):


		mu_m = (1.0+np.log(phi_m))/self.N_M  - self.chi_m*phi_p + 2*self.a*phi_p*(phi_r + phi_m)  + self.b*phi_p**2 + 2*self.c*(phi_r + phi_m)*phi_p**2 + 2.0*self.rho_r*(phi_r + phi_m)

		
		if np.sum(phi_m.value) > 0.0:
			mu_m = mu_m + (1.0+np.log(phi_m))/self.N_M 

		return mu_m
			
		
	def dmu_p_dphi_p(self, phi_p, phi_r, phi_m):     
		"""
		Returns derivative of protein chemical potential with protein concentration (except for the surface tension term)

		.. math::
			 \\frac{d^{2}f (except surf tension)}{d \phi_{p}^{2}}
		"""
		return (1/self.N_P + phi_p*(1.0-phi_p)**(-1) - 2*self.chi_p*phi_p + 
				   2*self.b*phi_p*(phi_r + phi_m) + 2*self.c*phi_p*(phi_r + phi_m)**2)

	def dmu_p_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi_r + 2*self.a*(phi_r + phi_m) + 2*self.b*phi_p + 4*self.c*phi_p*(phi_r+phi_m))*phi_p  

	
	def dmu_p_dphi_m(self, phi_p, phi_r, phi_m):
		"""
		Returns mixed second derivative of free-energy

		.. math::
			 \\frac{d^{2}f}{d \phi_{p} \phi_{r}}
		"""
		return (-self.chi_m + 2*self.a*(phi_r + phi_m) + 2*self.b*phi_p + 4*self.c**phi_p*(phi_r+phi_m))*phi_p  

	def dmu_r_dphi_p(self, phi_p, phi_r, phi_m):

		return (-self.chi_r + 2*self.a*(phi_r + phi_m) + 2*self.b*phi_p + 4*self.c*phi_p*(phi_r+phi_m))*phi_r


	def dmu_r_dphi_r(self, phi_p, phi_r, phi_m):
		"""
		Returns derivative of RNA chemical potential with RNA concentration

		.. math::
			 \\frac{d^{2}f}{d \phi_{r}^{2}}
		"""
		return (1/self.N_R + (2.0*self.rho_r + 2*self.a*phi_p +  2*self.c*phi_p**2)*phi_r)
	
	def dmu_r_dphi_m(self, phi_p, phi_r, phi_m):

		return (2.0*self.rho_r + 2*self.a*phi_p +  2*self.c*phi_p**2)*phi_r

	# def dmu_m_dphi_m(self, phi_p, phi_r, phi_m):

	# 	dmu = 2.0*self.rho_r + phi_p +  phi_p**2
	# 	if np.sum(phi_m.value) > 0.0:
	# 		dmu = dmu + (self.NR*phi_m)**(-1)

	# 	return dmu

	def mu_r_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		return(fp.CellVariable(mesh = mesh, name = '$K exp(-(r-r_0)^2/L^2_R)$', value=self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2)))
		# return (self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

	def mu_p_multiplier(self, mesh):
		"""
		Returns the parabolic multiplier to phi_r in the RNA chemical potential
		
		..math::
			K(\frac{|r-r_0|^2}{L^2_r}+1)
		"""
		
		return(fp.CellVariable(mesh = mesh, name = '$c_max exp(-(r-r_E)^2/sigma^2)$', value=self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))
		# return (self.c_max*np.exp(-((X_CV-self.rE[0])**2 + (Y_CV-self.rE[1])**2)/self.sigma**2))
	
	# def det_J(self, phi_p, phi_r, phi_m, X_CV, Y_CV):
	# 	"""
	# 	Returns product of eigenvalues of the Jacobian associated with the bulk free energy of the system (not including the surface tension term)
	# 	"""
	# 	return (self.dmu_p_dphi_p(phi_p, phi_r, phi_m)*(self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)**2) -
	# 			self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_m_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_m(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)) + 
	# 			self.dmu_p_dphi_m(phi_p, phi_r, phi_m)*(self.dmu_p_dphi_r(phi_p, phi_r, phi_m)*self.dmu_r_dphi_m(phi_p, phi_r, phi_m) - self.dmu_r_dphi_r(phi_p, phi_r, phi_m)*self.dmu_p_dphi_m(phi_p, phi_r, phi_m)))

	# def mu_r_multiplier(self, mesh):
	# 	return(fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_RNA',value=2.0*self.rho_r*(1.0 - self.K*np.exp(-((mesh.cellCenters[0]-self.r0[0])**2 + (mesh.cellCenters[1]-self.r0[1])**2)/self.L_R**2))))

	# def mu_p_multiplier(self, mesh):
	# 	return (fp.CellVariable(mesh = mesh, name = 'Gaussian_Multiplier_Protein',value=-2.0*self.c_max*np.exp(-((mesh.cellCenters[0]-self.rE[0])**2 + (mesh.cellCenters[1]-self.rE[1])**2)/self.sigma**2)))

	def set_initial_rna_profile(self, phi_r, mesh, rna_nucleus_N_total, domain_radius, circ_flag):
		"""
		Function to set concentration profile of RNA in such a way that it's initial chemical potential gradients are 0
		"""
		
		if circ_flag: # Since all the calculations below are performed assuming a circular domain

			X_CV = fp.CellVariable(mesh=mesh, name=r'$X_CV$', value=mesh.cellCenters[0])
			Y_CV = fp.CellVariable(mesh=mesh, name=r'$Y_CV$', value=mesh.cellCenters[1])

			C1 = rna_nucleus_N_total*(domain_radius/self.L_R)**2*1.0/(sc.expi(self.N_R*self.K)-sc.expi(self.N_R*self.K*np.exp(-(domain_radius/self.L_R)**2)))
			phi_r.value = C1*np.exp(self.N_R*self.K*np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2))

			# C1 = np.exp(self.K*(np.exp(-((X_CV-self.r0[0])**2 + (Y_CV-self.r0[1])**2)/self.L_R**2)-1.0))
			# phi_r.value = rna_nucleus_N_total*C1 + (C1-1.0)/(2*self.rho_r)
			# phi_r[phi_r < 0.0] = 1e-8



class RNA_reactions:

	def __init__(self, mesh, k_p_max, k_degradation, spread, center, phi_threshold=0.0, n_hill=1.0):

		self.k_p_max = k_p_max
		self.k_d = k_degradation
		self.std = spread
		self.center = center
		self.phi_threshold = phi_threshold
		self.n_hill = n_hill

		self.k_p_x = self.k_p_max*fp.CellVariable(mesh = mesh, name=r'$k_p(x)$', value = np.exp(-((self.center[0]-mesh.cellCenters[0])**2 + (self.center[1]-mesh.cellCenters[1])**2)/self.std**2))
		# self.k_p_x_flat = self.k_p_max*fp.CellVariable(mesh = mesh, name=r'$k_p(x)_flat$', value = heaviside_limit_flux((self.center[0]-mesh.cellCenters[0])**2 + (self.center[1]-mesh.cellCenters[1])**2 - self.std**2))

	# def calculate_k_p_x(self, mesh, k_p_integral):
		
	# 	gaussian_rate = fp.CellVariable(mesh = mesh, name=r'$k_p(x)$', value = np.exp(-((self.center[0]-mesh.cellCenters[0])**2 + (self.center[1]-mesh.cellCenters[1])**2)/self.std**2))

	# 	kp_max = self.k_p_average/self.gaussian_rate.cellVolumeAverage.value

	#	return (kp_max*gaussian_rate)

	def production_rate(self, phi):

		return self.k_p_x*(phi - self.phi_threshold)*(phi > self.phi_threshold)

	def production_rate_hill_gaussian(self, phi):

		return self.k_p_x*(phi**self.n_hill)*(phi**self.n_hill + self.phi_threshold**self.n_hill)**(-1)

	def production_rate_flat_in_space(self, phi):

		return self.k_p_max*(phi - self.phi_threshold)*(phi > self.phi_threshold)

	def production_rate_thresholded(self, phi):

		return self.k_p_max*(phi > self.phi_threshold)

	def production_rate_no_concentration_dependence(self):

		return self.k_p_x

	def degradation_rate(self, phi):

		return self.k_d*phi


# Heaviside step function to ensure that the fluxes are non-zero only when the species concentrations are non zero

def heaviside_limit_flux(phi_var):
	heaviside_multiplier = np.ones(len(phi_var.value))
	heaviside_multiplier[phi_var <= 0.0] = 0.0
	return heaviside_multiplier