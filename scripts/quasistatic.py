from dolfin import *
#set_log_level(40)

# Compiler options
flags = ["-O3", "-ffast-math", "-march=native"]
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# ----------------- Import and mark mesh ----------------- #

# Mesh with dimensions
mesh = Mesh('LV_mesh.xml')
a1 = 0.02 + 0.013 	; c1 = 0.06 + 0.01	# m (principal axes for big)
a2 = 0.02  			; c2 = 0.06 		# m (principal axes for small)

# Marking boundaries 
class Epicardium(SubDomain): 			
	def inside(self, x, on_boundry):
		return on_boundry

class Endocardium(SubDomain): 			
	def inside(self, x, on_boundry):
		return (x[0]*x[0]+x[2]*x[2] < (a2*a2)*(1.0-(x[1]*x[1])/(c2*c2)) + 1e-7) and on_boundry

class Base(SubDomain): 		
	def inside(self, x, on_boundry):
		return (x[1] > -1e-7) and on_boundry


# ----------------- Functions ----------------- #

def Make_SubDomains(mesh):
	mf = FacetFunction("size_t", mesh)
	mf.set_all(0)
	Epicardium().mark(mf, 40)  		# Outside
	Endocardium().mark(mf, 30)		# Inside
	Base().mark(mf,10) 				# Base
	#plot(mf,interactive=True)
	for facet in facets(mesh):
		mesh.domains().set_marker((facet.index(), mf[facet]), 2)

	return mf

def Create_Fiberfield(mesh):

	# This does not work i parallel so field has to be 
	# created and saved in series, then loaded in parallel.

	from fiberrules import dolfin_fiberrules, dolfin_to_vtk
	fiber_angle_epi = 150
	fiber_angle_endo = 30
	fiber_space = FunctionSpace(mesh, 'Quadrature',2)

	fibers, sheet_normals, cross_sheet = \
	        dolfin_fiberrules(mesh, fiber_space,
	                          fiber_angle_epi, 
	                          fiber_angle_endo)

	#dolfin_to_vtk(fibers, "fibers_test")
	#Save('fiberfield.h5','f_0',fibers)
	#Save('fiberfield.h5','s_0',sheet_normals)
	#Save('fiberfield.h5','n_0',cross_sheet)

	return fibers, sheet_normals, cross_sheet

def Solve_System(Ta,p_0):

	# Update time-dependent parameters
	T_a.assign(Ta)
	p0.assign(p_0)

	solve(eq==0,w,bcs,J=Jac)

def Isovolumetric_PressureUpdate(i,volume_constraint):
	dp = 10.
	iterations = 0
	err = 1.
	p_0 = pressure[i-1]

	while err > 0.01:
		iterations += 1

		print ' \n Now doing Newton iteration number ', iterations, ' for pressure update. \n'

		Solve_System(active_tension[i],p_0)
		Vf_0 = Compute_Volume(u)
		Solve_System(active_tension[i],p_0+dp)
		Vf_1 = Compute_Volume(u)

		p_1 = p_0 - (Vf_0-volume_constraint)/((Vf_1-Vf_0)/dp)
		err = abs(p_1 - p_0)
		p_0 = p_1

		print '\n Error after ' ,iterations, ' iterations = ', err

	print '\n PRESSURE UPDATE CONVERGED IN ', iterations ,  ' iterations \n ' 
	print '\n Pressure is now', p_1, ' \n ' 
	return p_1

def PressureUpdatePhase3(i,V_ED):
	#dp = 10./100
	dp = 100
	DT = (t[i] - t[i-1])*1000
	iterations = 0
	err = 1.
	p_0 = pressure[i-1]

	C_art = 0.001 
	R_per = 20000
	P_o = 500

	while err > 0.01:
		iterations += 1

		print ' \n Now doing iteration number ', iterations, ' for pressure update PHASE 3. \n'

		Solve_System(active_tension[i],p_0)
		Vf_0 = Compute_Volume(u)

		Solve_System(active_tension[i],p_0+dp)
		Vf_1 = Compute_Volume(u)

		dVdp = (Vf_1-Vf_0)/dp

		g = (Vf_0 - volume[i-1]) + C_art*(p_0-pressure[i-1]) + DT*(p_0-P_o)/R_per
		dg_dp = dVdp + C_art + DT/R_per

		p_1 = p_0 - g/dg_dp

		err = abs(p_1 - p_0)
		p_0 = p_1

		print '\n Error after ' ,iterations, ' iterations = ', err

	print '\n PRESSURE UPDATE CONVERGED IN ', iterations ,  ' iterations \n ' 
	print '\n Outflow is now', (Vf_0 - volume[i-1]), ' \n ' 

	return p_1

def Compute_Volume(u):
	X = SpatialCoordinate(mesh)
	vol = 1e6*abs(assemble((-1.0/3.0)*dot(X + u, J*inv(F).T*n_mesh)*ds(30)))
	return vol

def Save(h5name, h5group, func):
	import os
    
	group1 = "{}/function".format(h5group) # Put each function in its own subgroup

    # Append to the file if it allready exist, otherwise create a new file
	file_mode = "a" if os.path.isfile(h5name) else "w"

    # Open file and write
	with HDF5File(mpi_comm_world(), h5name, file_mode) as h5file:
		h5file.write(func, group1)

def Load(h5name, h5group, func):
	group1 = "{}/function".format(h5group)

    # Open file and read
	with HDF5File(mpi_comm_world(), h5name, "r") as h5file:
		h5file.read(func, group1)

def Plot_Results(t0,t1):
	VOLUME = np.loadtxt('volume.txt')
	PRESSURE = np.loadtxt('pressure.txt')

	plt.figure(1)
	plt.plot(t[t0:t1],VOLUME[t0:t1])
	plt.ylim([40,100])
	plt.xlabel('time(ms)')
	plt.ylabel('volume(ml)')

	plt.figure(2)
	plt.plot(t[t0:t1],PRESSURE[t0:t1])
	plt.xlabel('time(seconds)')
	plt.ylabel('pressure(Pa)')

	plt.show()

# --------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt 
import numpy as np

active_tension = np.loadtxt('active_tension_900ms.txt')*0.76
t = np.linspace(0,0.9,len(active_tension))	
pressure = np.zeros(len(t))
pressure[:500] = (t[:500]/0.5)*1500

dt = Constant(t[1])

# Set up mesh boundaries and normal 
boundaries = Make_SubDomains(mesh)
ds = ds[boundaries] 
n_mesh = FacetNormal(mesh)

# Set up function spaces and check size of system
V = VectorFunctionSpace(mesh, "Lagrange", 2)
P = FunctionSpace(mesh,"Lagrange",1)
W = V*P
print 'Number of DOF: ', W.dim()

# Define functions
w = Function(W)
(u, p) = split(w)
(du, dp) = TestFunctions(W)

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
F = variable(F)
C = F.T*F                   # Right Cauchy-Green tensor
B = F*F.T

# Invariants of deformation tensors
I_1 = tr(C)
J = det(F)

# Material parameters
eta = 0.1  						# weighting of transversal fiber tension
rho = Constant(1000.0) 			# kg
a = 228.0		*alpha 			# Pa
a_f = 116.85	*alpha 		 	# Pa
b = 7.780					 	# dimensionless
b_f = 11.83425				 	# dimensionless

# Unit vectors (needed to set up active stress)
FiberSpace = VectorFunctionSpace(mesh,'Quadrature',2)
f_0 = Function(FiberSpace) 	# fiber direction
s_0 = Function(FiberSpace) 	# sheet vector
n_0 = Function(FiberSpace) 	# sheet normal
Load('fiberfield.h5','f_0',f_0);
Load('fiberfield.h5','s_0',s_0)
Load('fiberfield.h5','n_0',n_0)
f = F*f_0
s = F*s_0
n = F*n_0

# Invariants
I_4f = inner(C*f_0,f_0)
I_4s = inner(C*s_0,s_0)
I_4n = inner(C*n_0,n_0)

# Initial conditions and boundary conditions
zero_displacement = Expression(("0.0", "0.0", "0.0"))
bcr = DirichletBC(W.sub(0), zero_displacement, boundaries, 10)
bcs = [bcr]
p0 = Constant(0); T_a = Constant(0)

# Stress relations

# Passive part
passive_cauchy_stress = a*exp(b*(I_1 - 3))*B + 2*a_f*(I_4f-1)*exp(b_f*pow(I_4f - 1, 2))*outer(f,f) - p*I
P_p = J*passive_cauchy_stress*inv(F).T

# Active part
active_cauchy_stress = T_a*(outer(f,f)+eta*outer(s,s)+eta*outer(n,n))	
P_a = J*active_cauchy_stress*inv(F.T)

P =  P_p + P_a

eq = inner(P,grad(du))*dx + inner(J-1,dp)*dx + dot(J*inv(F).T*n_mesh*p0, du)*ds(30) 
Jac = derivative(eq, w)


volume = np.zeros(len(t))

stop = len(t)

phase1 = 'True'
phase3 = 'False'
for i in range(0,stop):
	print 'Now solving time-step:', i, '/', stop

	if pressure[i-1] > 7000:
		phase3 = 'True'

	if active_tension[i] < 1.0:
		print '*** PHASE 1 ***'

	elif active_tension[i] > 1.0 and phase3 == 'False':
		pressure[i] = Isovolumetric_PressureUpdate(i,max(volume))

	elif phase3 == 'True' and volume[i-1] > (volume[0]+0.1): # ml
		pressure[i] = PressureUpdatePhase3(i,max(volume))

	else:
		pressure[i] = Isovolumetric_PressureUpdate(i,volume[0])

	Solve_System(active_tension[i],pressure[i])
	Save('quasistatic.h5', 'timestep'+str(i), w)

	volume[i] = Compute_Volume(u)
	
	print 'Volume:', volume[i]
	print 'Pressure:', pressure[i]


np.savetxt('volume.txt',volume)
np.savetxt('pressure.txt',pressure)
