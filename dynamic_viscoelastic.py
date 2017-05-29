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
mesh = Mesh('LV_mesh).xml')
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

def Isovolumetric_PressureUpdate(i,volume_constraint):
	dp = 10.
	iterations = 0
	err = 1.
	p_0 = pressure[i-1]

	while err > 0.01:
		iterations += 1

		print ' \n Now doing Newton iteration number ', iterations, ' for pressure update. \n'

		if iterations == 999:
			Solve_System_relaxed(active_tension[i],p_0)
		else:
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

		if iterations == 999:
			Solve_System_relaxed(active_tension[i],p_0)
		else:
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


def Solve_System(Ta,p_0):

	# Update time-dependent parameters
	T_a.assign(Ta)
	p0.assign(p_0)

	solve(eq==0,uvp,bcs,J=Jac)

def Solve_System_relaxed(Ta,p_0):

	# Update time-dependent parameters
	T_a.assign(Ta)
	p0.assign(p_0)

	solve(eq==0,uvp,bcs,J=Jac,solver_parameters={'newton_solver':{'relaxation_parameter': 0.5,'maximum_iterations': 100}})

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
# ----------------------------------------------------------------------------- #

import numpy as np 
import matplotlib.pyplot as plt

active_tension = np.loadtxt('active_tension_900ms.txt')*0.76
t = np.linspace(0,0.9,len(active_tension))
pressure = np.zeros(len(t))
pressure[:500] = (t[:500]/0.5)*1500

dt = Constant(t[1])
# Set up mesh boundaries and normal 
boundaries = Make_SubDomains(mesh)
ds = ds[boundaries] 
n_mesh = FacetNormal(mesh)

# Space and functions 
U = VectorFunctionSpace(mesh,"Lagrange", 2) 	# Space for displacement 
V = VectorFunctionSpace(mesh,"Lagrange", 1)		# Space for velocity				
Q = FunctionSpace(mesh,"Lagrange",1)  			# Space for pressure
W = MixedFunctionSpace([U,V,Q])

uvp = Function(W)
uvp_1 = Function(W)
u, v, p = split(uvp)
u_1, v_1, p_1 = split(uvp_1)
du, dv, dp = TestFunctions(W)

gamma_1 = Constant(22.6)
gamma_2 = Constant(0.01)

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             				# Identity tensor
F = I + grad(u)				 			# Deformation gradient     
C = F.T*F                   				# Right Cauchy-Green tensor
B = F*F.T

# Time-rates of deformation measures
F_dot = grad((u - u_1)/dt)
l = F_dot*inv(F)
d = 0.5*(l+l.T)
C_dot = 2*F.T*d*F
B_dot = l*B + B*l.T

# Invariants of deformation tensors
I_1 = tr(C)
I_1_dot = tr(C_dot)
J = det(F)

# Elasticity/material parameters
eta = 0.1  						# weighting of transversal fiber tension
rho = Constant(1000.0) 			# kg
alpha = 0.1 					# scaling parameter to get reasonable magnitude of deformation
a = 2280.0		*alpha 			# Pa*beta [N/m^3]
a_f = 1168.5	*alpha 		 	# Pa*beta [N/m^3]
b = 9.726		*0.8		 	# dimensionless
b_f = 15.779	*0.75 		 	# dimensionless

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
I_4f_dot = inner(C_dot*f_0,f_0)

# Initial conditions and boundary conditions
zero_displacement = Expression(("0.0", "0.0", "0.0"))
bcr = DirichletBC(W.sub(0), zero_displacement, boundaries, 10)
bcs = [bcr]
p0 = Constant(0); T_a = Constant(0)

# Stress relations

# Passive part
sigma_elastic = a*exp(b*(I_1 - 3))*B + 2*a_f*(I_4f-1)*exp(b_f*pow(I_4f - 1, 2))*outer(f,f) - p*I
sigma_viscous = gamma_1*exp(gamma_2*I_1_dot)*B_dot

# Active part
sigma_active = T_a*(outer(f,f)+eta*outer(s,s)+eta*outer(n,n))

P = J*(sigma_active+sigma_elastic+sigma_viscous)*inv(F.T)

eq1 = inner(P,grad(du))*dx + (rho/dt)*(inner(v - v_1 ,du)*dx) 
eq2 = (1/dt)*(inner(u - u_1,dv)*dx) - inner(v,dv)*dx
eq3 = inner(J-1,dp)*dx

eq = eq1 + eq2 + eq3 + dot(J*inv(F).T*n_mesh*p0, du)*ds(30) 

Jac = derivative(eq, uvp)

# Time-stepping
volume = np.zeros(len(t))
velocity = np.zeros(len(t))

phase = 1
i = 0
cycle = 1
cycles = 3
ts_counter = 0

while cycle <= cycles:

	print 'Now solving time-step:', i, '/', 900

	if phase == 1 and active_tension[i] > 1.0:
		phase = 2

	if phase == 2 and pressure[i-1] > 7000:
		phase = 3

	if phase == 3 and volume[i-1] < (volume[0]+0.1):
		phase = 4


	if phase == 1:
		Solve_System(active_tension[i],pressure[i])

	if phase == 2:
		pressure[i] = Isovolumetric_PressureUpdate(i,max(volume))
		Solve_System(active_tension[i],pressure[i])

	if phase == 3:
		pressure[i] = PressureUpdatePhase3(i,max(volume))
		Solve_System(active_tension[i],pressure[i])

	if phase == 4: 
		pressure[i] = Isovolumetric_PressureUpdate(i,volume[0])
		Solve_System(active_tension[i],pressure[i])

	if t[i+1] >= 0.9:

		np.savetxt('volume_cycle'+str(cycle)+'.txt',volume)
		np.savetxt('pressure_cycle'+str(cycle)+'.txt',pressure)
		np.savetxt('velocity_cycle'+str(cycle)+'.txt',velocity)
		cycle = cycle + 1
		phase = 1
		i = 0

	Save('viscoelastic.h5', 'timestep'+str(ts_counter), uvp)
	volume[i] = Compute_Volume(u)
	velocity[i] = assemble(sqrt(dot((u-u_1)/dt,(u-u_1)/dt))*dx)/assemble(1.0*dx(mesh))

	print 'Volume:', volume[i]
	print 'Pressure:', pressure[i]
	print 'Velocity:', velocity[i]*100, 'cm/s'

	uvp_1.assign(uvp)
	i = i + 1
	ts_counter = ts_counter + 1



