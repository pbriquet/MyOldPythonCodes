from VFModel import *
import matplotlib.pyplot as plt

a = [-1e-2,2e-2,0.5]
a0 = [-1e-2,1e-2,0.6]
K_real = lambda a,T: a[0]*T**2 + a[1]*T + a[2]
x = np.arange(0.0,1.0,0.05)
y = K_real(a,x)
y0 = K_real(a0,x)

nx = [2,2,2]
Lx = [1.0,1.0,1.0]
bc_args = {'h':1.0,'Tfar':0.0}
m = VFModel()
u = VFModel()
m.setMaterial(K=lambda T: K_real(a,T),rho=1.0,Cp=1.0)
u.setMaterial(K=lambda T: K_real(a0,T),rho=1.0,Cp=1.0)
m.setMesh(Lx=Lx,nx=nx)
u.setMesh(Lx=Lx,nx=nx)
m.setInitialConditions(T0=1.0)
u.setInitialConditions(T0=1.0)

m.addThermopair(Thermopair(m,(0.9,0.9,0.9),0.1))
m.setBoundaryConditions(BoundaryIndex.EAST,NewtonBoundaryConditionData(**bc_args))
u.setBoundaryConditions(BoundaryIndex.EAST,NewtonBoundaryConditionData(**bc_args))

m.RunTime(1.0,1e-4)
u.RunTime(1.0,1e-4)

print m.thermopairs[0].t
print m.thermopairs[0].T


