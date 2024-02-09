
import matplotlib.pyplot as plt
from VFModel import *

model = VFModel()
model_var = VFModel()
nx = 30
model.setMaterial(K=1.0,rho=1.0,Cp=1.0)
model.setMesh(Lx=[1.0],nx=[nx],coordinates=RectangularSystem())
model_var.setMaterial(K=lambda T:1.0 - 0.5*T,rho=lambda T:1.0 - 0.05*T,Cp=1.0)
model_var.setMesh(Lx=[1.0],nx=[nx],coordinates=RectangularSystem())

model.setInitialConditions(T0=1.0)
model_var.setInitialConditions(T0=1.0)
model.setBoundaryConditions(BoundaryIndex.EAST,NewtonBoundaryConditionData(h=2.0,Tfar=0.0))
model_var.setBoundaryConditions(BoundaryIndex.EAST,NewtonBoundaryConditionData(h=2.0,Tfar=0.0))
model.RunTime(1.0,1e-4)
model_var.RunTime(1.0,1e-4)

Temp = []
Temp_var = []

for i in xrange(nx):
    Temp.append((model.mesh(i,0,0).P[0], model.mesh(i,0,0).T))
    Temp_var.append((model_var.mesh(i,0,0).P[0], model_var.mesh(i,0,0).T))
x,y = zip(*Temp)
u,v = zip(*Temp_var)
plt.plot(x,y)
plt.plot(u,v,'-r')
plt.xlabel('x')
plt.ylabel('T')
plt.show()