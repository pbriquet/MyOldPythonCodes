from Model import *
import matplotlib.pyplot as plt

c = CylindricalModel(0.25,800.0,10)
c.setInitialConditions(lambda r: 500.0 + 50.0*r**2)
c.setBoundaryConditions(200.0,1200.0)
c.setMaterial(35.0,7800.0,400.0)
c.Dimensionless()

fig = plt.figure()
x = [[],[]]
for vf in c.mesh.vfs:
    x[0].append(vf.rc)
    x[1].append(vf.T)

plt.plot(x[0],x[1])
plt.show()