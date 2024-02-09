import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["DesignPatterns","LinearAlgebra"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
_l = 'Analytical'
s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Engineering\\Heat Transfer\\" + _l + "\\"
sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata
from VFModel import *
from Vector import *
from diffusion_analytical import *

model = VFModel()
nx = 30
model.setMaterial(K=lambda T:1.0-0.2*T**2+0.01*T,rho=1.0,Cp=lambda T:1.0+0.1*T)
model.setMesh(Lx=[1.0],nx=[nx],coordinates=RectangularSystem())

model.setInitialConditions(T0=1.0)
model.setBoundaryConditions(BoundaryIndex.EAST,NewtonBoundaryConditionData(h=1.0,Tfar=0.0))

model.RunTime(1.0,1e-4)
#print model.mesh(0,0,0).dV
#for i in model.mesh(0,0,0).boundaries:
#    print i.dA_vec
#    print i.dA
Temp = []
analytical = []
b = Rectangular1DNewton(1.0)
for i in xrange(nx):
    Temp.append((model.mesh(i,0,0).P[0], model.mesh(i,0,0).dTdt))
    analytical.append((model.mesh(i,0,0).P[0],b.dTdt(model.mesh(i,0,0).P[0],1.0)))
x,y = zip(*Temp)
u,v = zip(*analytical)
plt.plot(x,y)
plt.plot(u,v,'-r')
plt.xlabel('x')
plt.ylabel('T')
plt.show()

def plot_surface():
    Temp = []
    for i in xrange(nx):
        for j in xrange(nx):
            Temp.append((model.mesh.vfs[i][j][0].P[0], model.mesh.vfs[i][j][0].P[1], model.mesh.vfs[i][j][0].T))

    x,y,z = zip(*Temp)
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_line():
    Temp = []
    for i in xrange(nx):
        Temp.append((model.mesh.vfs[i][0][0].P[0], model.mesh.vfs[i][0][0].T))

    x,y = zip(*Temp)
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('T')
    plt.show()

#plot_line()
