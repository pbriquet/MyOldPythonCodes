import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm
from scipy.special import jv,cotdg
from enum import IntEnum
from enum import Enum
from diffusion_analytical import *
from NewtonRaphson import *

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

Bimin = 0.1
Bimax = 20.0
n_Biot = 20
dTmin = 0.001
dTmax = 2.0
ddT = 0.001

c = []
surface = []
invBiotmin = 1.0/Bimax
invBiotmax = 1.0/Bimin
print invBiotmin, invBiotmax
invBiot = np.linspace(invBiotmin,invBiotmax,n_Biot)
Biot = np.linspace(Bimin,Bimax,n_Biot)
dT = np.arange(dTmin,dTmax,ddT)
dT_iso = np.arange(dTmin,dTmax, ddT*10)
tmin = 0.0
tmax = 10.0
dt = 0.001
print invBiot
t = np.arange(tmin,tmax,dt)

for i in xrange(n_Biot):
    print 'Biot = ' + str(1.0/invBiot[i])
    c.append(CylinderNewton(1.0/invBiot[i]))
    for time in t:
        surface.append((invBiot[i],time,c[i].T(0.0,time)))

x,y,z = zip(*surface)
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    
data = np.array(surface)

fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)

#im = plt.imshow(Z, interpolation='bilinear', origin='lower',cmap=cm.inferno)

CS1 = plt.contour(grid_z, dT_iso,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 cmap=cm.magma,
                 extent=(invBiotmin, invBiotmax, tmin, tmax))
plt.clabel(CS1, inline=1, fontsize=10)

CS = plt.contourf(grid_z, dT,
                 origin='lower',
                 fmt='%1.2f',
                 cmap=cm.inferno,
                 extent=(invBiotmin, invBiotmax, tmin, tmax))
plt.colorbar(CS)

plt.show()