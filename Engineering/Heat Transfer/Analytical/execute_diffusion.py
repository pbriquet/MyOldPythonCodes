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
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm
from scipy.special import jv,cotdg
from enum import IntEnum
from enum import Enum
from NewtonRaphson import *
from diffusion_analytical import *


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

c = CylinderNewton(1.0,n=30)
print c.eigen
tmin = 0.01
tmax = 0.4
rmin = 0.0
rmax = 1.0
X, Y, Z = c.Surface_T(tmin,tmax,rmin,rmax)
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X,Y,Z, cmap=cm.inferno,alpha=0.7)
levels_iso = np.arange(-0.1,1.1,0.05)
levels = np.arange(-0.01,1.01,0.01)
CS = plt.contour(Z, levels_iso,
                 origin='lower',
                 fmt='%1.2f',
                 cmap=cm.binary,
                 linewidths=1,
                 extent=(tmin, tmax, rmin, rmax))
plt.clabel(CS, inline=1, fontsize=8)
CS1 = plt.contourf(Z, levels,
                 origin='lower',
                 fmt='%1.2f',
                 cmap=cm.summer,
                 extent=(tmin, tmax, rmin, rmax))
plt.colorbar(CS1)
plt.xlabel('t*')
plt.ylabel('r*')
#ax.set_xlabel('t*')
#ax.set_ylabel('r*')
#ax.set_zlabel('T*')

plt.show()
            