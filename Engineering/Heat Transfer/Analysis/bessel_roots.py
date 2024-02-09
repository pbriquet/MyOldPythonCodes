import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from math import *
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm
from collections import Counter
from scipy.special import jv,cotdg
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def func(x,Bi):
    return x*jv(1,x)-Bi*jv(0,x)
def func2(x,Bi):
    return x - Bi*cotdg(x*180.0/pi)

Bimax = 20.0
rmax = 50.0
rmin = 0.01
Bimin = 0.0
Bi = np.arange(Bimin,Bimax,1.0)
x = np.arange(rmin,rmax,0.01)
levels = np.arange(-20.0,12.0,0.5)
#levels = np.append(levels,np.arange(1000.0,10000.0,1000.0))
X, Y = np.meshgrid(x, Bi)
Z = func(X,Y)
#fig = plt.figure()
#im = plt.imshow(Z, interpolation='bilinear', origin='lower',cmap=cm.inferno)

#CS1 = plt.contour(Z, [0.0],
#                 origin='lower',
#                 linewidths=1,
#                 fmt='%1.2f',
#                 cmap=cm.magma,
#                 extent=(rmin, rmax, Bimin, Bimax))
#plt.clabel(CS1, inline=1, fontsize=10)

#CS = plt.contourf(Z, levels,
#                 origin='lower',
#                 fmt='%1.2f',
#                 cmap=cm.inferno,
#                 extent=(rmin, rmax, Bimin, Bimax))
#plt.colorbar(CS)



#plt.title('J1(x)-Bi*J0(x)=0')
#plt.xlabel('x')
#plt.ylabel('Bi')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z, cmap=cm.inferno,alpha=0.7)

plt.show()