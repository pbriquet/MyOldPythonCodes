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


def Biot(h,R,k):
    return h*R/k
def tau(R,alpha):
    return R**2/alpha/1e-6/3600.0

Bimax = 200.0
rmax = 1.0
rmin = 0.01
kmin = 1.0
kmax = 20.0
dr = 0.01
dk = 0.1
Bimin = 0.0
h_hold = 250.0
dBi_graf = 0.1
dBi_lines = 2.0
K = np.arange(kmin,kmax,dk)
R = np.arange(rmin,rmax,dr)
levels_Bi = np.logspace(0.0, 5.0, num=6, base=2.0)
levels = np.logspace(-10.0, 7.0, num=50, base=2.0)
X, Y = np.meshgrid(R,K)
Z = tau(X,Y)
fig = plt.figure()
#im = plt.imshow(Z, interpolation='bilinear', origin='lower',cmap=cm.inferno)

CS1 = plt.contour(Z, levels_Bi,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 cmap=cm.binary,
                 extent=(rmin, rmax, kmin, kmax))
#plt.clabel(CS1, inline=1, fontsize=10)
plt.clabel(CS1, inline=1, fontsize=8)
CS = plt.contourf(Z, levels,
                 origin='lower',
                 fmt='%1.2f',
                 cmap=cm.summer,
                 extent=(rmin, rmax, kmin, kmax))
plt.colorbar(CS)
plt.xlabel('R')
plt.ylabel('K')
plt.show()
