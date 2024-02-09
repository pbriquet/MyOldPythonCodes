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


rmin = 0.05
rmax = 1.0
kmin = 10.0
kmax = 60.0

K = np.arange(10.0,60.0,0.5)
R = np.arange(0.05,1.0, 0.05)


Bimin = 1e-3
Bimax = 1e-1
dBi = 1e-2
tau_min = 0.001
tau_max = 3.0
dtau = 0.001

X,Y = np.meshgrid(R,K)
Bi_h = np.log(X/Y*1e3)
tau = np.log(X**2/Y*1e3)

fig = plt.figure()
#im = plt.imshow(Z, interpolation='bilinear', origin='lower',cmap=cm.inferno)
levels_Bi = np.linspace(-2.0,10.0,num=25)
levels_tau =  np.linspace(-2.0,10.0,num=25)
CS1 = plt.contour(Bi_h, levels_Bi,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 cmap=cm.summer,
                 extent=(rmin, rmax, kmin, kmax))
#plt.clabel(CS1, inline=1, fontsize=10)
plt.clabel(CS1, inline=1, fontsize=8)
CS2 = plt.contour(tau, levels_tau,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 cmap=cm.cool,
                 extent=(rmin, rmax, kmin, kmax))
plt.clabel(CS2, inline=1, fontsize=8)
plt.xlabel('R')
plt.ylabel('K')
plt.show()
