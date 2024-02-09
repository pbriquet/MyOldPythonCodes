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
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def T0(Bi0,BiL):
    return 1.0/(1.0 + Bi0 + Bi0/BiL)

def TL(Bi0,BiL):
    return (1.0 + Bi0)/(1.0 + Bi0 + Bi0/BiL)

def G(Bi0,BiL):
    return TL(Bi0,BiL) - T0(Bi0,BiL)

Bi0_max = 10.0
BiL_max = 10.0

x = np.arange(0.1,Bi0_max,0.01)
y = np.arange(0.1,BiL_max,0.01)


# Make data.

X, Y = np.meshgrid(x, y)
Z_G = G(X,Y)
Z_T0 = T0(X,Y)
Z_TL = TL(X,Y)

isotherms = Y/X

isotherms2 = 8.0/X
# Plot the surface.
plt.figure()
#im = plt.imshow(Z, interpolation='bilinear', origin='lower',
 #               cmap=cm.inferno, extent=(-3, 3, -2, 2))
levels = np.arange(0.3,0.9,0.05)
levelsT0 = np.arange(0.0,0.5,0.05)
levelsTL = np.arange(0.6,1.0,0.05)
levels_iso = np.logspace(-1.0,1.0,num=11)
levels_iso2 = np.logspace(0.0,4.0,num=5,base=2)
iso = plt.contour(isotherms, levels_iso,
                 origin='lower',
                 linewidths=1,
                 colors='black',
                 fmt='%1.2f',
                 extent=(0, Bi0_max, 0, BiL_max))
plt.clabel(iso, inline=1, fontsize=8)
for c in iso.collections:
    c.set_dashes([(0, (2.0, 2.0))])

iso2 = plt.contour(isotherms2, levels_iso2,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 colors='gray',
                 extent=(0, Bi0_max, 0, BiL_max))
plt.clabel(iso2, inline=1, fontsize=8)
for c in iso2.collections:
    c.set_dashes([(0, (2.0, 2.0))])

CS = plt.contour(Z_G, levels,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 cmap=cm.rainbow,
                 extent=(0, Bi0_max, 0, BiL_max))
plt.clabel(CS, inline=1, fontsize=10)
plt.colorbar(CS,'G*')
CS1 = plt.contour(Z_T0, levelsT0,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 cmap=cm.winter,
                 extent=(0, Bi0_max, 0, BiL_max))
plt.clabel(CS1, inline=1, fontsize=8)
for c in CS1.collections:
    c.set_dashes([(0, (2.0, 2.0))])
plt.colorbar(CS1)
CS2 = plt.contour(Z_TL, levelsTL,
                 origin='lower',
                 linewidths=1,
                 fmt='%1.2f',
                 cmap=cm.autumn,
                 extent=(0, Bi0_max, 0, BiL_max))
plt.clabel(CS2, inline=1, fontsize=8)
for c in CS2.collections:
    c.set_dashes([(0, (2.0, 2.0))])
plt.colorbar(CS2)
plt.text(7.0, 7.0, 'BiL/Bi0', rotation=45)
plt.title('G*')
plt.xlabel('Bi0')
plt.ylabel('BiL')

# Customize the z axis.

#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#ax.set_xlabel('Bi0')
#ax.set_ylabel('BiL')
#ax.set_zlabel('G*')
# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()