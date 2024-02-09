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

def delta_kj(x,xk,xj):
    return 1.0 - (la.norm(x - xk)/la.norm(xj - xk))

class Zones:
    def __init__(self,zones):
        self.zones = zones
    def Tfar(self, x):
        tmp = 0.0
        belongs = []
        index_belongs = []
        for i in xrange(len(self.zones)):
            belongs.append(self.zones[i].belongs(x))
            if(belongs[i]):
                index_belongs.append(i)
        if(len(index_belongs) > 1):
            #print index_belongs
            tmp += delta_kj(x,self.zones[index_belongs[0]].x_end, self.zones[index_belongs[1]].x_start)*self.zones[index_belongs[1]].Tfar
            tmp += delta_kj(x,self.zones[index_belongs[1]].x_start, self.zones[index_belongs[0]].x_end)*self.zones[index_belongs[0]].Tfar
        else:
            tmp += self.zones[index_belongs[0]].Tfar
        return tmp

class Zone:
    def __init__(self,x_start, x_end, Tfar):
        self.x_start = x_start
        self.x_end = x_end
        self.Tfar = Tfar
    def belongs(self,x):
        if(x >= self.x_start and x <= self.x_end):
            return True
        else:
            return False

    def __str__(self):
        return '[' + str(self.x_start) + ',' + str(self.x_end) + ']' + ', ' + str(self.Tfar)

delta = 0.1
dT = 0.2
T0 = 0.0
size = 1.0
n_zones = 6
zones = []
for i in xrange(n_zones):
    if(i == 0):
        zones.append(Zone(i*size,(i+1)*size + delta, T0))
    else:
        zones.append(Zone(i*size - delta,(i+1)*size + delta, T0 + i*dT))

z = Zones(zones)
x = np.linspace(0.0,n_zones*size - delta, num=200).tolist()
y = []

for i in xrange(len(x)):
    y.append(z.Tfar(x[i]))

plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('T*')
plt.show()