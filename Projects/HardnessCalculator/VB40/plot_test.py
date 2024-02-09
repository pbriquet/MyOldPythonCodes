from math import *
import numpy as np
import matplotlib.pyplot as plt
from HB_Data import *
from hardnessmodel import *
from scipy import optimize
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib import cm
from hardnessmodel import *

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

f = []
f.append(open(os.path.join(__location__,"c82 HB.dat"),'r'))

l = []
hb_data = []

k = 0
for archive in f:
    hb_data.append(HB_Data())
    l.append([])
    for line in archive:
        l[len(l)- 1].append(line.split())
    hb_data[len(hb_data) - 1].readData(l[len(l) - 1])


print hb_data[0].data

ln_t_min, ln_t_max = [0.5, 5.5]
n_t, n_T = [50,50]
T_min, T_max = [500.0, 700.0]
Nsurf_t, Nsurf_T = [100,100]

a0 = [5.19148706e-01,3.97545668e+02,5.50929328e+04,4.43595129e+03]

Temp = [580.0,620.0,640.0]
time = np.logspace(ln_t_min, ln_t_max, num=n_t, base=2.0)

h = HardnessModel.functional(a0[0],a0[1],a0[2],a0[3])

t_array = []

t_T_array = []

for T in Temp:
    for t in time:
        t_T_array.append((T,t))

HB_t_array = []

for T in Temp:
    x,y = hb_data[0].getArrayTimeHardness(T)
    y = map(lambda g: g, y)
    for m in y:
        t_array.append(x)
        HB_t_array.append(m)
    plt.scatter(x,y)
    z = map(lambda g: h.H(g,T),time)
    plt.plot(time,z)

#plt.xscale('log',basex=2)
plt.xlabel('t (h)')
plt.ylabel('HB')
plt.show()
