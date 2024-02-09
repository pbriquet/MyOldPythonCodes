import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.optimize import least_squares
from HB_Data import *

def T_K(T_C):
    return T_C + 273.15


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

f = []
f.append(open(os.path.join(__location__,"c82 HB.dat"),'r'))
f.append(open(os.path.join(__location__,"c83 HB.dat"),'r'))

l = []
hb_data = []

k = 0
for archive in f:
    hb_data.append(HB_Data())
    l.append([])
    for line in archive:
        l[len(l)- 1].append(line.split())
    hb_data[len(hb_data) - 1].readData(l[len(l) - 1])

Temp = [580.0,620.0,640.0]

H0 = 500.0
Hgamma = 100.0

Hfar = [305.0,285.0,263.0]
T_lim = -273.15
time = [4.0,8.0,16.0,32.0]

ln_time = map(log,time)
T_K_inv = map(lambda x: 1/(T_K(x)-T_K(T_lim)), Temp)
ln_Hfar_nor = map(lambda x: log((x - Hgamma)/(H0-Hgamma)), Hfar)

t_array = []
HB_t_array = []

T_array = []
HB_T_array = []

for T in Temp:
    x,y = hb_data[0].getArrayTimeHardnessAverage(T)
    t_array.append(x)
    HB_t_array.append(y)

for t in time:
    x,y = hb_data[0].getArrayTemperatureHardnessAverage(t)
    T_array.append(x)
    HB_T_array.append(y) 

print("T_array = ", T_array)
print("HB_T_array = ", HB_T_array)
print("t_array = ", t_array)
print("HB_t_array = ", HB_t_array)

HB_t_array_norm = []
for i in range(len(t_array)):
    j = i
    tmp = map(lambda x: log(-log((x - Hfar[j])/(H0-Hfar[j]))), HB_t_array[j])
    HB_t_array_norm.append(tmp)

print("HB_t_array_norm = ", HB_t_array_norm)

HB_T_array_norm = []
for i in range(len(T_array)):
    HB_T_array_norm.append([])
    for j in range(len(T_array[i])):
        tmp = log(-log((HB_T_array[i][j] - Hfar[j])/(H0-Hfar[j])))
        HB_T_array_norm[i].append(tmp)

print("HB_T_array_norm = ", HB_T_array_norm)

for i in range(len(HB_T_array_norm)):
    plt.plot(T_K_inv,HB_T_array_norm[i])

#p = np.polyfit(T_K_inv, ln_Hfar_nor, 1)


def Hardness(T,t,Q,Qb,D0,m,Hmin,H0,Tmax):
    Hfar = Hmin + (H0 - Hmin)*exp(-Qb/R/(T-Tmax))
    H_tmp = Hfar + (H0 - Hfar)*exp(-(D0*exp(-Q/R/T))**m)
    return H_tmp


#plt.plot(T_K_inv,ln_Hfar_nor)
#plt.plot(T_K_inv,map(lambda x: p[1] + p[0]*x,T_K_inv),"--g")
plt.show()

#print T_K_inv