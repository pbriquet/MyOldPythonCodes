import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from math import *
import numpy as np
import os
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
import re


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

numbers = "1.28E+03 -4.5e-05 0.001"
scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = re.compile(scientific_match + '|' + float_match)
# [-+] faz aceitar tanto positivo quanto negativo no inicio, porem, "?" coloca como opcional nos casos que nao tenham.
# O [\d] faz aceitar um digito no inicio da notacao cientifica. + adiciona o . (\.) no seguinte, porem deixa opcinal esta parte caso o numero seja inteiro.
# Em seguida o [\d]* permite inumeros numeros nas casas decimais posteriores.
# [Ee] permite que a notacao cientifica esteja com e ou E

def f(x,a,b):
    return a*x + b

output = open(os.path.join(__location__,'coefficients.txt'),'r')
output2 = open(os.path.join(__location__,'data.txt'),'r')
data = []
data2 = []
data3 = []

scientific = re.compile('[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?')

for line in output:
    l_match = [float(x) for x in re.findall(numbers_match,line)]
    data3.append((l_match[0],l_match[1],l_match[2]))

for line in output2:
    if re.match('^T',line):
        l_match = [float(x) for x in re.findall(numbers_match,line)]
        dT = l_match[0]
        data2.append((dT,[]))
    else:
        l_match = [float(x) for x in re.findall(numbers_match,line)]
        invBiot = l_match[0]
        t = l_match[1]
        data.append((dT,invBiot,t))
        data2[len(data2)-1][1].append((invBiot,t))

x,y,z = zip(*data)
fig = plt.figure()
#ax = fig.gca(projection='3d')
#grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
#grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
print len(data2)

plots = range(len(data2))
adT = []
bdT = []
for j in plots:
    X,Y = zip(*data2[j][1])
    a = data3[j][1]
    b = data3[j][2]
    adT.append((log(data3[j][0]),a))
    bdT.append((log(data3[j][0]),b))
    Y_linear = []
    for i in X:
        Y_linear.append(f(i,a,b))
    log_X = np.log(X)
    log_Y = np.log(Y)
    log_Ylinear = np.log(Y_linear)
    plt.scatter(X,Y)
    plt.plot(X,Y)
x,y = zip(*adT)
u,w = zip(*bdT)


kwarg = [('loss', 'cauchy'),('f_scale',0.1)]
a0 = [-1.0, 0.5]
p, conv = curve_fit(f, x, y, a0, method='lm')
print p[0], p[1]
p, conv = curve_fit(f, u, w, a0, method='lm')
print p[0], p[1]

#plt.plot(x,y,'-r')
#plt.scatter(x,y)
#plt.scatter(u,w)
#plt.plot(u,w,'-b')
plt.show()