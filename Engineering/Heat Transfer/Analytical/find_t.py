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

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def f(x, a, b):
    return a*x + b

output = open(os.path.join(__location__,'coefficients.txt'),'w')
output2 = open(os.path.join(__location__,'data.txt'),'w')

def calculate_line_t(Bimin, Bimax, n_Biot, Tdim):
    
    c = []
    line = []
    invBiotmin = 1.0/Bimax
    invBiotmax = 1.0/Bimin
    log_biotmin = log(Bimin,2)
    log_biotmax = log(Bimax,2)
    print log_biotmin
    print log_biotmax
    #Biot = np.linspace(Bimin,Bimax,n_Biot)
    Biot = np.logspace(log_biotmin,log_biotmax,num=n_Biot,base=2,endpoint=True)
    #Biot = map(lambda x: 2.0**x,log_Biot)
    invBiot = map(lambda x: 1.0/x,Biot)
    print Biot
    for i in xrange(n_Biot):
        print '1.0/Bi = ' + str(invBiot[i])
        c.append(CylinderNewton(1.0/invBiot[i]))
        t = c[i].t(0.0,Tdim)
        line.append((invBiot[i],t))
        
    x,y = zip(*line)
    log_x = np.log(x)
    log_y = np.log(y)
    plt.plot(x,y)
    plt.scatter(x,y)
    kwarg = [('loss', 'cauchy'),('f_scale',0.1)]
    a0 = [1.0, 0.5]
    p, conv = curve_fit(f, x, y, a0, method='lm')
    plt.hold(True)
    p_sigma = np.sqrt(np.diag(conv))
    output.write(str(Tdim) + '\t' + str(p[0]) + '\t' + str(p[1]) + '\n')
    xy_line = []
    for i in xrange(n_Biot):
        xy_line.append((x[i], f(x[i],p[0],p[1])))

    x_line, y_line = zip(*xy_line)
    #plt.plot(x_line,y_line)
    
def calculate_line_t_file(Bimin, Bimax, n_Biot, Tdim):
    c = []
    line = []
    invBiotmin = 1.0/Bimax
    invBiotmax = 1.0/Bimin
    
    log_biotmin = log(Bimin,2)
    log_biotmax = log(Bimax,2)
    print log_biotmin
    print log_biotmax
    #Biot = np.linspace(Bimin,Bimax,n_Biot)
    Biot = np.logspace(log_biotmin,log_biotmax,num=n_Biot,base=2,endpoint=True)
    #Biot = map(lambda x: 2.0**x,log_Biot)
    invBiot = map(lambda x: 1.0/x,Biot)

    output2.write('T = ' + str(Tdim) + '\n')
    for i in xrange(n_Biot):
        #print '1.0/Bi = ' + str(invBiot[i])
        c.append(CylinderNewton(1.0/invBiot[i]))
        t = c[i].t(0.0,Tdim)
        line.append((invBiot[i],t))
        
        output2.write(str(invBiot[i]) + '\t' + str(t) + '\n')
    x,y = zip(*line)

    kwarg = [('loss', 'cauchy'),('f_scale',0.1)]
    a0 = [1.0, 0.5]
    p, conv = curve_fit(f, x, y, a0, method='lm')
    output.write(str(Tdim) + '\t' + str(p[0]) + '\t' + str(p[1]) + '\n')


def calculate_surface_t(Bimin,Bimax,n_Biot,Tdimmin,Tdimmax,n_Tdim):
    c = []
    surface = []
    Tdim = np.linspace(Tdimmin,Tdimmax,n_Tdim)
    Biot = np.linspace(Bimin,Bimax,n_Biot)
    for i in xrange(n_Biot):
        c.append(CylinderNewton(Biot[i]))
        line = []
        for j in xrange(n_Tdim):
            line.append(c[i].t(0.0,Tdim[j]))
            surface.append((Tdim[j],Biot[i],line[j]))
    x,y,z = zip(*surface)

    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        
    data = np.array(surface)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)
    levels_iso = np.arange(0.1,5.0,0.05)
    levels = np.arange(0.1,5.0,0.01)
    CS = plt.contour(grid_z, levels_iso,
                     origin='lower',
                     fmt='%1.2f',
                     cmap=cm.Spectral,
                     linewidths=1,
                     extent=(min(x), max(x), min(y), max(y)))
    plt.clabel(CS, inline=1, fontsize=8)
    CS1 = plt.contourf(grid_z, levels,
                     origin='lower',
                     fmt='%1.2f',
                     cmap=cm.inferno,
                     extent=(min(x), max(x), min(y), max(y)))
    plt.colorbar(CS1)
    ax.set_xlabel('T*')
    ax.set_ylabel('Bi')
    plt.xlabel('T*')
    plt.ylabel('Bi')

    plt.show()


#calculate_surface_t(1.0,2.0,5,0.1,0.3,5)
#calculate_line_t(0.5,10.0,20,0.05)
#calculate_line_t(0.5,10.0,20,0.1)

dT = np.arange(0.001,0.01,0.001)
for i in dT:
    print "dT = " + str(i)
    calculate_line_t_file(0.1,20.0,50,i)
#calculate_line_t(0.1,20.0,50,0.01)
#calculate_line_t(0.1,20.0,50,0.02)
#plt.xlabel('1/Bi')
#plt.ylabel('t*')
#plt.show()
output.close()
output2.close()