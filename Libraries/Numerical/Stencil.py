from math import *
import numpy as np
import os
from scipy.optimize import curve_fit

def stencil(yn,xn,x):
    if(len(xn)==5):
        return fpstencil(yn,xn,x)
    elif(len(xn)==3):
        return tpstencil(yn,xn,x)
    elif(len(xn)==2):
        return twopstencil(yn,xn,x)

def twopstencil(y2,x2,x):
    return (y2[1] - y2[0])/(x2[1] - x2[0])

def tpstencil(y3,x3,x):
    h = []

    for j in range(1,len(x3)):
        h.append(x3[j] - x3[j-1])

    h.reverse()
    h.append(0) # Corrigir index
    h.reverse()

    s = (x - x3[1])/h[1]

    l = []
    tmp = (2.0*s*h[1] - h[2])/(h[1]*(h[1] + h[2]))
    l.append(tmp)
    tmp = -((2.0*s + 1.0)*h[1] - h[2])/h[1]/h[2]
    l.append(tmp)
    tmp = ((2.0*s+1.0)*h[1])/h[2]/(h[1]+h[2])
    l.append(tmp)

    return l[0]*y3[0] + l[1]*y3[1] + l[2]*y3[2]

def fpstencil(y5,x5,x):
    h = []

    for j in range(1,len(x5)):
        h.append(x5[j] - x5[j-1])

    h.reverse()
    h.append(0) # Corrigir index
    h.reverse()

    s = (x - x5[1])/h[1]
    H2 = h[1] + h[2] + h[3] + h[4]
    l = []
    tmp = ((2.0*s*h[1] - h[2])*(s*h[1] - h[2] - h[3])*(s*h[1] - H2 + h[1]) + s*h[1]*(s*h[1] -h[2])*(2.0*s*h[1] - 2.0*h[2] - 2.0*h[3] - h[4]))/(h[1]*(h[1]+h[2])*(h[1] + h[2] + h[3])*(h[1] + h[2] + h[3] + h[4]))
    l.append(tmp)
    tmp = -((2.0*s*h[1] + h[1] - h[2])*(s*h[1] - h[2] - h[3])*(s*h[1] - H2 + h[1]) + h[1]*(s+1.0)*(s*h[1] -h[2])*(2.0*s*h[1] - 2.0*h[2] - 2.0*h[3] - h[4])) / (h[1]*h[2]*(h[1]+h[3])*(h[2] + h[3] + h[4]))
    l.append(tmp)
    tmp = ((2.0*s*h[1] + h[1])*(s*h[1] - h[2] - h[3])*(s*h[1] - H2 + h[1]) + h[1]*h[1]*(s*s + s)*(2.0*s*h[1] - 2.0*h[2] - 2.0*h[3] - h[4])) / ((h[1] + h[2])*h[2]*h[3]*(h[3] + h[4]))
    l.append(tmp)
    tmp = -((2.0*s*h[1] + h[1])*(s*h[1] - h[2])*(s*h[1] - H2 + h[1]) + h[1]*h[1]*(s*s + s)*(2.0*s*h[1] - 2.0*h[2] - h[3] - h[4]))/((h[1]+h[2]+h[3])*(h[2] + h[3])*h[3]*h[4])
    l.append(tmp)
    tmp = ((2.0*s + 1.0)*h[1]*(s*h[1] - h[2])*(s*h[1]-h[2]-h[3]) + h[1]*h[1]*(s*s + s)*(2.0*s*h[1] - 2.0*h[2] - h[3]))/((h[1]+h[2]+h[3]+h[4])*(h[2] + h[3] + h[4])*(h[3] + h[4])*h[4])
    l.append(tmp)

    return l[0]*y5[0] + l[1]*y5[1] + l[2]*y5[2] + l[3]*y5[3] + l[4]*y5[4]    

def derivative_arrays(y,x):
    dydx = []
    x2 = []
    for i in xrange(len(x)):
        _x = []
        _y = []
        if(i==0):
            _x = [x[i],x[i+1]]
            _y = [y[i],y[i+1]]
        elif(i == len(x)-1):
            _x = [x[i-1],x[i]]
            _y = [y[i-1],y[i]]
        elif(i == 1 or i == len(x) - 2):
            _x = [x[i-1],x[i],x[i+1]]
            _y = [y[i-1],y[i],y[i+1]]
        else:
            _x = [x[i-2],x[i-1],x[i],x[i+1],x[i+2]]
            _y = [y[i-2],y[i-1],y[i],y[i+1],y[i+2]]

        dydx.append(stencil(_y,_x,x[i]))
        x2.append(x[i])

    return np.array(x2),np.array(dydx)

def smooth_derivative(x,y,n=1):
    N = len(x)
    xd = []
    yyd = []
    for i in range(n, N-n, 1):
        p = np.polyfit(x[i-n:(i+n+1)], y[i-n:(i+n+1)], 1)
        xd.append(x[i])
        yyd.append(p[0])
    return np.array(xd), np.array(yyd)
