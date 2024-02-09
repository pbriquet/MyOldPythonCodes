
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.special import jv,cotdg
from matplotlib import cm


def rectangular(biot,x):
    return x*np.tan(x)-biot

def cylindrical(biot,x):
    return x*jv(1,x) - biot*jv(0,x)

log_invBiot = np.linspace(-1.0,0.0,num=2000)
inv_Biot = 10**log_invBiot
Biot = np.array([10**(-i) for i in log_invBiot])

x = np.linspace(0.0,100.0,num=2000)
y = Biot
X,Y = np.meshgrid(x,y)
z = rectangular(Y,X)

fig = plt.figure()
surf = plt.contourf(X,Y,z,cmap=cm.rainbow)
plt.contour(X,Y,z,[0.0],colors='black')
fig.colorbar(surf)
plt.show()
