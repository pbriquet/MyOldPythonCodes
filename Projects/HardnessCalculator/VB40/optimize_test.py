from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm


def f(X, a, b, c):
    x,y = X
    return a*x*x + b*y*y + c

x_min, x_max = [0.0, 2.0]
y_min, y_max = [0.0, 2.0]
n_x, n_y = [40,40]
Nsurf_x, Nsurf_y = [300,300]
a0 = [20.0, 0.1, 0.2]
a_true = [250.0, 100.0, 30.0]

noise_amplitude = 50.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


x = np.linspace(x_min,x_max,n_x)
y = np.linspace(y_min,y_max,n_y)
mesh = np.meshgrid(x,y,sparse=False,indexing='ij')
xv, yv = mesh
XX = xv.flatten()
YY = yv.flatten()

z = f((XX,YY), a_true[0], a_true[1], a_true[2]) + noise_amplitude*np.random.randn(len(x)*len(y))

p, conv = curve_fit(f, (XX,YY), z, a0)
plt.hold(True)

x_surf, y_surf = np.meshgrid(np.linspace(x_min,x_max,Nsurf_x),np.linspace(y_min,y_max,Nsurf_y), sparse=False,indexing='ij')
z_surf = f((x_surf,y_surf), p[0],p[1], p[2])
ax.plot_surface(x_surf,y_surf,z_surf, cmap=cm.hot)
ax.scatter(XX,YY,z)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
print p
#plt.plot(x,map(lambda t,T: f(result.x,t,T),x,y))
plt.show()
