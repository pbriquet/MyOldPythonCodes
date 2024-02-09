from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm
from hardnessmodel import *
from HB_Data import *

def f(X, m, D0, QD, Qinf):
    t,T = X
    return HardnessModel.functional(m,D0,QD,Qinf).tau(t,T)

ln_t_min, ln_t_max = [2.0, 8.0]
T_min, T_max = [500.0, 700.0]
n_t, n_T = [10,10]
Nsurf_t, Nsurf_T = [100,100]
a0 = [1.5, 0.35, 500.0,2000.0]
a_true = [0.5, 0.28, 1000.0, 4.1e3]

noise_amplitude = 0.05

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


t = np.logspace(ln_t_min, ln_t_max, num=n_t, base=2.0)
T = np.linspace(T_min,T_max,n_T)
mesh = np.meshgrid(t,T,sparse=False,indexing='ij')
tv, Tv = mesh
tt = tv.flatten()
TT = Tv.flatten()

z = f((tt,TT), a_true[0], a_true[1], a_true[2], a_true[3]) + noise_amplitude*np.random.randn(len(t)*len(T))

kwarg = [('loss', 'cauchy'),('f_scale',0.1)]
p, conv = curve_fit(f, (tt,TT), z, a0, method='lm')
plt.hold(True)

x_surf, y_surf = np.meshgrid(np.logspace(ln_t_min, ln_t_max, num=Nsurf_t, base=2.0),np.linspace(T_min,T_max,Nsurf_T), sparse=False,indexing='ij')
z_surf = f((x_surf,y_surf), p[0],p[1], p[2],p[3])
ax.plot_surface(x_surf,y_surf,z_surf, cmap=cm.hot)
ax.scatter(tt,TT,z)


ax.set_xlabel('t')
ax.set_ylabel('T')
ax.set_zlabel('Tau')
print p
#plt.plot(x,map(lambda t,T: f(result.x,t,T),x,y))
plt.show()
