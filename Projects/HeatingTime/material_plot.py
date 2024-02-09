import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

K_min = 11
K_max = 21
R_min = 1e-1
R_max = 3e-1

K_range = [K_min,K_max]
R_range = [R_min,R_max]

n = 100
k = np.linspace(5.0,80.0,num=n)
r = np.linspace(0.05,0.5,num=n)
K,R = np.meshgrid(k,r)
Omega1 = K/R # K constante
Omega2 = K/R**2    # R constante
fig = plt.figure()
surf3 = plt.contourf(K,R,np.log(Omega2[:]),cmap=cm.winter)
surf1 = plt.contourf(K,R,np.log(Omega1[:]),cmap=cm.hot)
surf2 = plt.contour(K,R,np.log(Omega2[:]),cmap=cm.winter)
plt.plot(k,[R_range[0]]*n,linestyle='--',color='black')
plt.plot(k,[R_range[1]]*n,linestyle='--',color='black')
plt.plot([K_range[0]]*n,r,linestyle='--',color='black')
plt.plot([K_range[1]]*n,r,linestyle='--',color='black')
clb1 = fig.colorbar(surf1)
clb2 = fig.colorbar(surf3)
clb1.ax.set_title(r'$\ln(\Omega_1)$')
clb2.ax.set_title(r'$\ln(\Omega_2)$')
plt.xlabel(r'$K (W/m/K)$')
plt.ylabel(r'$R (m)$')
'''
fig = plt.figure()
x = np.linspace(0.05,3.0,num=n)
y = np.linspace(0.05,3.0,num=n)
X,Y = np.meshgrid(x,y)
Z1 = X**2/Y
Z2 = X/Y
surf1 = plt.contourf(X,Y,Z1[:],cmap=cm.hot)
surf3 = plt.contour(X,Y,Z1[:],levels=K_range,colors='gray',linestyles='--')
surf2 = plt.contour(X,Y,Z2[:],cmap=cm.cool)
surf4 = plt.contour(X,Y,Z2[:],levels=R_range,colors='black',linestyles='--')
plt.xlabel('Ω1 = R/K')
plt.ylabel('Ω2 = R^2/K')
plt.title('Isoconductivity / Isolength (log)')
#fig.colorbar(surf1)
#fig.colorbar(surf2)
#surf2 = ax.plot_surface(X,Y,Z2[:],cmap=cm.hot)
'''
plt.show()
