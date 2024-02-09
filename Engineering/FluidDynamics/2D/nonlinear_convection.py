from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

Lx, Ly = [2.0,2.0]
nx, ny = [81,81]
nt = 100
dx, dy = [Lx/nx,Ly/ny]
sigma = .2
dt = sigma*dx
c_u = 1.0
c_v = 0.5

x = np.linspace(0,Lx,nx)
y = np.linspace(0,Ly,ny)

u = np.ones((ny, nx)) ##create a 1xn vector of 1's
v = np.ones((ny, nx))
un = np.ones((ny, nx))
vn = np.ones((ny, nx))

###Assign initial conditions
##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
##set hat function I.C. : v(.5<=x<=1 && .5<=y<=1 ) is 2
v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 4.0

for n in range(nt+1):
    un = u.copy()
    vn = v.copy()
    u[1:, 1:] = (un[1:, 1:] - 
                 (un[1:, 1:] * c_u * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
                  vn[1:, 1:] * c_v * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    v[1:, 1:] = (vn[1:, 1:] -
                 (un[1:, 1:] * c_u * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) -
                 vn[1:, 1:] * c_v * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]))
    
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1



fig = plt.figure(figsize=(11,7),dpi=100)
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
surfu = ax.plot_surface(X,Y,u[:],cmap=cm.hot)
plt.colorbar(surfu)

fig = plt.figure(figsize=(11,7),dpi=100)
ax = fig.gca(projection='3d')
surfv = ax.plot_surface(X,Y,v[:],cmap=cm.hot)
plt.colorbar(surfv)

plt.show()


