from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm


dim = 3
kronecker = np.zeros((3,3),int)
for i in range(dim):
    kronecker[i,i] = 1

print(kronecker[0])
L0 = np.array([0.0,0.0,-math.pi])    # Initial Coordinate
Lx = np.array([10.0,2*math.pi,math.pi])    # Final Coordinate
nx = [6,6,6]  # Mesh
dx =  np.array([x/y for x, y in zip(map(float, (Lx-L0)), map(int, nx))])   # Pythonic way for dividing one list for another list
grid = np.indices(nx)   # Mesh Grid of Coordinates
deltas = np.array([grid[i]*dx[i] for i in range(dim)])
print(grid.shape)
# Coordinate System Position

p = np.array([L0[i] + 0.5*dx[i] + deltas[i] for i in range(dim)])
x_plus = np.array([[[p[i] + (-1)**u*dx[j]/2.0*kronecker[i,j] for u in range(2)] for j in range(dim)] for i in range(dim) ])
x_minus = np.array([[[p[i] - (-1)**u*dx[j]/2.0*kronecker[i,j] for u in range(2)] for j in range(dim)] for i in range(dim) ])

print(x_plus.shape)
tranformation = lambda x: np.array([x[0]*np.cos(x[1]),x[0]*np.sin(x[1]),x[2]])
tranformation = lambda x: np.array([x[0]*np.sin(x[1])*np.cos(x[2]),x[0]*np.cos(x[1])*np.cos(x[2]),x[0]*np.sin(x[2])])
p = tranformation(p)
x_minus = tranformation(x_minus)
x_plus = tranformation(x_plus)
#p_real = np.array([tranformation(p[i]) for i in range(dim)])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(p[0],p[1],p[2],marker='o',s=20)
ax.scatter(x_plus[0],x_plus[1],x_plus[2],marker='X',s=20)
ax.scatter(x_minus[0],x_minus[1],x_minus[2],marker='>',s=20)
#ax.set_xlim(0.0,Lx[0])
#ax.set_ylim(0.0,Lx[1])
#ax.set_zlim(0.0,Lx[2])
plt.show()

'''
nx = 31
ny = 31
nt = 17
nu = .05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))  # create a 1xn vector of 1's
un = np.ones((ny, nx))

###Run through nt timesteps
def diffuse(nt):
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2  
    
    for n in range(nt + 1): 
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)
    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    

if __name__=='__main__':
    diffuse(10)
    diffuse(20)
    diffuse(40)

    plt.show()
'''