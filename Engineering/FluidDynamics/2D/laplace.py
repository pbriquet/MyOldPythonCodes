import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy
from matplotlib import cm

##variable declarations
nx = 31
ny = 31
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)


##initial conditions



##plotting aids
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)



def plot2D(x, y, p):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()
def laplace_boundary_conditions(p):
    ##boundary conditions
    p[:, 0] = 2.0*x  # p = 0 @ x = 0
    p[:, -1] = y  # p = y @ x = 2
    p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
    p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1

def laplace2d(p, y, dx, dy, l1norm_target):  
    l1norm = 1
    pn = np.empty_like(p)
    boundary_conditions(p)
    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))
            
        laplace_boundary_conditions(p)
        l1norm = (np.sum(np.abs(p[:]-pn[:])))/np.sum(np.abs(pn[:]))
     
    return p

u = np.ones((ny,nx))
v = np.ones((ny,nx))
un = u.copy()
vn = v.copy()

comb = np.ones((ny,nx))
p = np.zeros((ny, nx))  # create a XxY vector of 0's
Lxi, Lxf = [0.5,1.0]
Lyi, Lyf = [0.5,1.0]
p = laplace2d(p, y, dx, dy, 1e-4)
plot2D(x, y, p)