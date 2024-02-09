import numpy as np
import sympy
import matplotlib.pyplot as plt
from sympy import init_printing
from sympy.utilities.lambdify import lambdify
init_printing(use_latex=True)

def plot(x,u,u_analytical):
    plt.plot(x,u, marker='o', lw=2, label='Computational')
    plt.plot(x, u_analytical, label='Analytical')

x, nu, t = sympy.symbols('x nu t')
phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
       sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))

phiprime = phi.diff(x)
u = -2.0*nu*(phiprime/phi) + 4.0

ufunc = lambdify((t,x,nu),u)

nx = 101
nt = 100
nu = 0.07
Lx = 2.0*np.pi
dx = Lx/(nx - 1)
dt = dx * nu

x = np.linspace(0,Lx,nx)
un = np.empty(nx)
t = 0.0

u = np.asarray([ufunc(t,x0,nu) for x0 in x])



timer = 0
for n in range(nt):
    if(timer > 10):
        timer = 0
        u_analytical = np.asarray([ufunc(n*dt,xi,nu) for xi in x])
        plot(x,u,u_analytical)
    un = u.copy()
    for i in range(1,nx-1):
        u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 *\
                (un[i+1] - 2 * un[i] + un[i-1])
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 *\
                (un[1] - 2 * un[0] + un[-2])
    u[-1] = u[0]
    timer += 1

plt.figure(figsize=(11, 7), dpi=100)

plt.xlim([0, Lx])
plt.ylim([0, 10])
plt.legend()
plt.show()