import numpy as np                  #here we load numpy
import matplotlib.pyplot as plt      #here we load matplotlib
import time, sys  

nx = 41
dx = 2 / (nx - 1)
nt = 100    #the number of timesteps we want to calculate
nu = 0.3   #the value of viscosity
sigma = .2 #sigma is a parameter, we'll learn more about it later
dt = sigma * dx**2 / nu #dt is defined using sigma ... more later!
step_plot = 10

u = np.ones(nx)      #a numpy array with nx elements all equal to 1.
u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s

un = np.ones(nx) #our placeholder array, un, to advance the solution in time

plt.plot(np.linspace(0, 2, nx), u)
for n in range(nt):  #iterate through time
    t = n*dt
    if(n%step_plot==0):
        plt.plot(np.linspace(0, 2, nx), u,label=f't={t:.2f}')
    un = u.copy() ##copy the existing values of u into un
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

plt.legend()
plt.show()