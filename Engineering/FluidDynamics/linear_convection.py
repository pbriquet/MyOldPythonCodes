import numpy as np                  #here we load numpy
import matplotlib.pyplot as plt      #here we load matplotlib
import time, sys  

nx = 81  # try changing this number from 41 to 81 and Run All ... what happens?
dx = 2 / (nx-1)
nt = 100    #nt is the number of timesteps we want to calculate
dt = .005  #dt is the amount of time each timestep covers (delta t)
c = 1      #assume wavespeed of c = 1
u = np.ones(nx)      #numpy function ones()
u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s

un = np.ones(nx) #initialize a temporary array
plt.plot(np.linspace(0, 2, nx), u)
for n in range(nt):  #loop for values of n from 0 to nt, so it will run nt times
    un = u.copy() ##copy the existing values of u into un
    for i in range(1, nx): ## you can try commenting this line and...
    #for i in range(nx): ## ... uncommenting this line and see what happens!
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])


plt.plot(np.linspace(0, 2, nx), u)
plt.show()