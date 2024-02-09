import numpy as np 
import copy

if __name__=='__main__':
    h = 50.0
    L = 0.1
    K = 35.0
    rho = 7800.0
    Cp = 460.0

    T0 = 1260.0
    Tamb = 25.0
    stefan_boltzmann = 5.6703e-8

    T = T0*np.ones(shape=(100,))
    dt = 1e-3

    tf = 2.0*3600.0
    t = 0.0
    
    Bi = h*L/K
    print(Bi)
    exit()
    while(t < tf):
        Tlast = copy.copy(T)
        T[0] = (1.0 + Bi)
        #T[1:] = Tlast[1:] - 2.0*
        t += dt


