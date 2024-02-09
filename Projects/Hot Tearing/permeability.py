import numpy as np
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
from matplotlib import cm

def Kozeny_Carman(fs,lambda2):
    return lambda2**2*(1.0-fs)**3/(180.0*fs**2)

if __name__=='__main__':
    mu = 1e-3
    fs = np.linspace(0.7,0.999,num=1000)
    lambda2 = np.logspace(-6,-3,num=1000)
    x,y = np.meshgrid(fs,lambda2)
    z = -np.log(Kozeny_Carman(x,y)/mu/(1.0-fs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cont = ax.contourf(x,y,z,cmap=cm.inferno)
    ax.set_yscale('log')
    cb = plt.colorbar(cont)
    ax.set_title(r'$-\log(\frac{K}{\mu\epsilon_l})=-\log(\frac{v_l}{-(dp/dx)})$',fontsize=15)
    ax.set_xlabel(r'$\epsilon_s$')
    ax.set_ylabel(r'$\lambda_2$')
    #cb.ax.set_title(r'$-log(\frac{K}{\mu\epsilon_l})')
    #ax.set_zscale('log')
    plt.show()