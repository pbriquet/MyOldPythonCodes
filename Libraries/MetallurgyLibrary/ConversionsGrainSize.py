import pandas as pd
import os
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from matplotlib import cm
from scipy.optimize import curve_fit

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
R = 8.3144598
D0 = 7.403163088184348e-06
n = 0.25
A0, Q = [1.21201593e-01,4.58694959e+05]
def GrainDiamater(t,T):
    return np.power(np.power(D0,1.0/n) + A0*np.exp(-Q/R/T)*t,n)

# Tetrakaidecahedron relations. Cahn et al. (1955)
# Given a average grain size, it returns: 
#   - Number of Points per volume
#   - Total length of Lines per Volume
#   - Total Area of surfaces
def Tetrakaidecahedron(D):
    return 12.0/D**3, 8.5/D**2, 3.35/D

class ASTMConversion:
    @staticmethod
    def l_to_D(l):
        return 1.571*l
    @staticmethod
    def D_to_l(D):
        return D/1.571
    @staticmethod
    def l_to_G(l):
        return -3.288 - 6.643956*np.log10(l*1e3)
    @staticmethod
    def G_to_l(G):
        return np.power(10.0,-(G + 3.288)/6.643956)/1e3
    @staticmethod
    def D_to_G(D):
        return ASTMConversion.l_to_G(ASTMConversion.D_to_l(D))
class ASTM:
    def __init__(self,**kwargs):
        self.data = {'D':0.0,'N_A':0.0,'N_V':0.0,'N_L':0.0,'l':0.0}
        if('D' in kwargs):
            pass
def plot_surface():
    T = np.linspace(750.0 + 273.15,950.0 + 273.15,num=50)
    t = np.linspace(0*3600.0,32*3600.0,num=50)
    mesh = np.meshgrid(np.array(t),np.array(T),sparse=False,indexing='ij')

    C0,L0,S0 = Tetrakaidecahedron(D0)
    print(C0,L0,S0)
    G0 = ASTMConversion.D_to_G(D0)
    print(C0,L0,S0)
    print(G0)
    tv, Tv = mesh
    tt = np.array(tv.flatten())
    t_h = tv/3600.0
    TT = np.array(Tv.flatten())
    T_C = Tv - 273.15
    z_surf = GrainDiamater(tv,Tv)
    G_surf = ASTMConversion.D_to_G(z_surf)
    C_surf, L_surf, S_surf = Tetrakaidecahedron(z_surf)
    C,L,S = np.log10([C_surf/C0,L_surf/L0,S_surf/S0])

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(t_h,T_C,G_surf, cmap=cm.hot)
    #plt.colorbar(surf)
    contour = plt.contourf(t_h,T_C,G_surf, cmap=cm.rainbow)
    cb = plt.colorbar(contour)
    plt.xlabel('t (h)')
    plt.ylabel('T (C)')
    plt.xlim(1.0,32.0)
    plt.title('Tamanho de Grão - ASTM')
    cb.ax.set_title(r'G')
    #c = np.abs(G_surf)
    #sc = ax.scatter(t_h,T_C,G_surf,c=c,cmap=cm.rainbow,label='JMatPro')
    #plt.colorbar(sc)


    fig = plt.figure()
    contour = plt.contourf(t_h,T_C,C, cmap=cm.rainbow)
    cb = plt.colorbar(contour)
    plt.xlabel('t (h)')
    plt.ylabel('T (C)')
    plt.xlim(1.0,32.0)
    plt.title('C = Densidade de Pontos ' + r'$(1/m^3)$')
    cb.ax.set_title(r'$\log(C/C_0)$')
    fig = plt.figure()
    contour = plt.contourf(t_h,T_C,L, cmap=cm.rainbow)
    cb = plt.colorbar(contour)
    plt.xlabel('t (h)')
    plt.ylabel('T (C)')
    plt.xlim(1.0,32.0)
    plt.title('L = Densidade de Linha ' + r'$(m/m^3)$')
    cb.ax.set_title(r'$\log(L/L_0)$')
    fig = plt.figure()
    contour = plt.contourf(t_h,T_C,S, cmap=cm.rainbow)
    cb = plt.colorbar(contour)
    plt.xlabel('t (h)')
    plt.ylabel('T (C)')
    plt.xlim(1.0,32.0)
    plt.title('S = Densidade de Área ' + r'$(m^2/m^3)$')
    cb.ax.set_title(r'$\log(S/S_0)$')
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(tt,TT,C_surf, cmap=cm.hot)
    plt.colorbar(surf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(tt,TT,L_surf, cmap=cm.hot)
    plt.colorbar(surf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(tt,TT,S_surf, cmap=cm.hot)
    plt.colorbar(surf)
    '''
    plt.show()
def plot_lines():
    G0 = 12.0
    D0 = ASTMConversion.l_to_D(ASTMConversion.G_to_l(G0))
    C0,L0,S0 = Tetrakaidecahedron(D0)
    print(C0,L0,S0)
    G = np.linspace(4.0,G0,num=9)
    D = ASTMConversion.l_to_D(ASTMConversion.G_to_l(G))
    C,L,S = Tetrakaidecahedron(D)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(G,np.log10(C/C0),label=r'C/C$_0$')
    ax1.plot(G,np.log10(L/L0),label=r'L/L$_0$')
    ax1.plot(G,np.log10(S/S0),label=r'S/S$_0$')
    ax1.set_xlim(12.0,4.0)
    ax1.grid(b=True,linestyle='--')
    D_plot = [float("{0:.1f}".format(a*1e6)) for a in D][::-1]
    ax2.set_xticks(G)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(D_plot)
    ax1.set_xlabel('Tamanho de Grão ASTM')
    ax1.set_ylabel(r'log$_{10}(\phi/\phi_0)$')
    ax2.set_xlabel(r'Tamanho de Grão ($\mu$m)')
    ax1.legend()
    plt.tight_layout()
    plt.show()

def estimate_G_from_nv():
    nv = 3.82e15

    error = 1.0
    G0 = 12.0

    G = np.linspace(1.0,G0,num=10000)

    for _G in G:
        D = ASTMConversion.l_to_D(ASTMConversion.G_to_l(_G))
        C,L,S = Tetrakaidecahedron(D)
        if(C >= nv*(1.0 - error/100) and C <= nv*(1.0 + error/100)):
            print(_G)
    

if __name__=='__main__':
    estimate_G_from_nv()
