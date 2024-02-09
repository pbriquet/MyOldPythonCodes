import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm
import re, os
import pandas as pd

'''
Algoritmo dedicado para funções do artigo de Haider e Levenspiel (1989)
A função do coeficiente de atrito é ajustado em função do fator de forma (As/A_sph)
'''
class HaiderLevenspiel:
    sphere = {'K1':0.7554,'K2':0.8243}
    octahedron = {'K1':1.1272,'K2':0.9697}

    def __init__(self,shape='sphere',**kwargs):
        if(shape=='octa'):
            self.shape = HaiderLevenspiel.octahedron
        else:
            self.shape = HaiderLevenspiel.sphere

        self.rho_s = kwargs['rho_s']
        self.rho_f = kwargs['rho_f']
        self.g = kwargs['g']
        self.mu = kwargs['mu']

    def calculate(self,d_sph):
        d_str = d_sph*np.power( self.g*self.rho_f*(self.rho_s - self.rho_f)/self.mu**2, 1.0/3.0)
        u_str = self.haider_solution(d_str)
        u_t = u_str*np.power(self.rho_f**2/self.g/self.mu/(self.rho_s - self.rho_f),-1.0/3.0)
        return u_t

    
    def haider_solution(self,d_str):
        return np.power(np.power(18.0/d_str/d_str,self.shape['K2']) + np.power(3.0*self.shape['K1']/4.0/np.sqrt(d_str),self.shape['K2']),-1.0/self.shape['K2'])
    
    @staticmethod
    def u_str(d_str,phi):
        K1 = 3.1131 - 2.3252*phi
        return np.power(18.0/np.power(d_str,2) + 3.0*K1/4.0/np.sqrt(d_str),-1)
    @staticmethod
    def CD(shape_factor,Re):
        # 24/Re*(1 + A*Re^B) + C/(1+D/Re)
        A = np.exp(2.3288 - 6.4581*shape_factor + 2.4486*np.power(shape_factor,2.0))
        B = 0.0964 + 0.5565*shape_factor
        C = np.exp(4.905 - 13.8944*shape_factor + 18.4222*np.power(shape_factor,2.0) - 10.2599*np.power(shape_factor,3.0))
        D = np.exp(1.4681 + 12.2584*shape_factor - 20.7322*np.power(shape_factor,2.0) + 15.8855*np.power(shape_factor,3.0))
        return 24.0/Re*(1.0 + A*np.power(Re,B)) + C/(1.0 + D/Re)

    # Método para plotar gráfico C_D de Haider-Levenspiel baseado no artigo.
    @staticmethod
    def build_CD_map():
        Re = np.logspace(-2,6,num=100)
        shape_factors = np.array([1.0,0.906,0.846,0.806,0.67])
        disk_shape_factors = np.array([0.23,0.123,0.043,0.026])

        # Gerar cores para cada linha baseado no cmap
        cmap = plt.get_cmap('brg')
        colors = [cmap(i) for i in np.linspace(0, 1, len(shape_factors) + len(disk_shape_factors))]
        fig = plt.figure()
        # Plot de shape factors regulares
        for k,s in enumerate(shape_factors):
            y = HaiderLevenspiel.CD(s,Re)
            plt.plot(Re,y,color=colors[k],label=r'$\phi = $' + '{:0.3f}'.format(s))
        # Plot de shape factors para discos
        for k,s in enumerate(disk_shape_factors):
            y = HaiderLevenspiel.CD(s,Re)
            plt.plot(Re,y,color=colors[k + len(shape_factors)],linestyle='--',label=r'$\phi = $' + '{:0.3f}'.format(s))
        
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='major',color='darkgrey')
        plt.grid(which='minor',linestyle='--',color='lightgrey')
        plt.minorticks_on()
        plt.xlim(1e-2,1e6)
        plt.ylim(1e-1,1e5)
        plt.ylabel(r'$C_D$')
        plt.xlabel(r'$Re$')
        plt.legend()
    @staticmethod
    def solidfraction():
        solidfraction = np.logspace(-2,0,num=5)
        for s in solidfraction:
            rho_c = 2633.2*s + (1.0 - s)*2370.0
            settling = {'rho_s':rho_c,'rho_f':2370.0,'mu':1e-3,'g':9.81}
            h= HaiderLevenspiel(**settling)
            d = np.logspace(-6,-1,num=100)
            plt.plot(d,h.calculate(d))

        plt.grid(which='major',color='darkgrey')
        plt.grid(which='minor',linestyle='--',color='lightgrey')
        plt.minorticks_on()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    @staticmethod
    def build_map_analysis():
        shape_factors = np.array([1.0,0.906,0.846,0.806,0.67])
        disk_shape_factors = np.array([0.23,0.123,0.043,0.026])
        Re = np.logspace(-2,6,num=100)
        fig = plt.figure()
        cmap = plt.get_cmap('brg')
        colors = [cmap(i) for i in np.linspace(0, 1, len(shape_factors) + len(disk_shape_factors))]
        for k,s in enumerate(shape_factors):
            Cd = HaiderLevenspiel.CD(s,Re)
            x = np.power(3.0/4.0*Cd*np.power(Re,2),1.0/3.0)
            y = np.power(4.0*Re/3.0/Cd,1.0/3.0)
            plt.plot(x,y,color=colors[k],label=r'$\phi = $' + '{:0.3f}'.format(s))
        for k,s in enumerate(disk_shape_factors):
            Cd = HaiderLevenspiel.CD(s,Re)
            x = np.power(3.0/4.0*Cd*np.power(Re,2),1.0/3.0)
            y = np.power(4.0*Re/3.0/Cd,1.0/3.0)
            plt.plot(x,y,color=colors[k + len(shape_factors)],label=r'$\phi = $' + '{:0.3f}'.format(s))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='major',color='darkgrey')
        plt.grid(which='minor',linestyle='--',color='lightgrey')
        plt.minorticks_on()
        plt.xlim(1e0,1e4)
        plt.ylim(1e-2,1e2)
        plt.ylabel(r'$u_*$')
        plt.xlabel(r'$d_*$')
        plt.legend()
    @staticmethod
    def build_map():
        d_str = np.logspace(0,4,num=100)
        shape_factors = np.array([1.0,0.906,0.806,0.67])
        disk_shape_factors = np.array([0.23,0.123,0.043,0.026])
        cmap = plt.get_cmap('brg')
        fig = plt.figure()
        colors = [cmap(i) for i in np.linspace(0, 1, len(shape_factors) + len(disk_shape_factors))]
        for k,s in enumerate(shape_factors):
            y = HaiderLevenspiel.u_str(d_str,s)
            plt.plot(d_str,y,color=colors[k],label=r'$\phi = $' + '{:0.3f}'.format(s))
        for k,s in enumerate(disk_shape_factors):
            y = HaiderLevenspiel.u_str(d_str,s)
            plt.plot(d_str,y,color=colors[k + len(shape_factors)],linestyle='--',label=r'$\phi = $' + '{:0.3f}'.format(s))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='major',color='darkgrey')
        plt.grid(which='minor',linestyle='--',color='lightgrey')
        plt.minorticks_on()
        plt.xlim(1e0,1e4)
        plt.ylim(1e-2,1e2)
        plt.ylabel(r'$u_*$')
        plt.xlabel(r'$d_*$')
        plt.legend()

if __name__=='__main__':
    HaiderLevenspiel.build_map_analysis()
    HaiderLevenspiel.build_map()
    plt.show()