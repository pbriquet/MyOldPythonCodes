import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

parameters = dict(
    C=dict(K=0.34,m=-78.0,beta=-0.08),
    Si=dict(K=0.59,m=-17.1,beta=-0.087),
    Mn=dict(K=0.75,m=-3.32,beta=-0.014),
    S=dict(K=0.024,m=-30.4,beta=-0.09),
    P=dict(K=0.09,m=-27.1,beta=-0.084),
    Cu=dict(K=0.96,m=-1.7,beta=0.004),
    Cr=dict(K=0.76,m=-2.61,beta=-0.029),
    Ni=dict(K=0.94,m=-1.6,beta=0.005),
    Mo=dict(K=0.56,m=-3.25,beta=0.014),
    V=dict(K=0.93,m=-2.65,beta=0.045),
    W=dict(K=0.40,m=-0.5,beta=0.026)
)
betaT = -0.881e-3
Tref = 1536.0
Tliq0 = 1538.0
def Tliquidus(comp):
    _Tliq = Tliq0
    for k,C in comp.items():
        _Tliq += C*parameters[k]['m']
    return _Tliq

def liquid_density(comp,T):
    _drhos = {}
    _drhos['T'] = (T - Tref)*betaT
    _rho = (T-Tref)*betaT
    for k,C in comp.items():
        _drhos[k] = C*parameters[k]['beta']
        _rho += C*parameters[k]['beta']
    return _rho,_drhos

def scheil(comp,fl):
    f_comp = {}
    for k,C in comp.items():
        f_comp[k] = C*np.power(fl,parameters[k]['K'] - 1.0)
    return f_comp
    
if __name__=='__main__':
    '''
    comp = dict(
        C=0.41,
        Si=1.71,
        Mn=0.79,
        S=0.0013,
        P=0.0032,
        Cu=0.03,
        Cr=0.9,
        Ni=1.85,
        Mo=0.45,
        V=0.01,
        W=0.01
        )
    '''
    comp = dict(
        Mo=1.0
    )
    #comp = dict(Ni=0.5,C=0.3)
    eps_s = np.linspace(0.0,0.95,num=100)
    drho_C0_T = []
    T0 = Tliquidus(comp)
    rho_C0,_ = liquid_density(comp,T0)
    print('rho(C_0) = ' + str(rho_C0))
    print('T_liq(C_0) = ' + str(T0))
    scheil_comp = []
    Tliq = []
    drho = []
    rhos = []
    drhos = []
    for eps in eps_s:
        scheil_comp.append(scheil(comp,1.0 - eps))  # Calculate scheil composition in Liquid ( C_{L,j} = C_{0,j}*(1.0 - eps_s)^(1.0 - K_j))
        Tliq.append(Tliquidus(scheil_comp[-1]))     # Calculate Tliq with scheil composition
        rho,d = liquid_density(scheil_comp[-1],Tliq[-1])    # rho = Calculate density with Scheil Liquid Composition and T liquidus of liquid composition
        rho_C0_Tliq,_ = liquid_density(comp,Tliq[-1])       # rho_C0_Tliq = Calculate density with T liquidus of interdendritic liquid and C0
        drho.append(rho - rho_C0_Tliq)                      # Gather density difference between (rho - rho_C0_Tliq) = difference of density between External liquid and interdendritic liquid
        rhos.append(rho)
        drhos.append(d)
        drho_C0_T.append(rho_C0_Tliq)
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    y = {}
    y_drho = {'T':[],'C_Total':[]}
    n = len(comp.keys())
    colors=cm.jet(np.linspace(0,1,n))
    for k,eps in enumerate(eps_s):
        y_drho['T'].append(drhos[k]['T'])
        sum_eps_C = 0.0
        for element,_ in comp.items():
            sum_eps_C += parameters[element]['beta']*scheil_comp[k][element]
        y_drho['C_Total'].append(sum_eps_C)
    for j,(element,_) in enumerate(comp.items()):
        y[element] = []
        y_drho[element] = []
        for k,eps in enumerate(eps_s):
            y[element].append(scheil_comp[k][element])
            y_drho[element].append(drhos[k][element])
        ax1.plot(eps_s,y[element],label=element,color=colors[j],lw=1)
        ax2.plot(eps_s,y_drho[element],label=element,color=colors[j],lw=1)
    ax2.plot(eps_s,y_drho['T'],label='T',color='red',linestyle='--',lw=2)
    ax2.plot(eps_s,y_drho['C_Total'],label='C total',color='black',linestyle='--',lw=2)
    ax3.plot(eps_s,Tliq,label=r'$T_{liq}$')
    ax2.set_ylabel(r'$d\rho$')
    ax1.vlines(0.8,ax1.get_ylim()[0],ax1.get_ylim()[1],linestyle='--',color='black')
    ax2.vlines(0.8,ax2.get_ylim()[0],ax2.get_ylim()[1],linestyle='--',color='black')
    ax4 = ax3.twinx()
    ax4.plot(eps_s,drho,color='red',label=r'$d\rho(C_L,T_L)$')
    ax4.plot(eps_s,drho_C0_T,color='green',label=r'$d\rho(C_0,T_L)$')
    ax3.legend(loc='upper left')
    ax4.legend()
    ax1.set_xlim(0.0,1.0)
    ax1.set_xlabel(r'$\epsilon_s$' + ' (-)')
    ax2.set_xlabel(r'$\epsilon_s$' + ' (-)')
    ax1.set_ylabel(r'$C_{L,j}$' + ' (wt%)')
    ax1.legend()
    ax2.legend()
    plt.show()

    
