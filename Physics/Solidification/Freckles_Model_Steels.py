import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

class MassBalance:
    def __init__(self):
        self.profile = {
            'T':[],
            'fs':[],
            'Cl':[]
        }
class Freckles:
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
    Tf = 1538.0
    fs_crit = 0.8
    rho0 = 7000.0

    @staticmethod
    def ScheilCL(comp,fs):
        f_comp = {}
        for k,C in comp.items():
            f_comp[k] = C*np.power(1.0 - fs,Freckles.parameters[k]['K'] - 1.0)
        return f_comp
    def __init__(self,composition):
        self.comp = composition
        self.Cl = {}
        self.Cs = {}
        self.drho_L_Cj = {}
        self.drho_L_C = [None]
        self.Tliq = []
        self.TliqC0 = Freckles.Tf + sum([Freckles.parameters[k]['m']*v for k,v in self.comp.items()])
        self.fs = []
        self.Tcrit = 0.0
    def calculate_segregation_profile(self,fs_vector):
        self.fs = fs_vector
        self.drho_L_C = [0.0 for eps in fs_vector]
        
        for element,C in self.comp.items():
            self.Cl[element] = []
            self.Cs[element] = []
            self.drho_L_Cj[element] = []
            
            for k,eps in enumerate(fs_vector):
                self.Cl[element].append(C*np.power(1.0 - eps,Freckles.parameters[element]['K'] - 1.0))
                self.Cs[element].append(Freckles.parameters[element]['K']*self.Cl[element][-1])
                self.drho_L_Cj[element].append(Freckles.parameters[element]['beta']*self.Cl[element][-1])
                self.drho_L_C[k] += self.drho_L_Cj[element][-1]
        got_Tcrit = False
        for k,eps in enumerate(fs_vector):
            _Tliq = Freckles.Tf
            for element,C in self.comp.items():
                _Tliq += Freckles.parameters[element]['m']*self.Cl[element][k]
            self.Tliq.append(_Tliq)
            if(not got_Tcrit):
                if(eps > Freckles.fs_crit):
                    self.Tcrit = self.Tliq[k-1] + (Freckles.fs_crit - self.fs[k-1])/(self.fs[k] - self.fs[k-1])*(self.Tliq[k] - self.Tliq[k-1])
                    got_Tcrit = True

            
    def get_fs(self,T):
        if(T > self.Tliq[0]):
            return 0.0
        elif(T < self.Tliq[-1]):
            return 1.0
        else:
            for k,_T in enumerate(np.array(self.Tliq)[1:]):
                if(T <= self.Tliq[(k+1)-1] and T > self.Tliq[(k+1)]):
                    return self.fs[(k+1)-1] + (self.fs[(k+1)] - self.fs[(k+1)-1])/(self.Tliq[(k+1)] - self.Tliq[(k+1)-1])*(T - self.Tliq[(k+1)-1])

    def get_Cl(self,element,T):
        if(T > self.Tliq[0]):
            return self.comp[element]
        elif(T < self.Tliq[-1]):
            return self.Cl[element][-1]
        else:
            for k,_T in enumerate(np.array(self.Tliq)[1:]):
                if(T <= self.Tliq[(k+1)-1] and T > self.Tliq[(k+1)]):
                    return self.Cl[element][(k+1)-1] + (self.Cl[element][(k+1)] - self.Cl[element][(k+1)-1])/(self.Tliq[(k+1)] - self.Tliq[(k+1)-1])*(T - self.Tliq[(k+1)-1])
        
def test():
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
    
    freckles = Freckles(comp)
    fs = np.linspace(0.0,0.95,num=20)
    freckles.calculate_segregation_profile(fs)
    
    
    T = np.linspace(freckles.Tcrit,freckles.TliqC0,num=200)
    
    
    new_fs = []
    
    for k,_T in enumerate(T):
        new_fs.append(freckles.get_fs(_T))
    color=cm.rainbow(np.linspace(0,1,len(comp.keys())))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freckles.Tliq,freckles.drho_L_C)
    '''
    for j,(k,v) in enumerate(freckles.Cl.items()):
        ax.plot(fs,v,label=k,c=color[j])
        ax.hlines(comp[k],0.0,1.0,linestyle='--',color=color[j])
    '''
    plt.legend()
    plt.show()

def plot_K_vs_beta():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    color=cm.jet(np.linspace(0,1,len(Freckles.parameters.keys())))
    x = []
    y = []
    c = []
    for j,(k,C) in enumerate(Freckles.parameters.items()):
        #ax.scatter(C['K'],C['beta'],label=k,cmap=cm.jet,c=C['m'],s=20)
        ax.annotate(k, (C['K'], C['beta']))
        x.append(C['K'])
        y.append(C['beta'])
        c.append(C['m'])
    sct = ax.scatter(x,y,cmap=cm.jet,c=c,s=20)
    ax.hlines(0.0,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--',color='black')
    clb = plt.colorbar(sct)
    clb.set_label('m = ' + r'$\frac{\partial T_{liq}}{\partial C_{L,j}}$')
    #ax.legend()
    ax.set_xlabel('Coeficiente de Partição (' + r'$K_j=\frac{C_{S,j}^*}{C_{L,j}^*}$' + ')')
    ax.set_ylabel('Parametro de Contração (' + r'$\beta_j=\frac{d\rho_L}{dC_{L,j}}$' + ')')
    plt.show()
def plot_K_lambda_fs():
    fs = np.linspace(1e-3,0.95,num=100)
    lambda2 = np.linspace(1e-4,1e-3,num=100)
    xx,yy = np.meshgrid(fs,lambda2,indexing='ij')
    K = lambda _fs,_l: np.power(_l,2)/180.0*np.power(1.0 - _fs,3)/np.power(_fs,2)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    zz = np.log10(K(xx,yy))
    norm = matplotlib.colors.Normalize(vmin = np.min(zz), vmax = np.max(zz), clip = False)
    ax.plot_surface(xx,yy,zz,cmap=cm.jet,norm=norm)
    ax.set_xlabel(r'$\epsilon_s$' + ' (-)')
    ax.set_ylabel(r'$\lambda_2$' + '(m)')
    ax.set_zlabel('log(Permeability)')
    plt.show()
def StationaryProfile():
    G = 1e3
    V = 1.0
    R = -V*G

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
    
    freckles = Freckles(comp)
    fs = np.linspace(0.0,0.95,num=100)
    freckles.calculate_segregation_profile(fs)
    T = np.linspace(freckles.Tcrit,freckles.TliqC0,num=200)
    new_fs = []
    new_Cl = {element:[] for element in comp.keys()}
    for k,_T in enumerate(T):
        new_fs.append(freckles.get_fs(_T))
        for element,C in comp.items():
            new_Cl[element].append(freckles.get_Cl(element,_T))
    
    y0 = freckles.Tcrit/G
    yL = freckles.TliqC0/G
    y = T/G - y0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(new_fs,new_Cl['Si'])
    plt.show()

    
    


if __name__ == "__main__":
    StationaryProfile()