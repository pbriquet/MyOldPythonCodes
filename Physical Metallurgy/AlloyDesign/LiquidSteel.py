import numpy as np


class LiquidSteel:
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
            if(k in LiquidSteel.parameters.keys()):
                f_comp[k] = C*np.power(1.0 - fs,LiquidSteel.parameters[k]['K'] - 1.0)
        return f_comp

    @staticmethod
    def ScheilCs(comp,fs):
        f_comp = {}
        for k,C in comp.items():
            if(k in LiquidSteel.parameters.keys()):
                f_comp[k] = LiquidSteel.parameters[k]['K']*C*np.power(1.0 - fs,LiquidSteel.parameters[k]['K'] - 1.0)
        return f_comp

    def __init__(self,composition):
        self.comp = composition
        self.Cl = {}
        self.Cs = {}
        self.drho_L_Cj = {}
        self.drho_L_C = [None]
        self.Tliq = []
        self.TliqC0 = LiquidSteel.Tf + sum([LiquidSteel.parameters[k]['m']*v for k,v in self.comp.items()])
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
                self.Cs[element].append(LiquidSteel.parameters[element]['K']*self.Cl[element][-1])
                self.drho_L_Cj[element].append(LiquidSteel.parameters[element]['beta']*self.Cl[element][-1])
                self.drho_L_C[k] += self.drho_L_Cj[element][-1]
        got_Tcrit = False
        for k,eps in enumerate(fs_vector):
            _Tliq = Freckles.Tf
            for element,C in self.comp.items():
                _Tliq += Freckles.parameters[element]['m']*self.Cl[element][k]
            self.Tliq.append(_Tliq)
            if(not got_Tcrit):
                if(eps > Freckles.fs_crit):
                    self.Tcrit = self.Tliq[k-1] + (LiquidSteel.fs_crit - self.fs[k-1])/(self.fs[k] - self.fs[k-1])*(self.Tliq[k] - self.Tliq[k-1])
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
        