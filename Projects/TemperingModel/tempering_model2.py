import numpy as np
import matplotlib.pyplot as plt 
import os
import pandas as pd 

R = 8.31

T1 = [3.07491610e+01,3.65553019e+04,1.67339327e-01,534.0]    
T2 = [2.20307159e+01,3.64893483e+04,2.10508550e-01,528.0]
T3 = [1.80493640e+00,2.28159347e+04,2.49862080e-01,497.0]

#T1 = [3.07491610e+01,3.65553019e+04,1.67339327e-01,534.0]    
#T2 = [2.20307159e+01,3.64893483e+04,2.10508550e-01,500.0]
#T3 = [1.80493640e+00,2.28159347e+04,2.49862080e-01,490.0]

cooling_rate = [20.0,0.15,0.035]
ln_cooling_rate = [np.log(c) for c in cooling_rate]

def A(dTdt):
    if(dTdt >= 0.15):
        return T2[0] + (T1[0] - T2[0])/(ln_cooling_rate[0] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])
    else:
        return T2[0] + (T3[0] - T2[0])/(ln_cooling_rate[2] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])

def B(dTdt):
    if(dTdt >= 0.15):
        return T2[1] + (T1[1] - T2[1])/(ln_cooling_rate[0] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])
    else:
        return T2[1] + (T3[1] - T2[1])/(ln_cooling_rate[2] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])

def C(dTdt):
    if(dTdt >= 0.15):
        return T2[2] + (T1[2] - T2[2])/(ln_cooling_rate[0] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])
    else:
        return T2[2] + (T3[2] - T2[2])/(ln_cooling_rate[2] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])

def H0(dTdt):
    if(dTdt >= 0.15):
        return T2[3] + (T1[3] - T2[3])/(ln_cooling_rate[0] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])
    else:
        return T2[3] + (T3[3] - T2[3])/(ln_cooling_rate[2] - ln_cooling_rate[1])*(np.log(dTdt) - ln_cooling_rate[1])

def see_functions():
    x = np.linspace(0.035,20.0,num=5000)
    x_ln = [np.log(i) for i in x]
    y = [H0(i) for i in x]
    plt.plot(x_ln,y)
    plt.show()

class Model:
    def __init__(self,dTdt):
        self.A = A(dTdt)
        self.B = B(dTdt)
        self.C = C(dTdt)
        self.H0 = H0(dTdt)
        self.Hfar = 245.0
        self.a = np.power(self.A,1.0/self.C)
        self.b = self.B/self.C
        self.c = self.C

    def calculate(self,df,col):
        self.H = [self.H0]
        self.alpha = [0.0]
        self.ksi = [0.0]
        t = list(df['t'])
        T = list(df[col])
        for j,v in enumerate(t):
            if(j!=0):
                T_ave = (T[j] + T[j-1])/2.0
                self.ksi.append(self.ksi[j-1] + self.k(T_ave)*(t[j]/3600.0 - t[j-1]/3600.0))
                self.alpha.append(self.ksi_function(self.ksi[-1]))
                self.H.append(self.H_function(self.alpha[-1]))
        #print(self.H)
    def k(self,T):
        return self.a*np.exp(-self.b/R/(T + 273.15))
    def ksi_function(self,ksi):
        return 1.0 - np.exp(-np.power(ksi,self.c))
    def H_function(self,alpha):
        return self.H0 - (self.H0 - self.Hfar)*alpha
    
if __name__ == "__main__":
    __loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    df = pd.read_excel(os.path.join(__loc__,'tempering.xlsx'))
    #print(df.head())
    TR1 = 20.0
    TR2 = 0.15
    TR3 = 0.035

    m1 = Model(TR1)
    m2 = Model(TR2)
    m2.H0 = H0(TR2)
    m3 = Model(TR3)
    m3.H0 = H0(TR3)
    #print(m1.a, m1.b, m1.c)
    #print(m2.a, m2.b, m2.c)
    #print(m3.a, m3.b, m3.c)

    m1.calculate(df,'P1')
    m2.calculate(df,'P2')
    m3.calculate(df,'P3')
    plt.plot(list(df['t']/3600.0),m1.H,label='dT/dt = 20°C/s')
    plt.plot(list(df['t']/3600.0),m2.H,label='dT/dt = 0.15°C/s')
    plt.plot(list(df['t']/3600.0),m3.H,label='dT/dt = 0.035°C/s')
    plt.legend()
    plt.xlabel('t (h)')
    plt.ylabel('H (Vickers)')
    plt.show()