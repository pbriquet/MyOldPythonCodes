from math import *
import numpy as np
import matplotlib.pyplot as plt
from HB_Data import *
import os



class HardnessModel:
    def __init__(self):

        self.m = 0.65
        self.D0 = 0.28
        self.QD = 1000.0
        self.Qinf = 6.7
        self.Qinf2 = 4.1e3
        self.Tmin = 430.0
        self.Tmax = 751.0
        self.R = 8.31
        self.H0 = 500.0
        self.H_inf_Tmax = 200.0

        self.dH = self.H0-self.H_inf_Tmax

    def TK(self,TC):
        return TC + 273.15

    def functional(self,m,D0,QD,Qinf):
        self.m = m
        self.D0 = D0
        self.QD = QD
        self.Qinf2 = Qinf
        return self

    def change_parameters(self,H0,H_inf_Tmax,Tmin,Tmax):
        self.H0 = H0
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.H_inf_Tmax = H_inf_Tmax
        self.dH = self.H0-self.H_inf_Tmax

    def phi(self,t,T):
        return np.exp(-(self.D(T)*t)**self.m)

    def beta(self,T):
        return np.exp(-self.Qinf2/self.R*(self.Tmin - T)/(T - self.Tmax)/T)

    def Hfar(self,T):
        return self.H0 - (1.0 - self.theta(T))*(self.H0 - self.H_inf_Tmax)

    def theta(self,T):
        if((T <= self.Tmin).all()):
            return 1.0
        elif((T >= self.Tmax).all()):
            return 0.0
        else:
            return np.exp(-self.Qinf2/self.R*(self.Tmin - T)/(T - self.Tmax)/T)

    def dthetadT(self,T):
        if(T <= self.Tmin + 5.0):
            return 0.0
        elif(T >= self.Tmax - 5.0):
            return 0.0
        else:
            return -self.theta(T)*log(self.theta(T))*(1.0/T - 1.0/(self.Tmax - T) - 1.0/(T - self.Tmin))

    def D(self,T):
        return self.D0*np.exp(-self.QD/self.R/self.TK(T))

    def H(self,t,T):
        return self.H0 - (self.H0 - self.H_inf_Tmax)*(1.0 - self.phi(t,T))*(1.0 - self.beta(T))

    def H_from_tau(self,tau):
        return self.H0 - (self.H0 - self.H_inf_Tmax)*tau

    def tau(self,t,T):
        return self.alpha(t,T)*(1.0 - self.theta(T))

    def alpha(self,t,T):
        return 1.0 - self.phi(t,T)

    def dalphadt(self,t,T,_alpha,dTdt,t0=0.0):
        if(t < t0):
            return 0.0
        else:
            return (1.0 - _alpha)*self.m*((t - t0)*self.D(T))**(self.m-1.0)*self.D(T)*(1.0 + (t - t0)*self.QD/self.R/self.TK(T)**2*dTdt)

    def dtaudt(self,t,T,alpha,dTdt,t0=0.0):
        if(T <= self.Tmin):
            return 0.0
        elif(T >= self.Tmax):
            return 0.0
        else:
            return self.dalphadt(t,T,alpha,dTdt,t0=t0)*(1.0 - self.theta(T)) - alpha*self.dthetadT(T)*dTdt

    def ConvertToTau(self,Hardness):
        return (self.H0 - Hardness)/(self.H0 - self.H_inf_Tmax)

def test():
    model = HardnessModel()
    Temp = np.linspace(model.Tmin, model.Tmax - 10.0, num=50)

    p_beta = map(lambda T: model.H_inf_Tmax + (model.H0 - model.H_inf_Tmax)*model.beta(T),Temp)
    p_theta = map(lambda T: model.H_inf_Tmax + (model.H0 - model.H_inf_Tmax)*model.theta(T),Temp)

    plt.plot(Temp,p_beta, 'r')
    plt.plot(Temp,p_theta, 'g')
    plt.show()
