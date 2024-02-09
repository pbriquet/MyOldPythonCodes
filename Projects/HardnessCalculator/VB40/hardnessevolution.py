from hardnessmodel import *
from math import *

class HardnessEvolution:
    def __init__(self,model,tmax,dtmax):
        self.model = model
        self.tmax = tmax
        self.dtmax = dtmax



    def setInitialConditions(self,T0,Treach,dTdt):
        self.T0 = T0
        self.Treach = Treach
        self.H = self.model.H0
        self.dTdt = dTdt
        self.t_ramp = (Treach - T0)/dTdt
        self.t0 = (self.model.Tmin - self.T0)/dTdt
        print "t0 = " + str(self.t0)

    def calculate(self):
        t_array = np.logspace(-5, log(self.tmax,2.0), num=1e5, base=2.0)
        tau_array = [1e-5]
        alpha_array = [1e-5]
        T_array = [self.T0]
        for i in xrange(1,len(t_array)):
            alpha = alpha_array[i-1]
            tau = tau_array[i-1]
            T = T_array[i-1]
            t = t_array[i-1]
            dt = t_array[i] - t_array[i-1]
            
            dT = self.T_in_time(t_array[i]) - T
            _dTdt = dT/dt
            tau += self.model.dtaudt(t,T,alpha,_dTdt,t0=self.t0)*dt
            tau_array.append(tau)
            alpha += self.model.dalphadt(t,T,alpha,_dTdt,t0=self.t0)*dt
            alpha_array.append(alpha)
            T_array.append(T + dT)

        return t_array,tau_array,alpha_array,T_array
            
    def T_in_time(self,t):
        if(t<self.t_ramp):
            return self.T0 + self.dTdt*t
        else:
            return self.Treach
        