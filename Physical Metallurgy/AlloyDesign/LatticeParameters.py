from TemperatureConverter import *
import numpy as np 

class LatticeParameters:
    @staticmethod
    def Alpha_Ferrite(T,TScale=TempScale.Kelvin):
        if(TempScale!=TempScale.Kelvin):
            T = TConvert(T,TempScale.Kelvin,TScale)
        return 2.8863*(1.0 + 17.5e-6*(T - 800.0))

    @staticmethod
    def Cementite(T,TScale=TempScale.Kelvin):
        if(TempScale!=TempScale.Kelvin):
            T = TConvert(T,TempScale.Kelvin,TScale)
        a = 4.5234*(1.0 + (5.311e-6 - 1.942e-9*T + 9.655e-12*np.power(T,2))*(T - 293.0))
        b = 5.0883*(1.0 + (5.311e-6 - 1.942e-9*T + 9.655e-12*np.power(T,2))*(T - 293.0))
        c = 6.7426*(1.0 + (5.311e-6 - 1.942e-9*T + 9.655e-12*np.power(T,2))*(T - 293.0))
        return a,b,c