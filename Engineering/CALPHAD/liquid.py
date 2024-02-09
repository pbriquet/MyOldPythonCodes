import mendeleev as me
import numpy as np
import matplotlib.pyplot as plt

class LIQUID:
    elements = ['VA','H','Mg','Ca','Y','Ti','Zr','Nb','Ta','Cr','Mo','W','Mn','Fe','Co','Ni','Cu','Zn','B','C','N','O','Al','Si','P','S','Ar','Ce']
    balances = ['Fe']
    def __init__(self):
        pass
    @staticmethod
    def G_C_0(P,T):
        return LIQUID.GCLIQ(P,T)+LIQUID.GPCLIQ(P,T)
    
    @staticmethod
    def GCLIQ(P,T):
        return 117369.0 - 24.63*T + LIQUID.GHSERCC(P,T)

    @staticmethod
    def GHSERCC(P,T):
        return -17368.441+170.73*T - 24.3*T*np.log(T) -4.723E-04*np.power(T,2)+2562600*np.power(T,-1)-2.643e8*np.power(T,-2)+1.2e10*np.power(T,-3)

    @staticmethod
    def GPCLIQ(P,T):
        return LIQUID.YCLIQ(P,T)*np.exp(LIQUID.ZCLIQ(P,T))
    @staticmethod
    def YCLIQ(P,T):
        return LIQUID.VCLIQ(P,T)*np.exp(-LIQUID.ECLIQ(P,T))
    @staticmethod
    def ZCLIQ(P,T):
        return np.log(LIQUID.XCLIQ(P,T))

    @staticmethod
    def VCLIQ(P,T):
        return 7.626e-6*np.exp(LIQUID.ACLIQ(P,T))

    @staticmethod
    def XCLIQ(P,T):
        return np.exp(0.5*LIQUID.DCLIQ(P,T))-1.0

    @staticmethod
    def DCLIQ(P,T):
        return np.log(LIQUID.BCLIQ(P,T))

    @staticmethod
    def BCLIQ(P,T):
        return 1.0+3.2e-10*P

    @staticmethod
    def ECLIQ(P,T):
        return np.log(LIQUID.CCLIQ(P,T))

    @staticmethod
    def CCLIQ(P,T):
        return 1.6e-10
    @staticmethod
    def ACLIQ(P,T):
        return 2.32e-5*T+2.85e-9*np.power(T,2)
if __name__=='__main__':
    T = np.linspace(273.15,1073.15,num=1000)
    plt.plot(T,LIQUID.G_C_0(1e5,T))
    plt.show()