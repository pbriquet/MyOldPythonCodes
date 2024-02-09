import sys, os
from math import *
from PeriodicTable import *

class Martensite:
    class Andrews:
        @staticmethod
        def Validity(alloy):
            tmp = []
            tmp.append(alloy['C'] < 0.6)
            tmp.append(alloy['Mn'] < 4.9)
            tmp.append(alloy['Cr'] < 5.0)
            tmp.append(alloy['Ni'] < 5.0)
            tmp.append(alloy['Mo'] < 5.4)
            return all(tmp)
        
        @staticmethod
        def Ms(alloy,TScale=TempScale.Celsius):
            tmp = (539.0 - 423.0*alloy['C'] - 30.4*alloy['Mn'] - 17.7*alloy['Ni'] - 12.1*alloy['Cr'] - 11.0*alloy['Si'] - 7.0*alloy['Mo'])
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

        @staticmethod
        def Mf(alloy,TScale=TempScale.Celsius):
            tmp = (512.0 - 453.0*alloy['C'] - 16.9*alloy['Ni'] - 9.5*alloy['Mo'] + 217.0*alloy['C']**2 - 71.5*alloy['C']*alloy['Mn'] + 15.0*alloy['Cr'] - 67.6*alloy['C']*alloy['Cr'])
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
    class Capdevila:
        @staticmethod
        def Ms(alloy,TScale=TempScale.Celsius):
            tmp = -273.15 + 764.2 - 302.6*alloy['C']- 30.6*alloy['Mn'] - 16.6*alloy['Ni'] - 8.9*alloy['Cr'] + 2.4*alloy['Mo'] - 11.3*alloy['Cu'] + 8.58*alloy['Co'] + 7.4*alloy['W'] - 14.5*alloy['Si']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
    class Carapella:
        @staticmethod
        def Ms(alloy,TScale=TempScale.Celsius):
            tmp = 492.0*(1.0 - 0.62*alloy['C'])*(1.0 - 0.092*alloy['Mn'])*(1.0 - 0.033*alloy['Si'])*(1.0 - 0.045*alloy['Ni'])*(1.0 - 0.07*alloy['Cr'])*(1.0 - 0.029*alloy['Mo'])*(1.0 - 0.018*alloy['W'])*(1.0 - 0.012*alloy['Co'])
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
    class Eldis:
        @staticmethod
        def Ms(alloy,TScale=TempScale.Celsius):
            tmp = 531.0 - 391.2*alloy['C'] - 43.3*alloy['Mn'] - 21.8*alloy['Ni'] - 16.2*alloy['Cr']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

    class GrangeStewart:
        @staticmethod
        def Ms(alloy,TScale=TempScale.Celsius):
            tmp = 538.0 - 350.0*alloy['C'] - 37.7*alloy['Mn'] - 37.7*alloy['Cr'] - 18.9*alloy['Ni'] - 27.0*alloy['Mo']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
    class Zhao:
        @staticmethod
        def Ms_TM(alloy,TScale=TempScale.Celsius):
            tmp = (420.0 - 208.33*alloy['C']
            - 33.428*alloy['Mn'] + 1.296*alloy['Mn']**2 - 0.02167*alloy['Mn']**3
            - 16.08*alloy['Ni'] + 0.7817*alloy['Ni']**2 - 0.02464*alloy['Ni']**3
            - 2.473*alloy['Cr']
            + 30.0*alloy['Mo']
            +12.86*alloy['Co'] - 0.2654*alloy['Co']**2 + 0.001547*alloy['Co']**3
            - 7.18*alloy['Cu']
            - 72.65*alloy['N'] - 43.36*alloy['N']
            - 16.28*alloy['Ru'] + 1.72*alloy['Ru']**2 - 0.08117*alloy['Ru']**3)

            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
        @staticmethod
        def Ms_LM(alloy,TScale=TempScale.Celsius):
            tmp = (540.0 - 356.25*alloy['C'] 
            - 47.59*alloy['Mn'] + 2.25*alloy['Mn']**2 - 0.0415*alloy['Mn']**3 
            - 24.56*alloy['Ni'] + 1.36*alloy['Ni']**2 - 0.0384*alloy['Ni']**3 
            - 17.82*alloy['Cr'] + 1.42*alloy['Cr']**2 
            + 17.50*alloy['Mo']
            + 21.87*alloy['Co'] - 0.468*alloy['Co']**2 + 0.00296*alloy['Co']**3
            - 16.52*alloy['Cu'] 
            - 260.64*alloy['N']
            - 17.66*alloy['Ru'])
            
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

    class JaffeHollomon:
        @staticmethod
        def Ms(alloy,TScale=TempScale.Celsius):
            tmp = (550.0 - 350.0*alloy['C'] - 40.0*alloy['Mn'] - 35.0*alloy['V']
            - 20.0*alloy['Cr'] - 17.0*alloy['Ni'] -10.0*alloy['Cu'] - 10.0*alloy['Mo']
            - 8.0*alloy['W'] + 15.0*alloy['Co'] + 30.0*alloy['Al'])
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

        @staticmethod
        def Mf(alloy,TScale=TempScale.Celsius):
            tmp = (512.0 - 453.0*alloy['C'] - 16.9*alloy['Ni'] - 9.5*alloy['Mo'] + 217.0*alloy['C']**2 - 71.5*alloy['C']*alloy['Mn'] + 15.0*alloy['Cr'] - 67.6*alloy['C']*alloy['Cr'])
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp