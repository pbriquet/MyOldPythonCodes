import sys, os  # Copy this heading to import from Libraries folders.
from math import *
from PeriodicTable import *
from TemperatureConverter import *

class Austenite:
    class Andrews:
        @staticmethod
        def Ae1(alloy,TScale=TempScale.Celsius):
            tmp = 723.0 - 16.9*alloy['Ni'] + 29.1*alloy['Si'] + 6.38*alloy['W'] - 10.7*alloy['Mn'] + 16.9*alloy['Cr'] + 290.0*alloy['As']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
        @staticmethod
        def Ae3(alloy,TScale=TempScale.Celsius):
            tmp = 910.0 - 203.0*sqrt(alloy['C']) + 44.7*alloy['Si'] - 15.2*alloy['Ni'] + 31.5*alloy['Mo'] + 104.0*alloy['V'] + 13.1*alloy['W'] - 30.0*alloy['Mn'] + 11.0*alloy['Cr'] + 20.0*alloy['Cu'] - 700.0*alloy['P'] - 400.0*alloy['Al'] - 120.0*alloy['As'] - 400.0*alloy['Ti']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
    class Eldis:
        @staticmethod
        def Ae1(alloy,TScale=TempScale.Celsius):
            tmp = 712.0 - 17.8*alloy['Mn'] - 19.1*alloy['Ni'] + 20.1*alloy['Si'] + 11.9*alloy['Cr'] + 9.8*alloy['Mo']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

        @staticmethod
        def Ae3(alloy,TScale=TempScale.Celsius):
            tmp = 871.0 - 254.4*sqrt(alloy['C']) + 51.7*alloy['Si'] - 14.2*alloy['Ni']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

    class Grange: # Fahrenheit (to change)
        @staticmethod
        def Ae1(alloy,TScale=TempScale.Fahrenheit):
            tmp = 1333.0 - 25.0*alloy['Mn'] + 40.0*alloy['Si'] + 42.0*alloy['Cr'] - 26.0*alloy['Ni']
            if(TScale != TempScale.Fahrenheit):
                return TConvert(tmp,TempScale.Fahrenheit,TScale)
            else:
                return tmp
    
        @staticmethod
        def Ae3(alloy,TScale=TempScale.Fahrenheit):
            tmp = 1570.0 - 323.0*alloy['C'] - 25.0*alloy['Mn'] + 80.0*alloy['Si'] - 3.0*alloy['Cr'] - 32.0*alloy['Ni']
            if(TScale != TempScale.Fahrenheit):
                return TConvert(tmp,TempScale.Fahrenheit,TScale)
            else:
                return tmp

    class Hougardy:
        @staticmethod
        def Ac1(alloy,TScale=TempScale.Celsius):
            tmp = 739.0 - 22.0*alloy['C'] - 7.0*alloy['Mn'] + 2.0*alloy['Si'] + 14.0*alloy['Cr'] + 13.0*alloy['Mo'] - 13.0*alloy['Ni']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
    
        @staticmethod
        def Ac3(alloy,TScale=TempScale.Celsius):
            tmp = 902.0 - 255.0*alloy['C'] - 11.0*alloy['Mn'] + 19.0*alloy['Si'] - 5.0*alloy['Cr'] + 13.0*alloy['Mo'] - 20.0*alloy['Ni'] + 55.0*alloy['V']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

    class Kariya: # Celsius
        @staticmethod
        def Ac1(alloy,TScale=TempScale.Celsius):
            tmp = 754.83 - 32.25*alloy['C'] - 17.76*alloy['Mn'] + 23.32*alloy['Si'] + 17.3*alloy['Cr'] + 4.51*alloy['Mo'] + 15.62*alloy['V']
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

    class Kasatkin: # Celsius
        @staticmethod
        def Validity(alloy):
            tmp = []
            tmp.append(alloy['C'] < 0.83)
            tmp.append(alloy['Mn'] < 2.0)
            tmp.append(alloy['Si'] < 1.0)
            tmp.append(alloy['Cr'] < 2.0)
            tmp.append(alloy['Mo'] < 1.0)
            tmp.append(alloy['Ni'] < 3.0)
            tmp.append(alloy['V'] < 0.5)
            tmp.append(alloy['W'] < 1.0)
            tmp.append(alloy['Ti'] < 0.15)
            tmp.append(alloy['Al'] < 0.2)
            tmp.append(alloy['Cu'] < 1.0)
            tmp.append(alloy['Nb'] < 0.2)
            tmp.append(alloy['P'] < 0.04)
            tmp.append(alloy['S'] < 0.04)
            tmp.append(alloy['N'] < 0.025)
            tmp.append(alloy['B'] < 0.01)
            return all(tmp)
        @staticmethod
        def Ac1(alloy,TScale=TempScale.Celsius):
            tmp = (723.0 - 7.08*alloy['Mn'] + 37.7*alloy['Si'] + 18.1*alloy['Cr'] + 44.2*alloy['Mo']
            + 8.95*alloy['Ni'] + 50.1*alloy['V'] + 21.7*alloy['Al'] + 3.18*alloy['W']
            + 297.0*alloy['S'] - 830.0*alloy['N'] - 11.5*alloy['C']*alloy['Si']
            - 14.0*alloy['Mn']*alloy['Si'] - 3.10*alloy['Si']*alloy['Cr'] - 57.9*alloy['C']*alloy['Mo']
            - 15.5*alloy['C']*alloy['Ni'] - 6.0*alloy['Mn']*alloy['Ni'] + 6.77*alloy['Si']*alloy['Ni'] - 0.8*alloy['Cr']*alloy['Ni']
            - 27.4*alloy['C']*alloy['V'] + 30.8*alloy['Mo']*alloy['V'] - 0.84*alloy['Cr']**2
            - 3.46*alloy['Mo']**2 - 0.46*alloy['Ni']**2 - 28.0*alloy['V']**2)
            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp
    
        @staticmethod
        def Ac3(alloy,TScale=TempScale.Celsius):
            tmp = (912.0 - 370.0*alloy['C'] - 27.4*alloy['Mn'] + 27.3*alloy['Si'] - 6.35*alloy['Cr']
            - 32.7*alloy['Ni'] + 95.2*alloy['V'] + 190.0*alloy['Ti'] + 72.0*alloy['Al'] + 64.5*alloy['Nb']
            + 5.57*alloy['W'] + 332.0*alloy['S'] + 276.0*alloy['P'] + 485*alloy['N'] - 900.0*alloy['B']
            + 16.2*alloy['C']*alloy['Mn'] + 32.3*alloy['C']*alloy['Si'] + 15.4*alloy['C']*alloy['Cr'] + 48.0*alloy['C']*alloy['Ni']
            + 4.32*alloy['Si']*alloy['Cr'] - 17.3*alloy['Si']*alloy['Mo'] - 18.6*alloy['Si']*alloy['Ni'] + 4.80*alloy['Mn']*alloy['Ni']
            + 40.5*alloy['Mo']*alloy['V'] + 174.0*alloy['C']**2 + 2.46*alloy['Mn']**2 - 6.86*alloy['Si']**2 + 0.322*alloy['Cr']**2 + 9.90*alloy['Mo']**2
            + 1.24*alloy['Ni']**2 - 60.2*alloy['V']**2)

            if(TScale != TempScale.Celsius):
                return TConvert(tmp,TempScale.Celsius,TScale)
            else:
                return tmp

        def dT(alloy):
            tmp = (188.0 - 370.0*alloy['C'] - 7.93*alloy['Mn'] - 26.8*alloy['Cr'] - 33.0*alloy['Mo'] - 23.5*alloy['Ni']
            + 52.5*alloy['V'] + 194.0*alloy['Ti'] + 47.8*alloy['Al'] + 87.4*alloy['Nb'] + 3.82*alloy['W']
            + 266.0*alloy['P'] + 53.0*alloy['C']*alloy['Si'] + 20.7*alloy['C']*alloy['Cr'] + 6.26*alloy['Si']*alloy['Cr']
            + 64.2*alloy['C']*alloy['Mo'] + 55.2*alloy['C']*alloy['Ni'] + 10.8*alloy['Mn']*alloy['Ni'] + 1.33*alloy['Cr']**2
            + 8.83*alloy['Mo']**2 + 1.91*alloy['Ni']**2 - 37.8*alloy['V']**2)
            
            return tmp