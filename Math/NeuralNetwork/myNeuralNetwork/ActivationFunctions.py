from math import *
import numpy as np
class ActivationFunctions:
    @staticmethod    
    def Identity(_input,_output,deriv=False):    # Identity
        if(deriv):
            return 1.0
        return _input

    @staticmethod
    def BinaryStep(x,deriv=False):    # Binary step
        if(deriv):
            return 0.0
        return 0.0 if x < 0.0 else 1.0

    @staticmethod
    def Logistic(_input,_output,deriv=False):    # Logistic (a.k.a. Sigmoid or Soft step)
        if(deriv):
            return _output*(1.0 - _output)
        return 1.0/(1.0 + np.exp(- _input))

    @staticmethod
    def Tanh(_input,_output,deriv=False):    # TanH
        if(deriv):
            return 1.0 - _output**2
        return (exp(_input)-exp(-_input))/(exp(_input) + exp(-_input))

    @staticmethod
    def Arctan(x,deriv=False):    # ArcTan
        if(deriv):
            return 1.0/(x**2 + 1.0)
        return atan(x)

    @staticmethod
    def Softsign(_input,_output,deriv=False):     # Softsign
        if(deriv):
            return 1.0/(1.0 + abs(_input))**2
        return _input/(1.0 + abs(_input))

    @staticmethod
    def ReLU(x,deriv=False):    # Rectified linear unit (ReLU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def ISRU(x,deriv=False):    # Inverse square root unit (ISRU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def LeakyReLU(x,deriv=False):    # Leaky rectified linear unit (Leaky ReLU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def PReLU(x,deriv=False):    # Parameteric rectified linear unit (PReLU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def RReLU(x,deriv=False):    # Randomized leaky rectified linear unit (RReLU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def ELU(x,deriv=False):    # Exponential linear unit (ELU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def SELU(x,deriv=False):    # Scaled exponential linear unit (SELU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def SReLU(x,deriv=False):    # S-shaped rectified linear activation unit (SReLU)
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def ISRLU(x,deriv=False):    # Inverse square root linear unit 
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def APL(x,deriv=False):    # Adaptive piecewise linear 
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def SoftPlus(x,deriv=False):    # SoftPlus
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def BentIdentity(x,deriv=False):    # Bent identity
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def SiLU(x,deriv=False):    # Sigmoid-weighted linear unit (SiLU)
        if(deriv):
            return 0.0
        return 0.0
    
    @staticmethod
    def SoftExponential(x,deriv=False):    # SoftExponential
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def Sinusoid(x,deriv=False):    # Sinusoid
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def Sinc(x,deriv=False):    # Sinc
        if(deriv):
            return 0.0
        return 0.0

    @staticmethod
    def Gaussian(x,deriv=False):    # Gaussian
        if(deriv):
            return 0.0
        return 0.0