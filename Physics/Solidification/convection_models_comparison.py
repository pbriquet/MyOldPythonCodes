import numpy as np
import matplotlib.pyplot as plt

vl = 0.9e-3
G = 1e-3
Q = 3000.0
rho = 7.000
Cp = 250.0
Lf = 5000.0

C0 = 5.0
ml = -2.3
k = 0.15
Tliq = 500.0
Teut = 300.0
Tf = Tliq - ml*C0

Ceut = 8.0


Ds = 1e-11
Dl = 1e-9

dt = 1e-3
dx = 1e-3


def del_s():
    return 1e-4

def fs_linear(T):
    return (T - Teut)/(Tliq - Teut)

def fs_alavanca(T):
    return (T - Tliq)/(T- Tf)/(1.0 - k)

def fs_scheil(T):
    return 1.0 - ((T - Tf)/(Tliq - Tf))**(1.0/(k - 1.0))

def fs_balance(T):
    _Tnext = 0.0
    _Clnext = 0.0
    _fs_next = 1.0

    return _Tnext, _Clnext, _fs_next

if __name__=="__main__":
