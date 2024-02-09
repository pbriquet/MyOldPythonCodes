import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt

def func_1(x, a,b,c):
    return a*x*x + b*x + c

def func_2(x,a,b):
    return a*np.log(x + 4 + b)

def func_3(x,a,b):
    return a*x + b
if __name__ == "__main__":
    T1 = [3.07491610e+01,3.65553019e+04,1.67339327e-01]    
    #T2 = [2.64007727e+01,4.02319023e+04,2.45523423e-01]
    T2 = [2.20307159e+01,3.64893483e+04,2.10508550e-01]
    T3 = [1.80493640e+00,2.28159347e+04,2.49862080e-01]

    TR1 = [778243353.679019, 218450.15500607732, 0.16733932704169427]
    TR2 = [616803.31594301, 163861.76867267754, 0.2455234230302942]
    TR3 = [10.627077878681785, 91314.11496012374, 0.24986208028235787]
    cooling_rate = [20.0,0.15,0.035]
    ln_cooling_rate = [np.log(c) for c in cooling_rate]
    index = 0
    #print(ln_cooling_rate)
    #print(T3[index],T2[index],T1[index])
    #exit()
    Ts = [T1,T2,T3]
    A = []
    B = []
    C = []
    for k,T in enumerate(Ts):
        A.append(T[0])
        B.append(T[1])
        C.append(T[2])

    '''
    p0 = [-1, -3e-3, 1]                                        # guessed params
    w, _ = opt.curve_fit(func, ln_cooling_rate, C, p0=p0)
    x_lin = np.linspace(ln_cooling_rate[0],ln_cooling_rate[-1],num=50)
    y_lin = func(x_lin,w[0],w[1],w[2])
    plt.plot(ln_cooling_rate,A)
    plt.plot(x_lin,y_lin)
    plt.show()
    '''
    choice = C
    func = func_3
    x = ln_cooling_rate
    p0 = [3000, 3000]                                        # guessed params
    w, _ = opt.curve_fit(func, x, choice, p0=p0)
    x_lin = np.linspace(x[0],x[-1],num=50)
    y_lin = func(x_lin,w[0],w[1])
    plt.plot(x,choice)
    plt.plot(x_lin,y_lin)
    plt.xlabel('ln(dT/dt)')
    plt.ylabel('m')
    print(w)
    plt.show()
