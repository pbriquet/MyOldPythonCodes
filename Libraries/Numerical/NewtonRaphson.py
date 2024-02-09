from math import *

def NewtonRaphson(f,dfdx,x0,epsilon):
    error = 1.0
    x = x0
    while(error > epsilon):
        xant = x
        if(dfdx(x)!= 0.0):
            x = x - f(x)/dfdx(x)
        else:
            return 'Error of method'
        
        if(xant != 0.0):
            error = abs((x-xant)/xant)
        else:
            error = abs(x-xant)
    return x
def SecantMethod(f,x0,epsilon,eps=1e-6):
    error = 1.0
    x = x0
    while(error > epsilon):
        xant = x
        if(dfdx(x)!= 0.0):
            x = x - f(x)/(f(x)-f(x+eps))*eps
        else:
            return 'Error of method'
        
        if(xant != 0.0):
            error = abs((x-xant)/xant)
        else:
            error = abs(x-xant)
    return x

def Bissection(f,a,b,epsilon=1e-6):
    error = 1.0
    xm = (a+b)/2.0

    while(error > epsilon):
        xm = (a + b)/2.0
        if(f(a)==0.0):
            return a
        elif(f(b)==0.0):
            return b
        elif(f(xm)==0.0):
            return xm
        elif(f(a)*f(xm) < 0.0):
            b = xm
        elif(f(xm)*f(b) < 0.0):
            a = xm
            
        error = abs(b-a)/2.0

    return xm