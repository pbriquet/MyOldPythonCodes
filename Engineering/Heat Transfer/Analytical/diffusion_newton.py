import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


Bi = 20.0


class VF:
    def __init__(self,i,rP,re,rw,C0):
        self.rP = rP
        self.re = re
        self.rw = rw
        self.r = [self.rw,self.rP,self.re]
        self.ri = [self.rw,self.re]
        self.i = i
        self.C = C0
        self.Cdt = C0

        
    
    def getNeighbors(self,vfs):
        self.nr = len(vfs)
        self.ve = None
        self.vw = None
        if(self.i!=0):
            self.vw = vfs[self.i-1]
        if(self.i!=len(vfs)-1):
            self.ve = vfs[self.i+1]
        
    def defineInterfaces(self,Cfar):
        if(self.i==0):
            self.fw = lambda: self.C
        else:
            self.fw = lambda: (self.C + self.vw.C)/2.0
        if(self.i==self.nr-1):
            self.fe = lambda: self.C*(1.0/(Bi*(self.re - self.rP) + 1.0))
        else:
            self.fe = lambda: (self.C+self.ve.C)/2.0

        self.f_interface = [self.fw, lambda: self.C, self.fe]

    def Calculate(self,dt):
        balance = 0.0
        for i in xrange(2):
            balance -= (-1)**i*(self.f_interface[i+1]()-self.f_interface[i]())/(self.r[i+1]- self.r[i])*self.ri[i]
        balance *= 2.0*dt/(self.re**2 - self.rw**2)
        self.Cdt = self.C + balance

    def Refresh(self):
        self.C = self.Cdt

class Model:
    def __init__(self,nr,dt, tmax, num_print=10):
        self.C0 = 1.0
        self.Cfar = 0.0
        self.vfs = []
        self.dt = dt
        self.tmax = tmax
        self.t = 0.0
        self.t_intervals = np.linspace(0.0,tmax,num=num_print,endpoint=True)
        self.C_values = []
        self.t_realintervals = []

        self.dr = 1.0/nr
        for i in xrange(nr):
            self.vfs.append(VF(i,(i+0.5)*self.dr,(i+1)*self.dr,i*self.dr,self.C0))

        for i in xrange(nr):
            self.vfs[i].getNeighbors(self.vfs)
            self.vfs[i].defineInterfaces(self.Cfar)

    def RunTime(self):
        while(self.t <= self.tmax + 10.0*self.dt):
            for i in self.t_intervals:
                if( (self.t - self.dt/2.0 < i).all() and (self.t + self.dt/2.0 > i).all()):
                    self.t_realintervals.append(self.t)
                    self.C_values.append([])
                    for k in self.vfs:
                        self.C_values[len(self.C_values)-1].append(k.C)

            for k in self.vfs:
                k.Calculate(self.dt)

            for k in self.vfs:
                k.Refresh()

            self.t += self.dt


nr = 100
num_plots =20
dtmax = 1e-5
tmax = 2.0

r = np.linspace(0.0,1.0,num=nr)
m = Model(nr,dtmax,tmax,num_print=num_plots)
R = 250.0e-3
D = 1.2e-6
tau = R**2/D
print "tau = " + str(tau/3600.0) + ' h'
m.RunTime()
cm = plt.get_cmap('plasma')
plt.figure()
plt.hlines(np.linspace(0.0,1.0,num=10),0.0,1.0,linestyles='--',linewidth=1,color='gray',alpha=0.5)
plt.vlines(np.linspace(0.0,1.0,num=10),0.0,1.0,linestyles='--',linewidth=1,color='gray',alpha=0.5)

plt.gca().set_color_cycle([cm(i) for i in np.linspace(0, 0.9, num_plots)])

i = 0
for t in m.t_realintervals:
    x_val = r
    y_val = m.C_values[i]
    plt.plot(x_val,y_val,label="t* = " + str(t))
    i+=1
plt.xlabel('r* = r/R')
plt.ylabel('C* = (C - Cfar)/(C0 - Cfar)')
plt.title('t* = t/tau, tau = R^2/D')
#plt.yscale('log')
plt.legend()
plt.show()