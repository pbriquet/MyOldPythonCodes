from math import *
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.dT0 = 0.0
        self.dT = self.dT0
        self.R =20.0
        self.tmax = 1.0
        self.dtmax = 5e-3
        self.dt = self.dtmax
        self.G = lambda r,dT: 1.0e-8*dT**2
        self.fg = 0.0
        self.dTn = 1.0
        self.t = 0.0
        self.nuclei = []
        self.nuclei.append(GrainPopulation.getGrainPopulationDistribution(5.0,10.0,1e8))
        self.nuclei.append(GrainPopulation.getGrainPopulationDistribution(20.0,10.0,2e8))
        self.population = []

    def Run(self,tmax=1.0):
        self.t = 0.0
        self.tmax=tmax
        while(self.t < self.tmax):
            for k in self.population:
                k.grow(self.G(k.r,self.dT),self.dt)
            for distribution in self.nuclei:
                for k in distribution:
                    if(k.dTn < self.dT):
                        k.nucleate()
                        self.population.append(k)
                        distribution.remove(k)
            self.t += self.dtmax
            self.dT += self.R*self.dt

    def getSizePopulation(self,dr=5e-7,rmax=1.2e-4):
        r = 0.0
        tmp = []
        while(r <= rmax):
            dn = 0.0
            for k in self.population:
                if(k.r >= r and k.r < r + dr):
                    dn += k.n
            tmp.append((r + dr/2.0,dn))
            r += dr
        return tmp

class GrainPopulation:
    def __init__(self,dTn,n):
        self.dTn = dTn
        self.n = n
        self.nucleated = False
    def nucleate(self):
        self.nucleated = True
        self.r = 0.0
    def grow(self,G,dt):
        self.r += G*dt
    @staticmethod
    def NormalDistribution(dT,dTu,dTs):
        return 1.0/sqrt(2.0*pi)/dTs*exp(-(dT-dTu)**2/2.0/dTs**2)
    @staticmethod
    def getGrainPopulationDistribution(dTu,dTs,nT,ddT=0.001,dTmax=10.0):
        dT = 0.0
        tmp = []
        while(dT<dTmax):
            dTnext = dT + ddT
            na = nT*GrainPopulation.NormalDistribution(dT,dTu,dTs)
            nb = nT*GrainPopulation.NormalDistribution(dTnext,dTu,dTs)
            dn = (na + nb)*ddT/2.0
            tmp.append(GrainPopulation((dT + dTnext)/2.0,dn))
            dT =dTnext
        return tmp

m = []
m.append(Model())
m.append(Model())
#m.append(Model())

m[0].Run(tmax=1.0)
print('Model 1 complete')
m[1].Run(tmax=1.5)
print('Model 2 complete')
#m[2].Run(tmax=10.0)
#print('Model 3 complete')

for model in m:
    x = []
    y = []
    for k in model.getSizePopulation():
        x.append(k[0])
        y.append(k[1])
    plt.plot(x,y)
plt.show()
#m.Run()

