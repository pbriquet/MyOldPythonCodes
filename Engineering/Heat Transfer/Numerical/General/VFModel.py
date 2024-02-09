import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["DesignPatterns","LinearAlgebra"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

from VFMesh import *
from enum import IntEnum
from math import *
import types
from Vector import *

class BoundaryIndex(IntEnum):
        EAST = 0
        WEST = 1
        NORTH = 2
        SOUTH = 3
        UP = 4
        DOWN = 5
class Material:
    def __init__(self,**kwargs):
        self.K = kwargs['K']
        self.rho = kwargs['rho']
        self.Cp = kwargs['Cp']
class Thermopair:
    def __init__(self,model,pos,dt):
        self.model = model
        self.pos = Vector(pos[0],pos[1],pos[2])
        self.T = []
        self.t = []
        self.last_t = 0.0
        self.dt = dt
        self.Vijk = [[[0 for i in range(2)] for j in range(2)] for k in range(2)]
        self.fijk = [[[0 for i in range(2)] for j in range(2)] for k in range(2)]
        self._find_interpolation()

    def _find_interpolation(self):
        xn, yn, zn = [1.0,0.0,0.0]
        self._find_quadrant()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.fijk[i][j][k] = (1 - i + (-1)**i*xn)*(1 - j + (-1)**j*yn)*(1 - k + (-1)**k*zn)
                    self.Vijk[i][j][k] = (1 - i + (-1)**i*xn)*(1 - j + (-1)**j*yn)*(1 - k + (-1)**k*zn)

    def _find_quadrant(self):
        uvw = self.model.mesh.coordinate.uvw(self.pos)
        self.ijk = (0,0,0)
        for i in xrange(self.model.mesh.nx[0]):
            for j in xrange(self.model.mesh.nx[1]):
                for k in xrange(self.model.mesh.nx[2]):
                    if(self.model.mesh.vfs[i][j][k].solid.checkPointBelongs(self.pos - self.model.mesh.vfs[i][j][k].P)):
                        self.ijk = (i,j,k)
                        print(self.ijk)
                        
                    

    def do_record(self,t):
        if(t > self.last_t + self.dt):
            self.last_t = t
            self._record(t)
        else:
            pass

    def _record(self,t):
        self.t.append(t)
        T = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T += self.fijk[i][j][k]*self.Vijk[i][j][k]
        self.T.append(T)
    
class VFModel:
    def __init__(self):
        self.t = 0.0
        self.thermopairs = []
    def setMesh(self,**kwargs):
        Lx = kwargs['Lx']
        nx = kwargs['nx']
        if('coordinates' in kwargs):
            coordinates = kwargs['coordinates']
        else:
            coordinates = RectangularSystem()
        self.dim = min(len(Lx),len(nx))
        self.mesh = VFMesh(self,Lx,nx,self.material,coordinates)
    def addThermopair(self,thermopair):
        self.thermopairs.append(thermopair)
    
    def setBoundaryConditions(self,index,data):
        self.mesh.boundaries_setted[index] = True
        for boundary in self.mesh.boundaries[index]:
            boundary.setBoundaryCondition(data)

    def setInitialConditions(self,**kwargs):
        T0 = kwargs['T0']
        if(isinstance(T0,types.FunctionType)):
            g = lambda vf: vf.setT(T0(vf.P))
        elif(isinstance(T0,(int,float))):
            g = lambda vf: vf.setT(T0)

        self.mesh.Iterate(g)
    def setMaterial(self,**kwargs):
        self.material = Material(**kwargs)

    def _initializedata(self):
        data = AdiabaticBoundaryConditionData()
        for i in xrange(6):
            if(not self.mesh.boundaries_setted[i]):
                for b in self.mesh.boundaries[i]:
                    b.setBoundaryCondition(data)
                self.mesh.boundaries_setted[i] = True

    def RunTime(self,tmax,dtmax,adjust_dt=False):
        self._initializedata()
        dt = dtmax
        i = 0
        while(self.t < tmax):
            if(i%100==0.0):
                print('t = ' + str(self.t))
            for k in self.thermopairs:
                k.do_record(self.t)
            self.mesh.Iterate(lambda vf: VF.calculate_heatbalance(vf,dt))
            self.mesh.Iterate(VF.Refresh)
            self.t += dt
            i += 1

    def Interpolate(self,pos):
        return 0.5