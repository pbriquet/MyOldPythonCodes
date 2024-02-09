import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["DesignPatterns","LinearAlgebra"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

from math import *
from Vector import *
import types

class Boundary:
    def __init__(self,vf,vertex):
        self.vf = vf
        self.vertex = vertex
        self._calculategeometry()
    def _calculategeometry(self):
        self.dA_vec = 0.5*((self.vertex[2] - self.vertex[0]).cross(self.vertex[3] - self.vertex[1]))
        self.dA = self.dA_vec.norm()
        tmp = Vector(0.0,0.0,0.0)
        for i in self.vertex:
            tmp += i
        self.P = tmp/4.0

    def setBoundaryCondition(self,boundary_condition_data):
        if(isinstance(boundary_condition_data,AdiabaticBoundaryConditionData)):
            self.bc = AdiabaticBoundaryCondition(self)
        elif(isinstance(boundary_condition_data,NewtonBoundaryConditionData)):
            self.bc = NewtonBoundaryCondition(self,boundary_condition_data)
        elif(isinstance(boundary_condition_data,NeumannBoundaryConditionData)):
            self.bc = NeumannBoundaryCondition(self,boundary_condition_data)
        elif(isinstance(boundary_condition_data,DirichletBoundaryConditionData)):
            self.bc = DirichletBoundaryCondition(self,boundary_condition_data)
        elif(isinstance(boundary_condition_data,BoundaryBetweenVFsData)):
            self.bc = BoundaryBetweenVFs(self,boundary_condition_data)
        else:
            self.bc = AdiabaticBoundaryCondition(self)

class BoundaryBetweenVFsData(object):
    def __init__(self,vf,neigh):
        self.vf = vf
        self.neigh = neigh
class BoundaryBetweenVFs:
    def __init__(self,boundary,data):
        self.boundary = boundary
        self.vf = data.vf
        self.neigh = data.neigh
        self.dx_p = (self.boundary.P - self.vf.P).norm()
        self.dx_o = (self.boundary.P - self.neigh.P).norm()
    def q(self):
        dx_Kp = self.dx_p/self.vf.K()
        dx_Ko = self.dx_o/self.neigh.K()
        _q = -(self.vf.T - self.neigh.T)/(dx_Kp + dx_Ko)
        return _q

class DirichletBoundaryConditionData(object):
    def __init__(self,**kwargs):
        self.T = kwargs['T']
class DirichletBoundaryCondition:
    def __init__(self,boundary,data):
        self.vf = vf
        self.T = data.T

    def q(self):
        return 0.0

class NeumannBoundaryConditionData:
    def __init__(self,**kwargs):
        self.q = kwargs['q']

class NeumannBoundaryCondition:
    def __init__(self,boundary,data):
        self.vf = boundary.vf
        if(isinstance(data.q,types.FunctionType)):
            self._q = data.q
        elif(isinstance(data.q,(float,int))):
            self._q = lambda: data.q

    def q(self):
        return self._q()

class AdiabaticBoundaryConditionData:
    def __init__(self,**kwargs):
        pass
class AdiabaticBoundaryCondition:
    def __init__(self,boundary):
        pass
    def q(self):
        return 0.0

class NewtonBoundaryConditionData:
    def __init__(self,**kwargs):
        self.h = kwargs['h']
        self.Tfar = kwargs['Tfar']

class NewtonBoundaryCondition:
    def __init__(self,boundary,data):
        self.boundary = boundary
        self.h = data.h
        self.Tfar = data.Tfar
        self.dx_p = (self.boundary.P - self.boundary.vf.P).norm()

    def q(self):
        Kp_dx = self.boundary.vf.K() / self.dx_p
        h_ = self.h
        Tfar_ = self.Tfar
        Ti_ = (Kp_dx * self.boundary.vf.T + h_ * Tfar_) / (Kp_dx + h_)
        q_ = -h_ * (Ti_ - Tfar_)
        return q_

