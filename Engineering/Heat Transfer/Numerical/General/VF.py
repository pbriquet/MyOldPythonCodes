from VF import *
from BoundaryConditions import *
import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["DesignPatterns","LinearAlgebra"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

from math import *
import numpy as np
import types
from enum import IntEnum
from Vector import *
from ConvexSolid import *

def Vertices_of_surfaces(i):
    if(i == Orientation.Neighbors.EAST):
        return [Orientation.Vertex.IOO, Orientation.Vertex.IIO, Orientation.Vertex.III, Orientation.Vertex.IOI]
    elif(i == Orientation.Neighbors.WEST):
        return [Orientation.Vertex.OOI, Orientation.Vertex.OII,  Orientation.Vertex.OIO, Orientation.Vertex.OOO]
    elif(i == Orientation.Neighbors.NORTH):
        return [Orientation.Vertex.OII, Orientation.Vertex.III, Orientation.Vertex.IIO, Orientation.Vertex.OIO]
    elif(i == Orientation.Neighbors.SOUTH):
        return [Orientation.Vertex.OOO, Orientation.Vertex.IOO, Orientation.Vertex.IOI, Orientation.Vertex.OOI]
    elif(i == Orientation.Neighbors.UP):
        return [Orientation.Vertex.OOI, Orientation.Vertex.IOI, Orientation.Vertex.III, Orientation.Vertex.OII]
    elif(i == Orientation.Neighbors.DOWN):
        return  [Orientation.Vertex.OIO, Orientation.Vertex.IIO,  Orientation.Vertex.IOO,Orientation.Vertex.OOO]

class VF:
    def __init__(self,mesh,material,index,center_pos):
        self.mesh = mesh
        self.index = index
        self.material = material
        self.P = center_pos
        self.T = 150.0
        self.Tdt = self.T
        self.vertex = [None]*8
        self.dt = 1.0
        self.lastT = self.T
        dx = mesh.dx
        uvw = mesh.coordinate.uvw(self.P)

        self.vertex[Orientation.Vertex.OOO] = mesh.coordinate.xyz(uvw + Vector(-dx[0] / 2.0, -dx[1] / 2.0, -dx[2] / 2.0))
        self.vertex[Orientation.Vertex.IOO] = mesh.coordinate.xyz(uvw + Vector(dx[0] / 2.0, -dx[1] / 2.0, -dx[2] / 2.0))
        self.vertex[Orientation.Vertex.OIO] = mesh.coordinate.xyz(uvw + Vector(-dx[0] / 2.0, dx[1] / 2.0, -dx[2] / 2.0))
        self.vertex[Orientation.Vertex.OOI] = mesh.coordinate.xyz(uvw + Vector(-dx[0] / 2.0, -dx[1] / 2.0, dx[2] / 2.0))
        self.vertex[Orientation.Vertex.IIO] = mesh.coordinate.xyz(uvw + Vector(dx[0] / 2.0, dx[1] / 2.0, -dx[2] / 2.0))
        self.vertex[Orientation.Vertex.IOI] = mesh.coordinate.xyz(uvw + Vector(dx[0] / 2.0, -dx[1] / 2.0, dx[2] / 2.0))
        self.vertex[Orientation.Vertex.OII] = mesh.coordinate.xyz(uvw + Vector(-dx[0] / 2.0, dx[1] / 2.0, dx[2] / 2.0))
        self.vertex[Orientation.Vertex.III] = mesh.coordinate.xyz(uvw + Vector(dx[0] / 2.0, dx[1] / 2.0, dx[2] / 2.0))
        
        dvertex = []
        for i in self.vertex:
            dvertex.append(i - self.P)
        self.solid = ConvexSolid(dvertex)
        boundaries = []
        for k in xrange(6):
            v = []
            i = Vertices_of_surfaces(k)
            for j in i:
                v.append(self.vertex[j])
            boundaries.append(Boundary(self,v))
        self.boundaries = boundaries
        self._calculategeometry()
        self._consolidatematerial()

    def _consolidatematerial(self):
        # K
        if(isinstance(self.material.K,types.FunctionType)):
            self.K = lambda: self.material.K(self.T)
        elif(isinstance(self.material.K,(float,int))):
            self.K = lambda: self.material.K

        # Cp
        if(isinstance(self.material.Cp,types.FunctionType)):
            self.Cp = lambda: self.material.Cp(self.T)
        elif(isinstance(self.material.Cp,(float,int))):
            self.Cp = lambda: self.material.Cp

        # rho
        if(isinstance(self.material.rho,types.FunctionType)):
            self.rho = lambda: self.material.rho(self.T)
        elif(isinstance(self.material.rho,(float,int))):
            self.rho = lambda: self.material.rho

    def setNeighbors(self,neighbors):
        self.neighbors = neighbors

    def _calculategeometry(self):
        tmp_dV = 0.0
        i = 0
        for b in self.boundaries:
            tmp_dV += b.dA_vec*b.P
            i += 1
        tmp_dV = tmp_dV/3.0
        self.dV = tmp_dV

    def __str__(self):
        return str(self.T)
    
    @property
    def dTdt(self):
        return (self.T - self.lastT)/self.dt

    @staticmethod
    def calculate_heatbalance(vf,dt):
        q = 0.0
        vf.dt = dt
        for b in vf.boundaries:
            q += b.bc.q()*b.dA
        q*=dt
        vf.Tdt = vf.T + q/vf.rho()/vf.Cp()/vf.dV

    @staticmethod
    def Refresh(vf):
        vf.lastT = vf.T
        vf.T = vf.Tdt

    def setT(self,T):
        self.T = T






