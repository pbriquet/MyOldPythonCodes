from VF import *
from BoundaryConditions import *
import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["DesignPatterns","LinearAlgebra"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

from math import *
from Vector import *
from CoordinateSystem import *

class VFMesh:
    def __call__(self,i,j,k):
        return self.vfs[i][j][k]
    def __init__(self,model,Lx,nx,material,coordinate_system):
        self.model = model
        self.Lx = Lx
        self.nx = nx
        self.boundaries = [[],[],[],[],[],[]]
        self.boundaries_setted = [False]*6
        self.coordinate = coordinate_system

        if(len(Lx)<3):
            for i in range(len(Lx),3):
                self.Lx.append(1e-3)
                self.nx.append(1)
        self.dx = map(lambda x,y: x/y, self.Lx,self.nx)

        self.vfs = []
        print("Initializing...")
        list_i = list(map(int,range(self.nx[0])))
        list_j = list(map(int,range(self.nx[1])))
        list_k = list(map(int,range(self.nx[2])))
        print(list_i)
        for i in list_i:
            self.vfs.append([])
            for j in list_j:
                self.vfs[i].append([])
                for k in list_k:
                    vf = VF(self,material,(i,j,k),self.coordinate.xyz(Vector((i+0.5)*self.dx[0],(j+0.5)*self.dx[1],(k+0.5*self.dx[2]))))
                    self.vfs[i][j].append(vf)

        print("Mesh Initialized. Setting Neighbors")
        self.getNeighbors()
        print("Done.")

    def Iterate(self,action):
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.nx[2]):
                    action(self.vfs[i][j][k])
    
    def getNeighbors(self):
        for i in xrange(self.nx[0]):
            for j in xrange(self.nx[1]):
                for k in xrange(self.nx[2]):
                    neighbors = []
                    if(i!=self.nx[0]-1):
                        neighbors.append(self.vfs[i+1][j][k])
                    else:
                        neighbors.append(None)
                    if(i!=0):
                        neighbors.append(self.vfs[i-1][j][k])
                    else:
                        neighbors.append(None)
                    if(j!=self.nx[1]-1):
                        neighbors.append(self.vfs[i][j+1][k])
                    else:
                        neighbors.append(None)
                    if(j!=0):
                        neighbors.append(self.vfs[i][j-1][k])
                    else:
                        neighbors.append(None)
                    if(k!=self.nx[2]-1):
                        neighbors.append(self.vfs[i][j][k+1])
                    else:
                        neighbors.append(None)
                    if(k!=0):
                        neighbors.append(self.vfs[i][j][k-1])
                    else:
                        neighbors.append(None)
                    self.vfs[i][j][k].setNeighbors(neighbors)
                    n = 0
                    for neighbor in self.vfs[i][j][k].neighbors:
                        boundary = self.vfs[i][j][k].boundaries[n]
                        if(neighbor != None):
                            boundary.setBoundaryCondition(BoundaryBetweenVFsData(self.vfs[i][j][k],neighbor))
                        else:
                            self.boundaries[n].append(boundary)
                        n+=1