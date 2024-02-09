import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

from math import *
from Vector import *

class RectangularSystem():
    def __init__(self,*args,**kwargs):
        pass
    def xyz(self,uvw):
        return Vector(uvw[0],uvw[1],uvw[2])
    def uvw(self,xyz):
        return Vector(xyz[0],xyz[1],xyz[2])
    def h(self,uvw):
        return Vector(1.0,1.0,1.0)
    def J(self,uvw):
        return 1.0
    
class CylindricalSystem():
    def __init__(self,*args,**kwargs):
        pass
    def xyz(self,uvw):
        return Vector(uvw[0]*cos(uvw[1]),uvw[0]*sin(uvw[1]),uvw[2])
    def uvw(self,xyz):
        return Vector(sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2), atan(xyz[1]/xyz[0]) if xyz[0] != 0.0 else 0.0, xyz[2])
    def h(self,uvw):
        return Vector(1.0,uvw[0],1.0)
    def J(self,uvw):
        return uvw[0]

class SphericalSystem():
    def __init__(self,*args,**kwargs):
        pass
    def xyz(self,uvw):
        return Vector(uvw[0]*sin(uvw[1])*cos(uvw[2]),uvw[0]*sin(uvw[1])*sin(uvw[2]),uvw[0]*cos(uvw[2]))
    def uvw(self,xyz):
        return xyz
    def h(self,uvw):
        return Vector(1.0,uvw[0],uvw[0]*sin(uvw[1]))
    def J(self,uvw):
        return uvw[0]**2*cos(uvw[1])

