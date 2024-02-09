from math import *

class Vector3:
    def __init__(self,*args,**kwargs):
        self.v = [0.0,0.0,0.0]
        for i in xrange(len(args)):
            self.v[i] = args[i]
        self.x = self.v[0]
        self.y = self.v[1]
    
    @property
    def x(self):
        return self.v[0]
    @property
    def y(self):
        return self.v[1]
    @property
    def z(self):
        return self.v[2]

    def __add__(self,other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self,other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self,other):
        if(type(other) is Vector3):
            return dot(self,other)
        elif(type(other) is float or type(other) is int):
            return Vector3(self.x*other,self.y*other,self.z*other)
    def __mul__(other,self):
        if(other is Vector3):
            return dot(self,other)
        elif(other is float or other is int):
            return Vector3(self.x*other,self.y*other,self.z*other)
    def dot(self,other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __str__(self):
        return '[' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ']'


a = Vector3(1.0,3.0,2.0)
b = Vector3(4.0,5.0,3.0)
print 2.0*a
