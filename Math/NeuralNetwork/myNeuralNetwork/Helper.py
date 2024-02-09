import numpy as np
import random as rnd
from math import *

class Helper:
    @staticmethod
    def ValidateNotNull(value,name):
        if(value==None):
            raise TypeError(name + ' is Null Reference')
    @staticmethod
    def ValidateEnum(enumType,value,name):
        if(value==None):
            raise TypeError(name + 'Null Reference')
    @staticmethod
    def ValidateNotNegative(value,name):
        if(value<0):
            raise ValueError(name + ' is Negative')
    @staticmethod
    def ValidatePositive(value,name):
        if(value<0):
            raise ValueError(name + ' is not Positive')
    @staticmethod
    def ValidateWithinRange(vector):
        pass
    @staticmethod
    def GetRandom(min_Limit,max_Limit):
        if(min_Limit > max_Limit):
            return rnd.uniform(max_Limit,min_Limit)
        else:
            return rnd.uniform(min_Limit,max_Limit)
    @staticmethod
    def GetRandom1():
        return rnd.uniform(0.0,1.0)
    @staticmethod
    def GetRandomOrder(length):
        tmp = range(length)
        #rnd.shuffle(tmp)
        return tmp
    @staticmethod
    def Normalize(vector,magnitude=1.0):
        factor = 0.0
        tmp = []
        for i in xrange(len(vector)):
            factor += vector[i]**2
        factor = sqrt(factor)
        if(factor == 0.0):
            return vector
        else:
            for j in xrange(len(vector)):
                tmp.append(vector[j]/factor)
            return tmp
        
    @staticmethod
    def GetRandomVector(count,magnitude):
        result = []
        for i in xrange(count):
            result.append(Helper.GetRandom1())
        return result