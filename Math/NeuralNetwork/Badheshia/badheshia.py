from math import *
import re
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
f = open(os.path.join(__location__,"PeetBadheshiaThermalConductivity.csv"))

scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = scientific_match + '|' + float_match
max_re = 'max'
split_char = ';'

class data:
    def __init__(self,**kwargs):
        not_compute = ['Label','References\n','k','Cond']
        self.name = kwargs['Label']
        self.ref = kwargs['References\n']
        self.cond = kwargs['Cond']
        self.k = kwargs['k']
        self.T = kwargs['T']
        self.comp_min = dict()
        self.comp_max = dict()

        for key,val in kwargs.iteritems():
            if key in not_compute:
                pass
            else:
                if(re.match(max_re,val)):
                    self.comp_min[key] = 0.0
                    self.comp_max[key] = float(re.findall(numbers_match,val)[0])
                elif(len(re.findall(numbers_match,val))==2):
                    self.comp_min[key] = float(re.findall(numbers_match,val)[0])
                    self.comp_max[key] = float(re.findall(numbers_match,val)[1])
                elif(len(re.findall(numbers_match,val))==1):
                    self.comp_min[key] = float(re.findall(numbers_match,val)[0])
                    self.comp_max[key] = float(re.findall(numbers_match,val)[0])


all_data = []

i = 0

for line in f:
    if(i != 0):
        values = line.split(split_char)
        d = dict()
        for j in xrange(len(keys)):
            d[keys[j]] = values[j]
        all_data.append(data(**d))
    else:
        keys = line.split(split_char)
    i += 1

print all_data[0].comp_min.keys()
counter = 0
for i in all_data:
    if(i.k != '-'):
        counter+=1

print counter