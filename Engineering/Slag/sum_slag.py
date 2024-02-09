from slag import *

class SumSlag:
    def __init__(self,slags,mass_percent):
        self.slags = slags
        self.mass_percent = mass_percent
    def calculate_slag(self):
        oxides = dict()
        k = 0
        for k_slag in self.slags:
            j = 0
            for j_oxide in k_slag.oxides:
                if(j_oxide.formula != 'Al'):
                    formula = j_oxide.formula
                else:
                    formula = j_oxide.formula
                if(formula not in oxides.keys()):
                    oxides[formula] = k_slag.mass_percent[j]*self.mass_percent[k]
                else:
                    oxides[formula] += k_slag.mass_percent[j]*self.mass_percent[k]
                j+=1
            k+=1
        tuples = []
        i = 0
        for i_oxide, j_mass in oxides.items():
            tuples.append((i_oxide,j_mass))
            i+=1
        
        print(tuples)
        sl = Slag(*tuples)
        return sl