# (pag 25) Calcule as seguintes basicidades para as 3 escorias indicadas a seguir.
# BB, BQ, BQ (Forno Panela)
# sum(ox. bas)/sum(ox. acidos)
# basicidade otica
# e o IB

from slag import *

# Exercicio
a = Slag(('CaO',45.0), ('Al2O3',20.0), ('MnO',1.0), ('SiO2',34.0))
b = Slag(('CaO',42.0), ('Al2O3',15.0), ('SiO2',34.0), ('P2O5',1.0), ('MgO',7.0, ('Fe2O3',5.0)))
c = Slag(('CaO',55.0), ('Al2O3',2.0), ('MnO',2.5), ('SiO2',17.5), ('P2O5',1.0), ('MgO', 7.0), ('Fe2O3',15.0))

Z = Slag(('SiO2',80.0),('CaO',10.0),('MgO',10.0))
X = Slag(('SiO2',50.0),('CaO',30.0),('MgO',20.0))
O = Slag(('SiO2',38.0),('CaO',45.0),('MgO',17.0))
K = Slag(('SiO2',35.0),('CaO',50.0),('MgO',15.0))
Y = Slag(('SiO2',20.0),('CaO',70.0),('MgO',10.0))

Slags = [a,b,c]
for i in Slags:
    print i.PrintBasicity()
    print '\n'
