# Calcule o valor do NBO/T das seguintes escorias.

from slag import *

# Exemplo
x = Slag(('SiO2',0.361),('TiO2',0.0046),('Al2O3',0.131),('CaO',0.409),('MgO',0.077),('FeO',0.004),('MnO',0.0025),('Na2O',0.0020),('K2O',0.0050))

# Exercicio
a = Slag(('CaO',0.450), ('Al2O3',0.200), ('MnO',0.01), ('SiO2',0.34))
b = Slag(('CaO',0.42), ('Al2O3',0.15), ('SiO2',0.34), ('P2O5',0.01), ('MgO',0.07, ('Fe2O3',0.05)))
c = Slag(('CaO',0.55), ('Al2O3',0.02), ('MnO',0.025), ('SiO2',0.175), ('P2O5',0.01), ('MgO', 0.07), ('Fe2O3',0.15))

print "Escoria x: NBOT = " + str(x.NBOT) + " (Exemplo)"
print "Escoria a: NBOT = " + str(a.NBOT)
print "Escoria b: NBOT = " + str(b.NBOT)
print "Escoria c: NBOT = " + str(c.NBOT)
