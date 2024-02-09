
import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

from math import *
import re
import numpy as np
import copy
from enum import IntEnum
from Singleton import *
from TemperatureConverter import *

class Alloy:
    def __init__(self,percent='wt',balance='Fe',**kwargs):
        p = PeriodicTable()
        self.elements = dict()
        self.atomic = dict()

        sum_percent = 0.0
        for k,wt in kwargs.items():
            self.elements[k] = wt
            sum_percent += wt
        self.elements[balance] = 100.0 - sum_percent
        self._calculte_atomic()

    def _calculte_atomic(self):
        p = PeriodicTable()

        sum_wk_Mk = sum([self.elements[k]/p.elements[k].atomicweigth for k in self.elements.keys()])
        for k,wt in self.elements.items():
            self.atomic[k] = (wt/p.elements[k].atomicweigth)/sum_wk_Mk


    def __getitem__(self,arg):
        if(arg in self.elements):
            return self.elements[arg]
        else:
            return 0.0
    def items(self):
        return self.elements.items()

    def __str__(self):
        tmp = ''
        for k,v in self.elements.items():
            tmp += k + ': ' + str(v) + '\n'
        return tmp

class Isopleth:
    def __init__(self,percent='wt',balance='Fe',var_element='C',interval=[0.0,0.2],num=50,**kwargs):
        self.data = []
        points = np.linspace(interval[0],interval[1],num)
        tmp_dict = kwargs.copy()
        for i in points:
            tmp_dict[var_element] = i
            self.data.append(Alloy(percent=percent,balance=balance,**tmp_dict))


class Element:
    class Classification(IntEnum):
        AlkaliMetals = 0
        AlkaliEarthMetals = 1
        TransitionMetals = 2
        PostTransitionMetals = 3
        Metalloids = 4
        Nonmetals = 5
        Halogens = 6
        NobleGases = 7
        Lanthanides = 8
        Actinides = 9
        Unknown = 10
    
    def __init__(self,symbol,name,atomicnumber,atomicweigth,classification):
        self.name = name
        self.symbol = symbol
        self.atomicnumber = atomicnumber
        self.atomicweigth = atomicweigth
        self.classification = classification

    def __eq__(self,other):
        return self.atomicnumber == other.atomicnumber

    def __str__(self):
        tmp = "Atomic Number: " + str(self.atomicnumber)
        tmp += "\nName: " + str(self.name)
        tmp += "\nSymbol: " + str(self.symbol)
        tmp += "\nAtomic Weight: " + str(self.atomicweigth) + " u";
        return tmp
    

class PeriodicTableClassification(IntEnum):
        AlkaliMetals = 0
        AlkaliEarthMetals = 1
        TransitionMetals = 2
        PostTransitionMetals = 3
        Metalloids = 4
        Nonmetals = 5
        Halogens = 6
        NobleGases = 7
        Lanthanides = 8
        Actinides = 9
        Unknown = 10

@singleton
class PeriodicTable:
    def __init__(self):
        self.elements = dict()
        self.elements['H'] = Element('H','Hydrogen',1,1.007825,PeriodicTableClassification.Nonmetals)
        self.elements['He'] = Element('He','Helium',2,4.00260,PeriodicTableClassification.NobleGases)
        self.elements['Li'] = Element('Li','Lithium',3,6.941,PeriodicTableClassification.AlkaliMetals)
        self.elements['Be'] = Element('Be','Beryllium',4,9.01218,PeriodicTableClassification.AlkaliEarthMetals)
        self.elements['B'] = Element('B','Boron',5,10.81,PeriodicTableClassification.Metalloids)
        self.elements['C'] = Element('C','Carbon',6,12.011,PeriodicTableClassification.Nonmetals)
        self.elements['N'] = Element('N','Nitrogen',7,14.0067,PeriodicTableClassification.Nonmetals)
        self.elements['O'] = Element('O','Oxygen',8,15.999,PeriodicTableClassification.Nonmetals)
        self.elements['F'] = Element('F','Fluorine',9,18.99840,PeriodicTableClassification.Halogens)
        self.elements['Ne'] = Element('Ne','Neon',10,20.179,PeriodicTableClassification.NobleGases)
        self.elements['Na'] = Element('Na','Sodium',11,22.98977,PeriodicTableClassification.AlkaliMetals)
        self.elements['Mg'] = Element('Mg','Magnesium',12,24.305,PeriodicTableClassification.AlkaliEarthMetals)
        self.elements['Al'] = Element('Al','Aluminum',13,26.98154,PeriodicTableClassification.PostTransitionMetals)
        self.elements['Si'] = Element('Si','Silicon',14,28.0855,PeriodicTableClassification.Metalloids)
        self.elements['P'] = Element('P','Phosphorus',15,30.973762,PeriodicTableClassification.Nonmetals)
        self.elements['S'] = Element('S','Sulphur',16,32.06,PeriodicTableClassification.Nonmetals)
        self.elements['Cl'] = Element('Cl','Chlorine',17,35.453,PeriodicTableClassification.Halogens)
        self.elements['Ar'] = Element('Ar','Argon',18,39.948,PeriodicTableClassification.NobleGases)
        self.elements['K'] = Element('K','Potassium',19,39.0983,PeriodicTableClassification.AlkaliMetals)
        self.elements['Ca'] = Element('Ca','Calcium',20,40.08,PeriodicTableClassification.AlkaliEarthMetals)
        self.elements['Sc'] = Element('Sc','Scandium',21,44.9559,PeriodicTableClassification.TransitionMetals)
        self.elements['Ti'] = Element('Ti','Titanium',22,47.90,PeriodicTableClassification.TransitionMetals)
        self.elements['V'] = Element('V','Vanadium',23,50.9414,PeriodicTableClassification.TransitionMetals)
        self.elements['Cr'] = Element('Cr','Chromium',24,51.996,PeriodicTableClassification.TransitionMetals)
        self.elements['Mn'] = Element('Mn','Manganese',25,54.9380,PeriodicTableClassification.TransitionMetals)
        self.elements['Fe'] = Element('Fe','Iron',26,55.85,PeriodicTableClassification.TransitionMetals)
        self.elements['Co'] = Element('Co','Cobalt',27,58.9332,PeriodicTableClassification.TransitionMetals)
        self.elements['Ni'] = Element('Ni','Nickel',28,58.71,PeriodicTableClassification.TransitionMetals)
        self.elements['Cu'] = Element('Cu','Copper',29,63.546,PeriodicTableClassification.TransitionMetals)
        self.elements['Zn'] = Element('Zn','Zinc',30,65.37,PeriodicTableClassification.TransitionMetals)
        self.elements['Ga'] = Element('Ga','Gallium',31,69.72,PeriodicTableClassification.PostTransitionMetals)
        self.elements['Ge'] = Element('Ge','Germanium',32,72.59,PeriodicTableClassification.Metalloids)
        self.elements['As'] = Element('As','Arsenic',33,74.9216,PeriodicTableClassification.Metalloids)
        self.elements['Se'] = Element('Se','Selenium',34,78.96,PeriodicTableClassification.Nonmetals)
        self.elements['Br'] = Element('Br','Bromine',35,79.904,PeriodicTableClassification.Halogens)
        self.elements['Kr'] = Element('Kr','Krypton',36,83.80,PeriodicTableClassification.NobleGases)
        self.elements['Rb'] = Element('Rb','Rubidium',37,85.4678,PeriodicTableClassification.AlkaliMetals)
        self.elements['Sr'] = Element('Sr','Strontium',38,87.62,PeriodicTableClassification.AlkaliEarthMetals)
        self.elements['Y'] = Element('Y','Yttrium',39,88.9059,PeriodicTableClassification.TransitionMetals)
        self.elements['Zr'] = Element('Zr','Zirconium',40,91.22,PeriodicTableClassification.TransitionMetals)
        self.elements['Nb'] = Element('Nb','Niobium',41,92.91,PeriodicTableClassification.TransitionMetals)
        self.elements['Mo'] = Element('Mo','Molybdenum',42,95.94,PeriodicTableClassification.TransitionMetals)
        self.elements['Tc'] = Element('Tc','Technetium',43,99.0,PeriodicTableClassification.TransitionMetals)
        self.elements['Ru'] = Element('Ru','Ruthenium',44,101.1,PeriodicTableClassification.TransitionMetals)
        self.elements['Rh'] = Element('Rh','Rhodium',45,102.91,PeriodicTableClassification.TransitionMetals)
        self.elements['Pd'] = Element('Pd','Palladium',46,106.42,PeriodicTableClassification.TransitionMetals)
        self.elements['Ag'] = Element('Ag','Silver',47,107.87,PeriodicTableClassification.TransitionMetals)
        self.elements['Cd'] = Element('Cd','Cadmium',48,112.4,PeriodicTableClassification.TransitionMetals)
        self.elements['In'] = Element('In','Indium',49,114.82,PeriodicTableClassification.PostTransitionMetals)
        self.elements['Sn'] = Element('Sn','Tin',50,118.69,PeriodicTableClassification.PostTransitionMetals)
        self.elements['Sb'] = Element('Sb','Antimony',51,121.75,PeriodicTableClassification.Metalloids)
        self.elements['Te'] = Element('Te','Tellurium',52,127.6,PeriodicTableClassification.Metalloids)
        self.elements['I'] = Element('I','Iodine',53,126.9045,PeriodicTableClassification.Halogens)
        self.elements['Xe'] = Element('Xe','Xenon',54,131.29,PeriodicTableClassification.NobleGases)
        self.elements['Cs'] = Element('Cs','Cesium',55,132.9054,PeriodicTableClassification.AlkaliMetals)
        self.elements['Ba'] = Element('Ba','Barium',56,137.33,PeriodicTableClassification.AlkaliEarthMetals)
        self.elements['La'] = Element('La','Lanthanum',57,138.91,PeriodicTableClassification.Lanthanides)
        self.elements['Ce'] = Element('Ce','Cerium',58,140.12,PeriodicTableClassification.Lanthanides)
        self.elements['Pr'] = Element('Pr','Praseodymium',59,140.91,PeriodicTableClassification.Lanthanides)
        self.elements['Nd'] = Element('Nd','Neodymium',60,144.242,PeriodicTableClassification.Lanthanides)
        self.elements['Pm'] = Element('Pm','Promethium',61,147.0,PeriodicTableClassification.Lanthanides)
        self.elements['Sm'] = Element('Sm','Samarium',62,150.35,PeriodicTableClassification.Lanthanides)
        self.elements['Eu'] = Element('Eu','Europium',63,167.26,PeriodicTableClassification.Lanthanides)
        self.elements['Gd'] = Element('Gd','Gadolinium',64,157.25,PeriodicTableClassification.Lanthanides)
        self.elements['Tb'] = Element('Tb','Terbium',65,158.925,PeriodicTableClassification.Lanthanides)
        self.elements['Dy'] = Element('Dy','Dysprosium',66,162.50,PeriodicTableClassification.Lanthanides)
        self.elements['Ho'] = Element('Ho','Holmium',67,164.9,PeriodicTableClassification.Lanthanides)
        self.elements['Er'] = Element('Er','Erbium',68,167.26,PeriodicTableClassification.Lanthanides)
        self.elements['Tm'] = Element('Tm','Thulium',69,168.93,PeriodicTableClassification.Lanthanides)
        self.elements['Yb'] = Element('Yb','Ytterbium',70,173.04,PeriodicTableClassification.Lanthanides)
        self.elements['Lu'] = Element('Lu','Lutetium',71,174.97,PeriodicTableClassification.Lanthanides)
        self.elements['Hf'] = Element('Hf','Hafnium',72,178.49,PeriodicTableClassification.TransitionMetals)
        self.elements['Ta'] = Element('Ta','Tantalum',73,180.95,PeriodicTableClassification.TransitionMetals)
        self.elements['W'] = Element('W','Tungsten',74,183.85,PeriodicTableClassification.TransitionMetals)
        self.elements['Re'] = Element('Re','Rhenium',75,186.23,PeriodicTableClassification.TransitionMetals)
        self.elements['Os'] = Element('Os','Osmium',76,190.2,PeriodicTableClassification.TransitionMetals)
        self.elements['Ir'] = Element('Ir','Iridium',77,192.2,PeriodicTableClassification.TransitionMetals)
        self.elements['Pt'] = Element('Pt','Platinum',78,195.09,PeriodicTableClassification.TransitionMetals)
        self.elements['Au'] = Element('Au','Gold',79,196.9655,PeriodicTableClassification.TransitionMetals)
        self.elements['Hg'] = Element('Hg','Mercury',80,200.59,PeriodicTableClassification.TransitionMetals)
        self.elements['Tl'] = Element('Tl','Thallium',81,204.383,PeriodicTableClassification.PostTransitionMetals)
        self.elements['Pb'] = Element('Pb','Lead',82,207.2,PeriodicTableClassification.PostTransitionMetals)
        self.elements['Bi'] = Element('Bi','Bismuth',83,208.9804,PeriodicTableClassification.PostTransitionMetals)
        self.elements['Po'] = Element('Po','Polonium',84,210.0,PeriodicTableClassification.PostTransitionMetals)
        self.elements['At'] = Element('At','Astatine',85,210.0,PeriodicTableClassification.Halogens)
        self.elements['Rn'] = Element('Rn','Radon',86,222.0,PeriodicTableClassification.NobleGases)
        self.elements['Fr'] = Element('Fr','Francium',87,233.0,PeriodicTableClassification.AlkaliMetals)
        self.elements['Ra'] = Element('Ra','Radium',88,226.0254,PeriodicTableClassification.AlkaliEarthMetals)
        self.elements['Ac'] = Element('Ac','Actinium',89,227.0,PeriodicTableClassification.Actinides)
        self.elements['Th'] = Element('Th','Thorium',90,232.04,PeriodicTableClassification.Actinides)
        self.elements['Pa'] = Element('Pa','Protactinium',91,231.0359,PeriodicTableClassification.Actinides)
        self.elements['U'] = Element('U','Uranium',92,238.03,PeriodicTableClassification.Actinides)
        self.elements['Np'] = Element('Np','Neptunium',93,237.0,PeriodicTableClassification.Actinides)
        self.elements['Pu'] = Element('Pu','Plutonium',94,244.0,PeriodicTableClassification.Actinides)
        self.elements['Am'] = Element('Am','Americium',95,243.0,PeriodicTableClassification.Actinides)
        self.elements['Cm'] = Element('Cm','Curium',96,247.0,PeriodicTableClassification.Actinides)
        self.elements['Bk'] = Element('Bk','Berkelium',97,247.0,PeriodicTableClassification.Actinides)
        self.elements['Cf'] = Element('Cf','Californium',98,251.0,PeriodicTableClassification.Actinides)
        self.elements['Es'] = Element('Es','Einsteinium',99,254.0,PeriodicTableClassification.Actinides)
        self.elements['Fm'] = Element('Fm','Fermium',100,257.0,PeriodicTableClassification.Actinides)
        self.elements['Md'] = Element('Md','Mendelevium',101,258.0,PeriodicTableClassification.Actinides)
        self.elements['No'] = Element('No','Nobelium',102,259.0,PeriodicTableClassification.Actinides)
        self.elements['Lr'] = Element('Lr','Lawrencium',103,262.0,PeriodicTableClassification.Actinides)
        self.elements['Rf'] = Element('Rf','Rutherfordium',104,260.9,PeriodicTableClassification.TransitionMetals)
        self.elements['Db'] = Element('Db','Dubnium',105,261.9,PeriodicTableClassification.TransitionMetals)
        self.elements['Sg'] = Element('Sg','Seaborgium',106,262.94,PeriodicTableClassification.TransitionMetals)
        self.elements['Bh'] = Element('Bh','Bohrium',107,262.0,PeriodicTableClassification.TransitionMetals)
        self.elements['Hs'] = Element('Hs','Hassium',108,264.8,PeriodicTableClassification.TransitionMetals)
        self.elements['Mt'] = Element('Mt','Meitnerium',109,265.9,PeriodicTableClassification.Unknown)
        self.elements['Ds'] = Element('Ds','Darmstadtium',110,261.9,PeriodicTableClassification.Unknown)
        self.elements['Ds'] = Element('Rg','Roentgenium',111,268.0,PeriodicTableClassification.Unknown)
        self.elements['Uub'] = Element('Cn','Copernicum',112,268.0,PeriodicTableClassification.TransitionMetals)
        self.elements['Uuq'] = Element('Uuq','Ununquadium',114,289.0,PeriodicTableClassification.Unknown)
        self.elements['Uuh'] = Element('Uuh','Ununhexium',116,292,PeriodicTableClassification.Unknown)
        self.elements_strings = self.elements.keys()

class ChemicalCompound:
    def __init__(self,Formula):
        p = PeriodicTable()
        self.formula = Formula
        critera1 = r'\D[A-Z]'
        critera2 = r'\D$'
        c1 = re.search(critera1,Formula)
        c2 = re.search(critera2,Formula)
        l = Formula
        if(c1!=None):
            l = re.sub(r'(\w)([A-Z])', r'\1 1\2', l)
        if(c2!=None):
            l = re.sub(r'(\w$)', r'\1 1', l)
        l = re.sub(' ', '', l)
        el = re.findall('\D+',l)
        self.elements = []
        for i in el:
            self.elements.append(copy.copy(p.elements[i]))
        self.stechiometry = [float(x) for x in re.findall('\d+',l)]
        self._calculate_molecularmass()

    def _calculate_molecularmass(self):
        self.atomweightfraction = []

        tmp = 0.0
        for i in range(len(self.elements)):
            tmp += self.stechiometry[i]*self.elements[i].atomicweigth
        for i in range(len(self.elements)):
            self.atomweightfraction.append(self.stechiometry[i]*self.elements[i].atomicweigth/tmp)
        self.molecularmass = tmp
    def __eq__(self,other):
        return self.formula == other.formula
    def __str__(self):
        tmp = self.formula + '\n'
        for i in range(len(self.elements)):
            tmp += self.elements[i].name + ' ' + str(self.stechiometry[i]) + '\n'
        return tmp


