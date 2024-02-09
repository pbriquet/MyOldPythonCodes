from math import *
import re
import copy
from enum import IntEnum

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
    

class PeriodicTable:
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
    def __init__(self):
        self.elements = dict()
        self.elements['H'] = Element('H','Hydrogen',1,1.007825,PeriodicTable.Classification.Nonmetals)
        self.elements['He'] = Element('He','Helium',2,4.00260,PeriodicTable.Classification.NobleGases)
        self.elements['Li'] = Element('Li','Lithium',3,6.941,PeriodicTable.Classification.AlkaliMetals)
        self.elements['Be'] = Element('Be','Beryllium',4,9.01218,PeriodicTable.Classification.AlkaliEarthMetals)
        self.elements['B'] = Element('B','Boron',5,10.81,PeriodicTable.Classification.Metalloids)
        self.elements['C'] = Element('C','Carbon',6,12.011,PeriodicTable.Classification.Nonmetals)
        self.elements['N'] = Element('N','Nitrogen',7,14.0067,PeriodicTable.Classification.Nonmetals)
        self.elements['O'] = Element('O','Oxygen',8,15.999,PeriodicTable.Classification.Nonmetals)
        self.elements['F'] = Element('F','Fluorine',9,18.99840,PeriodicTable.Classification.Halogens)
        self.elements['Ne'] = Element('Ne','Neon',10,20.179,PeriodicTable.Classification.NobleGases)
        self.elements['Na'] = Element('Na','Sodium',11,22.98977,PeriodicTable.Classification.AlkaliMetals)
        self.elements['Mg'] = Element('Mg','Magnesium',12,24.305,PeriodicTable.Classification.AlkaliEarthMetals)
        self.elements['Al'] = Element('Al','Aluminum',13,26.98154,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['Si'] = Element('Si','Silicon',14,28.0855,PeriodicTable.Classification.Metalloids)
        self.elements['P'] = Element('P','Phosphorus',15,30.973762,PeriodicTable.Classification.Nonmetals)
        self.elements['S'] = Element('S','Sulphur',16,32.06,PeriodicTable.Classification.Nonmetals)
        self.elements['Cl'] = Element('Cl','Chlorine',17,35.453,PeriodicTable.Classification.Halogens)
        self.elements['Ar'] = Element('Ar','Argon',18,39.948,PeriodicTable.Classification.NobleGases)
        self.elements['K'] = Element('K','Potassium',19,39.0983,PeriodicTable.Classification.AlkaliMetals)
        self.elements['Ca'] = Element('Ca','Calcium',20,40.08,PeriodicTable.Classification.AlkaliEarthMetals)
        self.elements['Sc'] = Element('Sc','Scandium',21,44.9559,PeriodicTable.Classification.TransitionMetals)
        self.elements['Ti'] = Element('Ti','Titanium',22,47.90,PeriodicTable.Classification.TransitionMetals)
        self.elements['V'] = Element('V','Vanadium',23,50.9414,PeriodicTable.Classification.TransitionMetals)
        self.elements['Cr'] = Element('Cr','Chromium',24,51.996,PeriodicTable.Classification.TransitionMetals)
        self.elements['Mn'] = Element('Mn','Manganese',25,54.9380,PeriodicTable.Classification.TransitionMetals)
        self.elements['Fe'] = Element('Fe','Iron',26,55.85,PeriodicTable.Classification.TransitionMetals)
        self.elements['Co'] = Element('Co','Cobalt',27,58.9332,PeriodicTable.Classification.TransitionMetals)
        self.elements['Ni'] = Element('Ni','Nickel',28,58.71,PeriodicTable.Classification.TransitionMetals)
        self.elements['Cu'] = Element('Cu','Copper',29,63.546,PeriodicTable.Classification.TransitionMetals)
        self.elements['Zn'] = Element('Zn','Zinc',30,65.37,PeriodicTable.Classification.TransitionMetals)
        self.elements['Ga'] = Element('Ga','Gallium',31,69.72,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['Ge'] = Element('Ge','Germanium',32,72.59,PeriodicTable.Classification.Metalloids)
        self.elements['As'] = Element('As','Arsenic',33,74.9216,PeriodicTable.Classification.Metalloids)
        self.elements['Se'] = Element('Se','Selenium',34,78.96,PeriodicTable.Classification.Nonmetals)
        self.elements['Br'] = Element('Br','Bromine',35,79.904,PeriodicTable.Classification.Halogens)
        self.elements['Kr'] = Element('Kr','Krypton',36,83.80,PeriodicTable.Classification.NobleGases)
        self.elements['Rb'] = Element('Rb','Rubidium',37,85.4678,PeriodicTable.Classification.AlkaliMetals)
        self.elements['Sr'] = Element('Sr','Strontium',38,87.62,PeriodicTable.Classification.AlkaliEarthMetals)
        self.elements['Y'] = Element('Y','Yttrium',39,88.9059,PeriodicTable.Classification.TransitionMetals)
        self.elements['Zr'] = Element('Zr','Zirconium',40,91.22,PeriodicTable.Classification.TransitionMetals)
        self.elements['Nb'] = Element('Nb','Niobium',41,92.91,PeriodicTable.Classification.TransitionMetals)
        self.elements['Mo'] = Element('Mo','Molybdenum',42,95.94,PeriodicTable.Classification.TransitionMetals)
        self.elements['Tc'] = Element('Tc','Technetium',43,99.0,PeriodicTable.Classification.TransitionMetals)
        self.elements['Ru'] = Element('Ru','Ruthenium',44,101.1,PeriodicTable.Classification.TransitionMetals)
        self.elements['Rh'] = Element('Rh','Rhodium',45,102.91,PeriodicTable.Classification.TransitionMetals)
        self.elements['Pd'] = Element('Pd','Palladium',46,106.42,PeriodicTable.Classification.TransitionMetals)
        self.elements['Ag'] = Element('Ag','Silver',47,107.87,PeriodicTable.Classification.TransitionMetals)
        self.elements['Cd'] = Element('Cd','Cadmium',48,112.4,PeriodicTable.Classification.TransitionMetals)
        self.elements['In'] = Element('In','Indium',49,114.82,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['Sn'] = Element('Sn','Tin',50,118.69,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['Sb'] = Element('Sb','Antimony',51,121.75,PeriodicTable.Classification.Metalloids)
        self.elements['Te'] = Element('Te','Tellurium',52,127.6,PeriodicTable.Classification.Metalloids)
        self.elements['I'] = Element('I','Iodine',53,126.9045,PeriodicTable.Classification.Halogens)
        self.elements['Xe'] = Element('Xe','Xenon',54,131.29,PeriodicTable.Classification.NobleGases)
        self.elements['Cs'] = Element('Cs','Cesium',55,132.9054,PeriodicTable.Classification.AlkaliMetals)
        self.elements['Ba'] = Element('Ba','Barium',56,137.33,PeriodicTable.Classification.AlkaliEarthMetals)
        self.elements['La'] = Element('La','Lanthanum',57,138.91,PeriodicTable.Classification.Lanthanides)
        self.elements['Ce'] = Element('Ce','Cerium',58,140.12,PeriodicTable.Classification.Lanthanides)
        self.elements['Pr'] = Element('Pr','Praseodymium',59,140.91,PeriodicTable.Classification.Lanthanides)
        self.elements['Nd'] = Element('Nd','Neodymium',60,144.242,PeriodicTable.Classification.Lanthanides)
        self.elements['Pm'] = Element('Pm','Promethium',61,147.0,PeriodicTable.Classification.Lanthanides)
        self.elements['Sm'] = Element('Sm','Samarium',62,150.35,PeriodicTable.Classification.Lanthanides)
        self.elements['Eu'] = Element('Eu','Europium',63,167.26,PeriodicTable.Classification.Lanthanides)
        self.elements['Gd'] = Element('Gd','Gadolinium',64,157.25,PeriodicTable.Classification.Lanthanides)
        self.elements['Tb'] = Element('Tb','Terbium',65,158.925,PeriodicTable.Classification.Lanthanides)
        self.elements['Dy'] = Element('Dy','Dysprosium',66,162.50,PeriodicTable.Classification.Lanthanides)
        self.elements['Ho'] = Element('Ho','Holmium',67,164.9,PeriodicTable.Classification.Lanthanides)
        self.elements['Er'] = Element('Er','Erbium',68,167.26,PeriodicTable.Classification.Lanthanides)
        self.elements['Tm'] = Element('Tm','Thulium',69,168.93,PeriodicTable.Classification.Lanthanides)
        self.elements['Yb'] = Element('Yb','Ytterbium',70,173.04,PeriodicTable.Classification.Lanthanides)
        self.elements['Lu'] = Element('Lu','Lutetium',71,174.97,PeriodicTable.Classification.Lanthanides)
        self.elements['Hf'] = Element('Hf','Hafnium',72,178.49,PeriodicTable.Classification.TransitionMetals)
        self.elements['Ta'] = Element('Ta','Tantalum',73,180.95,PeriodicTable.Classification.TransitionMetals)
        self.elements['W'] = Element('W','Tungsten',74,183.85,PeriodicTable.Classification.TransitionMetals)
        self.elements['Re'] = Element('Re','Rhenium',75,186.23,PeriodicTable.Classification.TransitionMetals)
        self.elements['Os'] = Element('Os','Osmium',76,190.2,PeriodicTable.Classification.TransitionMetals)
        self.elements['Ir'] = Element('Ir','Iridium',77,192.2,PeriodicTable.Classification.TransitionMetals)
        self.elements['Pt'] = Element('Pt','Platinum',78,195.09,PeriodicTable.Classification.TransitionMetals)
        self.elements['Au'] = Element('Au','Gold',79,196.9655,PeriodicTable.Classification.TransitionMetals)
        self.elements['Hg'] = Element('Hg','Mercury',80,200.59,PeriodicTable.Classification.TransitionMetals)
        self.elements['Tl'] = Element('Tl','Thallium',81,204.383,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['Pb'] = Element('Pb','Lead',82,207.2,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['Bi'] = Element('Bi','Bismuth',83,208.9804,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['Po'] = Element('Po','Polonium',84,210.0,PeriodicTable.Classification.PostTransitionMetals)
        self.elements['At'] = Element('At','Astatine',85,210.0,PeriodicTable.Classification.Halogens)
        self.elements['Rn'] = Element('Rn','Radon',86,222.0,PeriodicTable.Classification.NobleGases)
        self.elements['Fr'] = Element('Fr','Francium',87,233.0,PeriodicTable.Classification.AlkaliMetals)
        self.elements['Ra'] = Element('Ra','Radium',88,226.0254,PeriodicTable.Classification.AlkaliEarthMetals)
        self.elements['Ac'] = Element('Ac','Actinium',89,227.0,PeriodicTable.Classification.Actinides)
        self.elements['Th'] = Element('Th','Thorium',90,232.04,PeriodicTable.Classification.Actinides)
        self.elements['Pa'] = Element('Pa','Protactinium',91,231.0359,PeriodicTable.Classification.Actinides)
        self.elements['U'] = Element('U','Uranium',92,238.03,PeriodicTable.Classification.Actinides)
        self.elements['Np'] = Element('Np','Neptunium',93,237.0,PeriodicTable.Classification.Actinides)
        self.elements['Pu'] = Element('Pu','Plutonium',94,244.0,PeriodicTable.Classification.Actinides)
        self.elements['Am'] = Element('Am','Americium',95,243.0,PeriodicTable.Classification.Actinides)
        self.elements['Cm'] = Element('Cm','Curium',96,247.0,PeriodicTable.Classification.Actinides)
        self.elements['Bk'] = Element('Bk','Berkelium',97,247.0,PeriodicTable.Classification.Actinides)
        self.elements['Cf'] = Element('Cf','Californium',98,251.0,PeriodicTable.Classification.Actinides)
        self.elements['Es'] = Element('Es','Einsteinium',99,254.0,PeriodicTable.Classification.Actinides)
        self.elements['Fm'] = Element('Fm','Fermium',100,257.0,PeriodicTable.Classification.Actinides)
        self.elements['Md'] = Element('Md','Mendelevium',101,258.0,PeriodicTable.Classification.Actinides)
        self.elements['No'] = Element('No','Nobelium',102,259.0,PeriodicTable.Classification.Actinides)
        self.elements['Lr'] = Element('Lr','Lawrencium',103,262.0,PeriodicTable.Classification.Actinides)
        self.elements['Rf'] = Element('Rf','Rutherfordium',104,260.9,PeriodicTable.Classification.TransitionMetals)
        self.elements['Db'] = Element('Db','Dubnium',105,261.9,PeriodicTable.Classification.TransitionMetals)
        self.elements['Sg'] = Element('Sg','Seaborgium',106,262.94,PeriodicTable.Classification.TransitionMetals)
        self.elements['Bh'] = Element('Bh','Bohrium',107,262.0,PeriodicTable.Classification.TransitionMetals)
        self.elements['Hs'] = Element('Hs','Hassium',108,264.8,PeriodicTable.Classification.TransitionMetals)
        self.elements['Mt'] = Element('Mt','Meitnerium',109,265.9,PeriodicTable.Classification.Unknown)
        self.elements['Ds'] = Element('Ds','Darmstadtium',110,261.9,PeriodicTable.Classification.Unknown)
        self.elements['Ds'] = Element('Rg','Roentgenium',111,268.0,PeriodicTable.Classification.Unknown)
        self.elements['Uub'] = Element('Cn','Copernicum',112,268.0,PeriodicTable.Classification.TransitionMetals)
        self.elements['Uuq'] = Element('Uuq','Ununquadium',114,289.0,PeriodicTable.Classification.Unknown)
        self.elements['Uuh'] = Element('Uuh','Ununhexium',116,292,PeriodicTable.Classification.Unknown)

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
            


