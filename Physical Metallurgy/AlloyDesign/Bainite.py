import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns","ChemicalLibrary"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored

from math import *
from PeriodicTable import *

class Bainite:
    class Bodnar:
        @staticmethod
        def Bs(alloy,TScale=TempScale.Fahrenheit):
            tmp = 844.0 - 597.0*alloy['C'] - 63.0*alloy['Mn'] - 16.0*alloy['Ni'] - 78.0*alloy['Cr']

    class Kirkaldy:
        @staticmethod
        def Bs(alloy,TScale=TempScale.Fahrenheit):
            tmp = 844.0 - 597.0*alloy['C'] - 63.0*alloy['Mn'] - 16.0*alloy['Ni'] - 78.0*alloy['Cr']

    class Lee:
        @staticmethod
        def Bs(alloy,TScale=TempScale.Fahrenheit):
            tmp = 844.0 - 597.0*alloy['C'] - 63.0*alloy['Mn'] - 16.0*alloy['Ni'] - 78.0*alloy['Cr']
    
    class Suehiro:
        @staticmethod
        def Bs(alloy,TScale=TempScale.Fahrenheit):
            tmp = 844.0 - 597.0*alloy['C'] - 63.0*alloy['Mn'] - 16.0*alloy['Ni'] - 78.0*alloy['Cr']

    class StevensHaynes:
        @staticmethod
        def Bs(alloy,TScale=TempScale.Fahrenheit):
            tmp = 844.0 - 597.0*alloy['C'] - 63.0*alloy['Mn'] - 16.0*alloy['Ni'] - 78.0*alloy['Cr']

    class vanBohemen:
        @staticmethod
        def Bs(alloy,TScale=TempScale.Fahrenheit):
            tmp = 844.0 - 597.0*alloy['C'] - 63.0*alloy['Mn'] - 16.0*alloy['Ni'] - 78.0*alloy['Cr']

    class Zhao:
        @staticmethod
        def Bs(alloy,TScale=TempScale.Fahrenheit):
            tmp = 844.0 - 597.0*alloy['C'] - 63.0*alloy['Mn'] - 16.0*alloy['Ni'] - 78.0*alloy['Cr']
