from crystals import Crystal
import sys,os


if __name__=='__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    filepath = os.path.join(__location__,'9014359.cif')
    filepath = os.path.join(__location__,'2010109.cif')
    c = Crystal.from_cif(filepath)
    help(c.atoms.union())

    