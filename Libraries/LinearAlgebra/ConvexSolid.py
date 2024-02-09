from math import *
from Vector import *
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def hull_test(P, X, use_hull=True, verbose=False, hull_tolerance=1e-5, return_hull=False):
    if use_hull:
        L = []
        for x in X:
            L.append(np.array([x.values[0],x.values[1],x.values[2]]))
        hull = ConvexHull(L)
        return in_hull(P,hull)

    n_points = len(X)

    def F(x, X, P):
        return np.linalg.norm( np.dot( x.T, X ) - P )

    bnds = [(0.0, 1.0)]*n_points # coefficients for each point must be > 0
    cons = ( {'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)} ) # Sum of coefficients must equal 1
    x0 = np.ones((n_points,1))/n_points # starting coefficients

    result = scipy.optimize.minimize(F, x0, args=(X, P), bounds=bnds, constraints=cons)

    if result.fun < hull_tolerance:
        hull_result = True
    else:
        hull_result = False
    print(result)
    if verbose:
        print( '# boundary points:', n_points)
        print( 'x.T * X - P:', F(result.x,X,P) )
        if hull_result: 
            print( 'Point P is in the hull space of X')
        else: 
            print( 'Point P is NOT in the hull space of X')

    if return_hull:
        return hull_result, X
    else:
        return hull_result

class ConvexSolid:
    def __init__(self,L):
        self.L = L

    @staticmethod
    def beta(L1,v2):
        if(L1*v2<=0.0):
            return -1.0e-12
        else:
            if(L1*v2/L1.norm()**2.0 > 1.0):
                return -1.0e-12
            else:
                return L1*v2/L1.norm()**2.0

    def checkPointBelongs(self,v):
        return hull_test(v,self.L)