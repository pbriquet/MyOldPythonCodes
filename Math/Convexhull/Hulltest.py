import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull



def hull_test(P, X, use_hull=True, verbose=False, hull_tolerance=1e-5, return_hull=True):
    if use_hull:
        L = []
        for x in X:
            L.append(x.args)
        hull = ConvexHull(L)
        X = X[hull.vertices]
    return in_hull()
    n_points = len(X)

    def F(x, X, P):
        return np.linalg.norm( np.dot( x.T, X ) - P )

    bnds = [[0, None]]*n_points # coefficients for each point must be > 0
    cons = ( {'type': 'eq', 'fun': lambda x: np.sum(x)-1} ) # Sum of coefficients must equal 1
    x0 = np.ones((n_points,1))/n_points # starting coefficients

    result = scipy.optimize.minimize(F, x0, args=(X, P), bounds=bnds, constraints=cons)

    if result.fun < hull_tolerance:
        hull_result = True
    else:
        hull_result = False

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