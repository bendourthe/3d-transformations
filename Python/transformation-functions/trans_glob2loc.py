# LIBRARIES IMPORT

import numpy as np

# FUNCTION

def trans_glob2loc(Mglob, Rglob2loc, tglob2loc):
    '''
    Transforms a data set from a global to a local coordinate system.
    Input:
        M: nx3 array corresponding to the data set to be transformed (n = number of points, columns = X, Y and Z coordinates)
        Rglob2loc: rotation matrix from global to local coordinate system
        tglob2loc: translation vector from global to local coordinate system
    Output:
        Mloc: nx3 array corresponding to the transformed data set
    '''
    Mloc = np.transpose(np.dot(Rglob2loc, np.transpose(Mglob)) + np.transpose(np.tile(tglob2loc, np.shape(Mglob)[0]).reshape(np.shape(Mglob)[0],3)) )

    return Mloc
