# LIBRARIES IMPORT

import numpy as np

# FUNCTION

def glob2loc2glob(O, X, Y):
    '''
    Calculates the rotation and translation matrices allowing the transformations of a data set from a global to a local coordinate system (and vice versa).
    Input:
        O: 3x1 array corresponding to the x, y, z coordinates of the local origin
        X: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the x-axis direction
        Y: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the y-axis direction
    Output:
        Rglob2loc: rotation matrix from global to local coordinate system
        tglob2loc: translation vector from global to local coordinate system
        Rloc2glob: rotation matrix from local to global coordinate system
        tloc2glob: translation vector from local to global coordinate system
    '''
    Xaxis = (X - O)/np.linalg.norm(X - O)
    Yaxis = (Y - O)/np.linalg.norm(Y - O)
    Zaxis = np.cross(Xaxis, Yaxis)
    Yaxis = np.cross(Zaxis, Xaxis)

    Rglob2loc = np.array([Xaxis, Yaxis, Zaxis])
    tglob2loc = np.dot(-Rglob2loc, O)

    Rloc2glob = np.transpose(Rglob2loc)
    tloc2glob = np.array(O)

    return Rglob2loc, tglob2loc, Rloc2glob, tglob2loc
