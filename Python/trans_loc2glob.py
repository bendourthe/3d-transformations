# LIBRARIES IMPORT

import numpy as np

# FUNCTION

def trans_loc2glob(Mloc, Rloc2glob, tloc2glob):
    '''
    Transforms a data set from a local to a global coordinate system.
    Input:
        Mglob: nx3 array corresponding to the data set to be transformed (n = number of points, columns = X, Y and Z coordinates)
        Rloc2glob: rotation matrix from local to global coordinate system
        tloc2glob: translation vector from local to global coordinate system
    Output:
        Mglob: nx3 array corresponding to the transformed data set
    '''
    Mglob = np.transpose( np.dot(Rloc2glob, np.transpose(Mloc - np.tile(tloc2glob, np.shape(Mloc)[0]).reshape(np.shape(Mloc)[0],3)) ))

    return Mglob
