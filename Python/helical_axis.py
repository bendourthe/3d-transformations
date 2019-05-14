# LIBRARIES IMPORT

import numpy as np

# FUNCTION

def helical_axis(T):
    '''
    Calculate the components of the vector along which a solid translates and around which it rotates (i.e. helical axis, or screw axis).
    Based of the work of Spoor and Veldpaus (1980): equations [31] and [32].
    Input:
        T: 4x4 transformation matrix defining the object's movement in Euclidean space
    Output:
        hel: vector defining the direction of the helical axis
        p: point where the helical axis intersect the XY-plane (Z = 0)
    '''
    R = np.array(T[0:3,0:3])
    M = np.array([[R[2,1]-R[1,2]], [R[0,2]-R[2,0]], [R[1,0]-T[0,1]]])
    hel = M/np.linalg.norm(M)
    Q = R - np.identity(3)
    Q[:,2] = -hel[:,0]
    p = np.linalg.solve(Q, np.array(np.transpose([T[0:3,3]*-1])))
    p[2,0] = 0

    return hel, p
