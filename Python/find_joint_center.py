# LIBRARY IMPORT

import numpy as np

# FUNCTION

def find_joint_center(Os, Xs, Ys, stat_jc, O, X, Y):
    '''
    Find the coordinates of a joint center using static and dynamic motion capture data (e.g. Vicon).
    Input:
        Os: nx3 array corresponding to the x, y, z coordinates of the static local origin (all frames)
        Xs: nx3 array corresponding to the x, y, z coordinates of the landmark defining the x-axis direction (all frames)
        Ys: nx3 array corresponding to the x, y, z coordinates of the landmark defining the y-axis direction (all frames)
        stat_jc: 3x1 array corresponding to the x, y, z coordinates of the static joint center
        O: 3x1 array corresponding to the x, y, z coordinates of the local origin (current frame)
        X: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the x-axis direction (current frame)
        Y: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the y-axis direction (current frame)
    Output:
        dyn_jc: 3x1 array corresponding to the x, y, z coordinates of the dynamic joint center (current frame, global CS)
    Dependencies:
        glob2loc2glob
        trans_glob2loc
        trans_loc2glob
    '''
    # calculate glob2loc using mean static data
    Os = np.mean(Os, axis=0)
    Xs = np.mean(Xs, axis=0)
    Ys = np.mean(Ys, axis=0)
    Rglob2loc, tglob2loc, _, _ = glob2loc2glob(Os, Xs, Ys)
    # apply glob2loc to static joint center to know its position in local CS
    jc_loc = trans_glob2loc(stat_jc, Rglob2loc, tglob2loc)
    # calculate loc2glob using current frame dynamic data
    _, _, Rloc2glob, tloc2glob = glob2loc2glob(O, X, Y)
    # apply loc2glob to static stat_jc expressed in local CS to know its position for the current frame in global CS
    dyn_jc = trans_loc2glob(jc_loc, Rloc2glob, tloc2glob)
    dyn_jc = np.mean(dyn_jc, axis=0)

    return dyn_jc
