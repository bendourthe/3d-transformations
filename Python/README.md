___

<a href='http://www.dourthe.tech'> <img src='Dourthe_Technologies_Headers.png' /></a>
___
<center><em>For more information, visit <a href='http://www.dourthe.tech'>www.dourthe.tech</a></em></center>

# 3D transformations toolbox (Python)

## This toolbox includes a few custom python codes enabling essential 3D operations

__
## From global to local
#### Definition
Transforms a data set from a global to a local coordinate system.
#### Input
    M: nx3 array corresponding to the data set to be transformed (n = number of points, columns = X, Y and Z coordinates)
    Rglob2loc: rotation matrix from global to local coordinate system
    tglob2loc: translation vector from global to local coordinate system
#### Output
    Mloc: nx3 array corresponding to the transformed data set
#### Dependencies
    None
#### Example
    Mloc = trans_glob2loc(Mglob, Rglob2loc, tglob2loc)

__
## From local to global
#### Definition
Transforms a data set from a local to a global coordinate system.
#### Input
    Mglob: nx3 array corresponding to the data set to be transformed (n = number of points, columns = X, Y and Z coordinates)
    Rloc2glob: rotation matrix from local to global coordinate system
    tloc2glob: translation vector from local to global coordinate system
#### Output
    Mglob: nx3 array corresponding to the transformed data set
#### Dependencies
    None
#### Example
    Mglob = trans_loc2glob(Mloc, Rloc2glob, tloc2glob)

__
## Transformations from global to local coordinate system (and vice versa)
#### Definition
Calculates the rotation and translation matrices allowing the transformations of a data set from a global to a local coordinate system (and vice versa).
#### Input
    O: 3x1 array corresponding to the x, y, z coordinates of the local origin
    X: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the x-axis direction
    Y: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the y-axis direction
#### Output
    Rglob2loc: rotation matrix from global to local coordinate system
    tglob2loc: translation vector from global to local coordinate system
    Rloc2glob: rotation matrix from local to global coordinate system
    tloc2glob: translation vector from local to global coordinate system
#### Dependencies
    None
#### Example
    Rglob2loc, tglob2loc, Rloc2glob, tglob2loc = glob2loc2glob(O, X, Y)

__
## Find joint centers (motion capture)
#### Definition
Find the coordinates of a joint center using static and dynamic motion capture data (e.g. Vicon).
#### Input
    Os: nx3 array corresponding to the x, y, z coordinates of the static local origin (all frames)
    Xs: nx3 array corresponding to the x, y, z coordinates of the landmark defining the x-axis direction (all frames)
    Ys: nx3 array corresponding to the x, y, z coordinates of the landmark defining the y-axis direction (all frames)
    stat_jc: 3x1 array corresponding to the x, y, z coordinates of the static joint center
    O: 3x1 array corresponding to the x, y, z coordinates of the local origin (current frame)
    X: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the x-axis direction (current frame)
    Y: 3x1 array corresponding to the x, y, z coordinates of the landmark defining the y-axis direction (current frame)
#### Output
    dyn_jc: 3x1 array corresponding to the x, y, z coordinates of the dynamic joint center (current frame, global CS)
#### Dependencies
    glob2loc2glob
    trans_glob2loc
    trans_loc2glob
#### Example
    dyn_jc = find_joint_center(Os, Xs, Ys, stat_jc, O, X, Y)

__
## Nearest (Euclidean) neighbor
#### Definition:
Find the nearest (Euclidean) neighbor between two data sets of points.
#### Input
    src: Nxm array of points
    dst: Nxm array of points
#### Output
    distances: Euclidean distances of the nearest neighbor
    indices: dst indices of the nearest neighbor
#### Dependencies
    None
#### Example
    -> A and B are two clouds of points in 3D space
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    distances, indices = nearest_neighbour(src[:m,:].T, dst[:m,:].T)

__
## Best fit transform
#### Definition
Calculates the least-squares best-fit transform from points in A to points in B in m spatial dimensions.
#### Input
    A: Nxm numpy array of corresponding points
    B: Nxm numpy array of corresponding points
#### Output
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
#### Dependencies
    None
#### Example
    -> follow-up on previous example (i.e. nearest_neighbor)
    # compute the transformation between the current source and nearest destination points
    T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

__
## Iterative Closest Point (ICP)
#### Definition
Use an ICP method to find the best-fit transform that maps points A on to points B.
#### Input
    A: Nxm numpy array of source mD points
    B: Nxm numpy array of destination mD points
    init_pose: (m+1)x(m+1) homogeneous transformation
    max_iterations: exit algorithm after max_iterations
    tolerance: convergence criteria
#### Output
    T: final homogeneous transformation that maps A on to B
    distances: Euclidean distances (errors) of the nearest neighbor
    i: number of iterations to converge
#### Dependencies
    nearest_neighbor
    best_fit_transform
#### Example
    T, distances, i = icp_function(A, B, init_pose=None, max_iterations=20, tolerance=0.001)

__
## Helical axis
#### Definition
Calculate the components of the vector along which a solid translates and around which it rotates (i.e. helical axis, or screw axis).
Based of the work of Spoor and Veldpaus (1980): equations [31] and [32]
#### Input
    T: 4x4 transformation matrix defining the object's movement in Euclidean space
#### Output
    hel: vector defining the direction of the helical axis
    p: point where the helical axis intersect the XY-plane (Z = 0)
#### Dependencies
    None
#### Example
    hel, p = helical_axis(T)
