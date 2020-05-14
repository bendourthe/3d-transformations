# LIBRARIES IMPORT

import numpy as np
import sklearn

from sklearn.neighbors import NearestNeighbors          # for ICP algorithm

# FUNCTION

def icp_function(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B.
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD points
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    Dependencies:
        nearest_neighbor
        best_fit_transform
    '''
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbour(src[:m,:].T, dst[:m,:].T)
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        # update the current source
        src = np.dot(T, src)
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T
