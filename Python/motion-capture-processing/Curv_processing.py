# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

from scipy.signal import find_peaks
from scipy import interpolate               # for interpolation
from mpl_toolkits.mplot3d import Axes3D     # for 3D visualization

# DEPENDENCIES (FUNCTIONS)

def nan_find(y):
    '''
    Generates a NaNs logical array where the indices of each NaN observation is True.
    Generates a local function that can extract the indices of each NaN observation as a list.
    Input:
        - y: nx1 array that contains NaNs
    Output:
        - nan_logic: logical array where the indices of each NaN observation is True
        - find_true: function that returns the indices of all True observations in an array.
    Example:
        nan_logic, find_true = nan_find(y)
        find_true(nan_logic) -> returns array with indices of all NaN in y
        find_true(~nan_logic) -> returns array with indices of all non-NaN in y
    '''
    nan_logic = np.isnan(y)
    # lambda k: k.nonzero()[0] defines the function find_true, and k represents the corresponding input
    find_true = lambda k: k.nonzero()[0]

    return nan_logic, find_true

def cubic_spline_fill(signal, mode=None):
    '''
    Interpolates missing observations within a time series using a cubic spline.
    Notes:
        Only interpolates time series with NaN (otherwise return original time series)
        Does not interpolate empty time series (returns same empty time series)
    Input:
        signal: nx1 array: tested time series
        mode: select mode to deal with edge effect
            0: set edges to zero
            1: set to edge value
            2: set edges to mean
            3: set edges to NaN [default]
            4: set edges to reflected signal
    Output:
        signal_interp: new nx1 array with interpolated values

    Dependencies:
        nan_find
    '''

    # Deal with default values and potential missing input variables
    if mode == None:
        mode = 3

    # If no missing observation -> return original signal
    if np.shape(np.where(np.isnan(signal) == True))[1] == 0:

        return signal

    # If empty signal -> return original signal
    elif np.shape(np.where(np.isnan(signal) == True))[1] == np.shape(signal)[0]:

        return signal

    else:
        # Define frame vector
        fr = np.arange(0, len(signal))

        # Generate a NaNs logical array where the indices of each NaN observation is True
        nan_logic, find_true = nan_find(signal)

        # Find indices of non-missing observations
        obs = find_true(~nan_logic)

        # Isolate non-missing portion of the signal
        a = fr[obs]
        b = signal[obs]

        # Find equation of the cubic spline that best fits the corresponding signal
        cs = interpolate.CubicSpline(a, b)

        # Initialization
        signal_interp = np.array(np.empty(np.shape(signal)[0]))

        # Apply cubic spline equation to interpolate between edges
        signal_interp[obs[0]:obs[-1]] = cs(fr[obs[0]:obs[-1]])

        # Change edges values to neighboring values to prevent edge effects
        signal_interp[obs[0]] = signal_interp[obs[0]+1]
        signal_interp[obs[-1]] = signal_interp[obs[-1]-1]

        # Deal with edge effects
        if mode == 0:
            signal_interp[0:obs[0]] = 0
            signal_interp[obs[-1]:] = 0
        elif mode == 1:
            signal_interp[0:obs[0]] = signal_interp[obs[0]]
            signal_interp[obs[-1]:] = signal_interp[obs[-1]]
        elif mode == 2:
            signal_interp[0:obs[0]] = np.nanmean(signal)
            signal_interp[obs[-1]:] = np.nanmean(signal)
        elif mode == 3:
            signal_interp[0:obs[0]] = np.nan
            signal_interp[obs[-1]:] = np.nan
        elif mode == 4:
            pre = len(np.arange(fr[0], obs[0]))
            post = len(np.arange(obs[-1], fr[-1]))
            if pre > 0:
                if obs[-1]-obs[0] < pre:
                    signal_interp[obs[0]-(obs[-1]-obs[0]):obs[0]] = np.flip(signal_interp[obs[0]:obs[-1]])
                    signal_interp[0:obs[0]-(obs[-1]-obs[0])] = signal_interp[obs[0]-(obs[-1]-obs[0])+1]
                else:
                    signal_interp[0:obs[0]] = np.flip(signal_interp[obs[0]:obs[0]+pre])
            if post > 0:
                if obs[-1]-post-1 < 0:
                    signal_interp[obs[-1]:2*obs[-1]-obs[0]] = np.flip(signal_interp[obs[0]:obs[-1]])
                    signal_interp[2*obs[-1]-obs[0]:] = signal_interp[2*obs[-1]-obs[0]-1]
                else:
                    signal_interp[obs[-1]:] = np.flip(signal_interp[obs[-1]-post-1:obs[-1]])

        return signal_interp

def cubic_spline_fill_3D(data, mode=None):
    '''
    Applies cubic_spline_fill to each dimension of a 3D time series.
    Input:
        data: nx3 array: tested data set containing 3D time series
        mode: select mode to deal with edge effect
            0: set edges to zero
            1: set to edge value
            2: set edges to mean
            3: set edges to NaN [default]
            4: set edges to reflected signal
    Output:
        data_interp: new nx3 array with interpolated values
    Dependencies:
        nan_find
        cubic_spline_fill
    '''

    # Deal with default values and potential missing input variables
    if mode == None:
        mode = 3

    # Apply cubic_spline_fill to each dimension of the 3D data set
    a = cubic_spline_fill(data[:,0], mode=mode)
    b = cubic_spline_fill(data[:,1], mode=mode)
    c = cubic_spline_fill(data[:,2], mode=mode)
    data_interp = np.transpose(np.array([a, b, c]))

    return data_interp

# SETTINGS

#   Main Directory and Data type
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/ACL Injury Screening/Data"
data_type =  'Curv'

#   List of participants
#       Note: full list for this pipeline -> ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']

#   Trials
#       Note: full list -> ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']
trials = ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']

#   Key points
#       Note: full list -> ['ANKLE_LEFT', 'ANKLE_RIGHT', 'EAR_LEFT', 'EAR_RIGHT', 'ELBOW_LEFT', 'ELBOW_RIGHT', 'EYE_LEFT', 'EYE_RIGHT','HEEL_LEFT', 'HEEL_RIGHT', 'HIP_LEFT', 'HIP_RIGHT', 'KNEE_LEFT', 'KNEE_RIGHT', 'NOSE', 'PINKY_TOE_LEFT', 'PINKY_TOE_RIGHT', 'SHOULDER_LEFT', 'SHOULDER_RIGHT', 'TOE_LEFT', 'TOE_RIGHT', 'WRIST_LEFT', 'WRIST_RIGHT']
key_points = ['ANKLE_LEFT', 'ANKLE_RIGHT', 'EAR_LEFT', 'EAR_RIGHT', 'ELBOW_LEFT', 'ELBOW_RIGHT', 'EYE_LEFT', 'EYE_RIGHT','HEEL_LEFT', 'HEEL_RIGHT', 'HIP_LEFT', 'HIP_RIGHT', 'KNEE_LEFT', 'KNEE_RIGHT', 'NOSE', 'PINKY_TOE_LEFT', 'PINKY_TOE_RIGHT', 'SHOULDER_LEFT', 'SHOULDER_RIGHT', 'TOE_LEFT', 'TOE_RIGHT', 'WRIST_LEFT', 'WRIST_RIGHT']

#   Data columns
#       Additional columns
#           Note: full list -> ['closeToEdge', 'humanDetected', 'index', 'time']
add_columns = ['closeToEdge', 'humanDetected', 'index', 'time']
#       Dictionary columns
#           Note: full list -> ['x', 'y', 'logit', 'confidence', 'occluded', 'closeToEdge']
#           Note: will only work with ['x', 'y']
dict_columns = ['x', 'y']
#       Output columns
#        Note: to remain consistent with the Vicon and Kinect data sets, some transformations are applied:
#               CURV_x -> Vicon/Kinect_y    |   CURV_y -> Vicon/Kinect_z
out_columns = ['y', 'z']

#   Data processing
joint_center_trajectories = 1       # set to 1 of you want to export the processed results into CSV format
biomechanical_variables = 1         # set to 1 of you want to export the processed results into CSV format
sf = 30                 # sampling frequency or the original color video data (in Hz)
event_detection = 1     # use manual event detection? (1:Yes - 0:No)
interpolation = 1       # interpolate missing observations? (1=Yes - 0=No)
interpolation_mode = 1  # select mode to deal with edge effects during cubic spline interpolation
                            # 0: set edges to zero
                            # 1: set to edge value
                            # 2: set edges to mean
                            # 3: set edges to NaN [default]
                            # 4: set edges to reflected signal

#    Plotting
ft_size = 12            # define font size (for axes titles, etc.)
elevation = 10.         # elevation for 3D visualization
azimuth = 360           # azimuth for 3D visualization
plot = 1                # generate animation? (1=Yes - 0=No)
save_ani = 0            # save animated trial? (1: yes, 0: no)

#    Set up formatting for the movie files
Writer = mpl.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#   Event detection parameters and Initialization
height_threshold_factor = 80    # height threshold factor for the detection of peaks (in %)
                                # e.g. a value of X means that only peaks that have a y-value of more than X% of the max height of the signal will be found
event_window = 2                # number of frames that will be included to detect box jump, landings and take-off during DVJ
DVJ_events = np.zeros(5)        # array initialization
Dist_events = np.zeros([2])     # array initialization
Timed_events = np.array([])     # array initialization

#    Extensions
ext_data = '.json'      # extension for imported data files
ext_exp = '.csv'        # extension for exported data files
ext_vid = '.mp4'        # extension for exported video file

# DATA PROCESSING

for participant in participants:
    print('')
    print('Participant in progress -> ' + participant + '...')

    for trial in trials:
        print('...')
        print('Trial in progress -> ' + trial)

# EXCPETIONS

        #   Skip trials that are missing or that are corrupted
        if participant == '09' and (trial == 'DVJ_0' or trial == 'DVJ_1' or trial == 'DVJ_2' or trial == 'RDist_1' or trial == 'RDist_2') or participant == '10' and trial == 'LTimed_0' or participant == '17' and trial == 'LDist_0':

            print('-> Warning! Skipped due to missing/corrupted data')

        else:

# PATHS DEFINITION

            #   Import
            #       Path for participant folder
            path = os.path.join(DATADIR, 'Raw\\' + data_type + '\\' + participant)
            #       Path for trial file
            file_path = os.path.join(path, data_type + '-' + participant + '_' + trial + ext_data)

            #   Export
            #       Path for joint centers data
            export_jc_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_jc' + ext_exp)
            #       Path for biomechanical variables
            export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_bvars' + ext_exp)
            #       Path for video export
            export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + ext_vid)

# DATA IMPORT

            data = pd.read_json(file_path)

            print('-> Data imported')

# DATA STRUCTURE & ALIGNMENT

            #   Initialize each column and the restructured array
            for key_point in key_points:
                for i in range(0, len(out_columns)):
                    exec(key_point + '_' + out_columns[i] + ' = np.zeros(len(data))')
            new_data = np.zeros(len(data))

            #   Generate each column of the data frame
            for key_point in key_points:
                for i in range(0, len(out_columns)):
                    for j in range(0, len(data)):
                        exec(key_point + '_' + out_columns[i] + "[j] = data['" + key_point + "'][j]['" + dict_columns[i] + "']")
                #   Combine each column into the restructured data set
                exec('new_data = np.vstack([new_data, np.vstack([np.zeros(len(data)), np.vstack([' + key_point + '_y, ' + key_point + '_z])])])')
            #   Transpose resulting array and delete the first column of zeros
            new_data = np.transpose(new_data[1:,:])

            #   Initialize flattened coordinates
            y = np.zeros(len(data))
            z = np.zeros(len(data))

            #   Generate arrays for Y and Z coordinates
            for key_point in key_points:
                exec('y = np.vstack([y, ' + key_point + '_y])')
                exec('z = np.vstack([z, ' + key_point + '_z])')

            #   Remove first row of zeros and transpose
            y = np.transpose(y[1:,:])
            z = np.transpose(z[1:,:])

            #   Flatten resulting arrays
            y = y.flatten()
            z = z.flatten()

            #   Generate vector of zeros to account for missing X coordinates
            x = np.zeros(len(y))

            #   Normalize falttened data around 0 for x and y, and flip and translates z to have only positive values
            #       Note: (-) sign in front of a and b to change the direction and match with Vicon
            x = -(x - np.mean(x))
            y = -(y - np.mean(y))
            z = -(z - np.max(z))

            #   Align each column of the data frame
            new_data = np.zeros(len(data))
            for key_point in key_points:
                for i in range(0, len(out_columns)):
                    if out_columns[i] == 'y':
                        exec(key_point + '_' + out_columns[i] + ' = -(' + key_point + '_' + out_columns[i] + ' - np.mean(y))')
                    if out_columns[i] == 'z':
                        exec(key_point + '_' + out_columns[i] + ' = -(' + key_point + '_' + out_columns[i] + ' - np.max(z))')
                #   Re-combine each column into the restructured data set
                exec('new_data = np.vstack([new_data, np.vstack([np.zeros(len(data)), np.vstack([' + key_point + '_y, ' + key_point + '_z])])])')
            #   Transpose resulting array and delete the first column of zeros
            new_data = np.transpose(new_data[1:,:])

            #   Generate restructured data frame
            f_joints = pd.MultiIndex.from_product([key_points, ['X', 'Y', 'Z']])
            df_joints = pd.DataFrame(new_data, columns=f_joints)

            print('-> Data restructured and aligned')

# LANDMARKS SELECTION & LABELING

            #   Select landmarks
            RHJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([HIP_RIGHT_y, HIP_RIGHT_z])]))
            RKJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([KNEE_RIGHT_y, KNEE_RIGHT_z])]))
            RAJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([ANKLE_RIGHT_y, ANKLE_RIGHT_z])]))
            RFT = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([TOE_RIGHT_y, TOE_RIGHT_z])]))
            RSJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([SHOULDER_RIGHT_y, SHOULDER_RIGHT_z])]))
            LHJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([HIP_LEFT_y, HIP_LEFT_z])]))
            LKJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([KNEE_LEFT_y, KNEE_LEFT_z])]))
            LAJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([ANKLE_LEFT_y, ANKLE_LEFT_z])]))
            LFT = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([TOE_LEFT_y, TOE_LEFT_z])]))
            LSJ = np.transpose(np.vstack([np.zeros(len(data)), np.vstack([SHOULDER_LEFT_y, SHOULDER_LEFT_z])]))

            #   Define artificial landmarks (to best match with Kinect data)
            SB = RHJ + LHJ / 2
            SHM = RSJ + LSJ / 2

            #   Isolate z-axis trajectories for event detection
            RAJz = RAJ[:,2]
            LAJz = LAJ[:,2]

            print('-> Landmarks selection and labeling completed')

# JOINTS VELOCITIES & ACCELERATIONS

            #   Velocity: calculated as velocity(frame_i) = [position(frame_i) - position(frame_i-1)] * sf
            RHJ_vel = np.array(np.diff(RHJ, axis=0)*sf)
            RKJ_vel = np.array(np.diff(RKJ, axis=0)*sf)
            RAJ_vel = np.array(np.diff(RAJ, axis=0)*sf)
            RSJ_vel = np.array(np.diff(RSJ, axis=0)*sf)
            LHJ_vel = np.array(np.diff(LHJ, axis=0)*sf)
            LKJ_vel = np.array(np.diff(LKJ, axis=0)*sf)
            LAJ_vel = np.array(np.diff(LHJ, axis=0)*sf)
            LSJ_vel = np.array(np.diff(LSJ, axis=0)*sf)

            #   Acceleration: calculated as acceleration(frame_i) = [velocity(frame_i) - velocity(frame_i-1)] * sf
            RHJ_acc = np.array(np.diff(RHJ_vel, axis=0)*sf)
            RKJ_acc = np.array(np.diff(RKJ_vel, axis=0)*sf)
            RAJ_acc = np.array(np.diff(RAJ_vel, axis=0)*sf)
            RSJ_acc = np.array(np.diff(RSJ_vel, axis=0)*sf)
            LHJ_acc = np.array(np.diff(LHJ_vel, axis=0)*sf)
            LKJ_acc = np.array(np.diff(LKJ_vel, axis=0)*sf)
            LAJ_acc = np.array(np.diff(LHJ_vel, axis=0)*sf)
            LSJ_acc = np.array(np.diff(LSJ_vel, axis=0)*sf)

            empty_row = np.zeros(3)
            empty_row[:] = np.nan

            #   Note: first frame cannot have xm velocity (as 2 observations are needed), so to maintain the same shape
            #   compared to position data, one empty row is added
            RHJ_vel = np.vstack([empty_row, RHJ_vel])
            RKJ_vel = np.vstack([empty_row, RKJ_vel])
            RAJ_vel = np.vstack([empty_row, RAJ_vel])
            RSJ_vel = np.vstack([empty_row, RSJ_vel])
            LHJ_vel = np.vstack([empty_row, LHJ_vel])
            LKJ_vel = np.vstack([empty_row, LKJ_vel])
            LAJ_vel = np.vstack([empty_row, LAJ_vel])
            LSJ_vel = np.vstack([empty_row, LSJ_vel])

            #   Note: first two frames cannot have accelerations (as 3 observations are needed), so to maintain the same shape
            #   compared to position data, two rows empty rowws are added
            RHJ_acc = np.vstack([empty_row, empty_row, RHJ_acc])
            RKJ_acc = np.vstack([empty_row, empty_row, RKJ_acc])
            RAJ_acc = np.vstack([empty_row, empty_row, RAJ_acc])
            RSJ_acc = np.vstack([empty_row, empty_row, RSJ_acc])
            LHJ_acc = np.vstack([empty_row, empty_row, LHJ_acc])
            LKJ_acc = np.vstack([empty_row, empty_row, LKJ_acc])
            LAJ_acc = np.vstack([empty_row, empty_row, LAJ_acc])
            LSJ_acc = np.vstack([empty_row, empty_row, LSJ_acc])

            print('-> Joints velocities and accelerations calculated')

# BIOMECHANICAL VARIABLES CALCULATION & EVENT DETECTION

        #   Vectors definition for custom angles calculation
            #       From spine middle to spine base
            SHM2SB = np.array(SB - SHM)
            #       From right to left shoulder
            R2LSH = np.array(LSJ - RSJ)
            #       From hip to knee
            RH2K = np.array(RKJ - RHJ)
            LH2K = np.array(LKJ - LHJ)
            #       From knee to ankle
            RK2A = np.array(RAJ - RKJ)
            LK2A = np.array(LAJ - LKJ)
            #       2D projections
            #           Frontal/Coronal
            SHM2SB_F = np.array(SHM2SB[:,1:3])
            RH2K_F = np.array(RH2K[:,1:3])
            LH2K_F = np.array(LH2K[:,1:3])
            RK2A_F = np.array(RK2A[:,1:3])
            LK2A_F = np.array(LK2A[:,1:3])
            #           Sagittal
            SHM2SB_S = np.array(np.transpose([SHM2SB[:,0], SHM2SB[:,-1]]))
            RH2K_S = np.array(np.transpose([RH2K[:,0], RH2K[:,-1]]))
            LH2K_S = np.array(np.transpose([LH2K[:,0], LH2K[:,-1]]))
            RK2A_S = np.array(np.transpose([RK2A[:,0], RK2A[:,-1]]))
            LK2A_S = np.array(np.transpose([LK2A[:,0], LK2A[:,-1]]))
            #           Transverse
            SHM2SB_T = np.array(SHM2SB[:,0:2])
            R2LSH_T = np.array(R2LSH[:,0:2])
            RH2K_T = np.array(RH2K[:,0:2])
            LH2K_T = np.array(LH2K[:,0:2])
            RK2A_T = np.array(RK2A[:,0:2])
            LK2A_T = np.array(LK2A[:,0:2])

        #   Define frame and time vectors
            fr = np.array(range(0,int(len(data))))
            t = fr/sf

        #   Trunk angles
            #       3D angle
            ground = np.transpose(np.array([np.ones(len(SHM2SB)), np.zeros(len(SHM2SB)), np.zeros(len(SHM2SB))]))
            TR_3D = np.arccos(np.dot(SHM2SB[0,:], ground[0,:]) / (np.linalg.norm(SHM2SB[0,:])*np.linalg.norm(ground[0,:]))) * 180 / math.pi
            for i in range(1,len(SHM2SB)):
                X = np.arccos(np.dot(SHM2SB[i,:], ground[i,:]) / (np.linalg.norm(SHM2SB[i,:])*np.linalg.norm(ground[i,:]))) * 180 / math.pi
                TR_3D = np.vstack([TR_3D, X])
            TR_3D = TR_3D[:,0]
            #       Frontal/Coronal plane (abduction-adduction)
            ground = np.transpose(np.array([np.ones(len(SHM2SB)), np.zeros(len(SHM2SB))]))
            TR_AB = np.arccos(np.dot(SHM2SB_F[0,:], ground[0,:]) / (np.linalg.norm(SHM2SB_F[0,:])*np.linalg.norm(ground[0,:]))) * 180 / math.pi
            for i in range(1,len(SHM2SB)):
                X = np.arccos(np.dot(SHM2SB_F[i,:], ground[i,:]) / (np.linalg.norm(SHM2SB_F[i,:])*np.linalg.norm(ground[i,:]))) * 180 / math.pi
                TR_AB = np.vstack([TR_AB, X])
            TR_AB = 90-TR_AB[:,0]
            #       Sagittal plane (flexion-extension)
            TR_FL = np.arccos(np.dot(SHM2SB_S[0,:], ground[0,:]) / (np.linalg.norm(SHM2SB_S[0,:])*np.linalg.norm(ground[0,:]))) * 180 / math.pi
            for i in range(1,len(SHM2SB)):
                X = np.arccos(np.dot(SHM2SB_S[i,:], ground[i,:]) / (np.linalg.norm(SHM2SB_S[i,:])*np.linalg.norm(ground[i,:]))) * 180 / math.pi
                TR_FL = np.vstack([TR_FL, X])
            TR_FL = 90-TR_FL[:,0]
            #       Transverse plane (internal-external)
            TR_INT = np.arccos(np.dot(R2LSH_T[0,:], ground[0,:]) / (np.linalg.norm(R2LSH_T[0,:])*np.linalg.norm(ground[0,:]))) * 180 / math.pi
            for i in range(1,len(SHM2SB)):
                X = np.arccos(np.dot(R2LSH_T[i,:], ground[i,:]) / (np.linalg.norm(R2LSH_T[i,:])*np.linalg.norm(ground[i,:]))) * 180 / math.pi
                TR_INT = np.vstack([TR_INT, X])
            TR_INT = 90-TR_INT[:,0]

        #   Hip angles
            #       3D angles
            RH_3D = np.arccos(np.dot(SHM2SB[0,:], RH2K[0,:]) / (np.linalg.norm(SHM2SB[0,:])*np.linalg.norm(RH2K[0,:]))) * 180 / math.pi
            LH_3D = np.arccos(np.dot(SHM2SB[0,:], LH2K[0,:]) / (np.linalg.norm(SHM2SB[0,:])*np.linalg.norm(LH2K[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(SHM2SB[i,:], RH2K[i,:]) / (np.linalg.norm(SHM2SB[i,:])*np.linalg.norm(RH2K[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(SHM2SB[i,:], LH2K[i,:]) / (np.linalg.norm(SHM2SB[i,:])*np.linalg.norm(LH2K[i,:]))) * 180 / math.pi
                RH_3D = np.vstack([RH_3D, X])
                LH_3D = np.vstack([LH_3D, Y])
            RH_3D = RH_3D[:,0]
            LH_3D = LH_3D[:,0]
            #       Frontal/Coronal plane (abduction-adduction)
            RH_AB = np.arccos(np.dot(SHM2SB_F[0,:], RH2K_F[0,:]) / (np.linalg.norm(SHM2SB_F[0,:])*np.linalg.norm(RH2K_F[0,:]))) * 180 / math.pi
            LH_AB = np.arccos(np.dot(SHM2SB_F[0,:], LH2K_F[0,:]) / (np.linalg.norm(SHM2SB_F[0,:])*np.linalg.norm(LH2K_F[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(SHM2SB_F[i,:], RH2K_F[i,:]) / (np.linalg.norm(SHM2SB_F[i,:])*np.linalg.norm(RH2K_F[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(SHM2SB_F[i,:], LH2K_F[i,:]) / (np.linalg.norm(SHM2SB_F[i,:])*np.linalg.norm(LH2K_F[i,:]))) * 180 / math.pi
                RH_AB = np.vstack([RH_AB, X])
                LH_AB = np.vstack([LH_AB, Y])
            RH_AB = -RH_AB[:,0]
            LH_AB = -LH_AB[:,0]
            #       Sagittal plane (flexion-extension)
            RH_FL = np.arccos(np.dot(SHM2SB_S[0,:], RH2K_S[0,:]) / (np.linalg.norm(SHM2SB_S[0,:])*np.linalg.norm(RH2K_S[0,:]))) * 180 / math.pi
            LH_FL = np.arccos(np.dot(SHM2SB_S[0,:], LH2K_S[0,:]) / (np.linalg.norm(SHM2SB_S[0,:])*np.linalg.norm(LH2K_S[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(SHM2SB_S[i,:], RH2K_S[i,:]) / (np.linalg.norm(SHM2SB_S[i,:])*np.linalg.norm(RH2K_S[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(SHM2SB_S[i,:], LH2K_S[i,:]) / (np.linalg.norm(SHM2SB_S[i,:])*np.linalg.norm(LH2K_S[i,:]))) * 180 / math.pi
                RH_FL = np.vstack([RH_FL, X])
                LH_FL = np.vstack([LH_FL, Y])
            RH_FL = -RH_FL[:,0]
            LH_FL = -LH_FL[:,0]
            #       Transverse plane (internal-external)
            RH_INT = np.arccos(np.dot(SHM2SB_T[0,:], RH2K_T[0,:]) / (np.linalg.norm(SHM2SB_T[0,:])*np.linalg.norm(RH2K_T[0,:]))) * 180 / math.pi
            LH_INT = np.arccos(np.dot(SHM2SB_T[0,:], LH2K_T[0,:]) / (np.linalg.norm(SHM2SB_T[0,:])*np.linalg.norm(LH2K_T[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(SHM2SB_T[i,:], RH2K_T[i,:]) / (np.linalg.norm(SHM2SB_T[i,:])*np.linalg.norm(RH2K_T[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(SHM2SB_T[i,:], LH2K_T[i,:]) / (np.linalg.norm(SHM2SB_T[i,:])*np.linalg.norm(LH2K_T[i,:]))) * 180 / math.pi
                RH_INT = np.vstack([RH_INT, X])
                LH_INT = np.vstack([LH_INT, Y])
            RH_INT = 180-RH_INT[:,0]
            LH_INT = 180-LH_INT[:,0]

        #   Knee angles
            #       3D angle
            RK_3D = np.arccos(np.dot(RH2K[0,:], RK2A[0,:]) / (np.linalg.norm(RH2K[0,:])*np.linalg.norm(RK2A[0,:]))) * 180 / math.pi
            LK_3D = np.arccos(np.dot(LH2K[0,:], LK2A[0,:]) / (np.linalg.norm(LH2K[0,:])*np.linalg.norm(LK2A[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(RH2K[i,:], RK2A[i,:]) / (np.linalg.norm(RH2K[i,:])*np.linalg.norm(RK2A[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(LH2K[i,:], LK2A[i,:]) / (np.linalg.norm(LH2K[i,:])*np.linalg.norm(LK2A[i,:]))) * 180 / math.pi
                RK_3D = np.vstack([RK_3D, X])
                LK_3D = np.vstack([LK_3D, Y])
            RK_3D = RK_3D[:,0]
            LK_3D = LK_3D[:,0]
            #       Frontal/Coronal plane (abduction-adduction)
            RK_AB = np.arccos(np.dot(RH2K_F[0,:], RK2A_F[0,:]) / (np.linalg.norm(RH2K_F[0,:])*np.linalg.norm(RK2A_F[0,:]))) * 180 / math.pi
            LK_AB = np.arccos(np.dot(LH2K_F[0,:], LK2A_F[0,:]) / (np.linalg.norm(LH2K_F[0,:])*np.linalg.norm(LK2A_F[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(RH2K_F[i,:], RK2A_F[i,:]) / (np.linalg.norm(RH2K_F[i,:])*np.linalg.norm(RK2A_F[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(LH2K_F[i,:], LK2A_F[i,:]) / (np.linalg.norm(LH2K_F[i,:])*np.linalg.norm(LK2A_F[i,:]))) * 180 / math.pi
                RK_AB = np.vstack([RK_AB, X])
                LK_AB = np.vstack([LK_AB, Y])
            RK_AB = RK_AB[:,0]
            LK_AB = -LK_AB[:,0]
            #       Sagittal plane (flexion-extension)
            RK_FL = np.arccos(np.dot(RH2K_S[0,:], RK2A_S[0,:]) / (np.linalg.norm(RH2K_S[0,:])*np.linalg.norm(RK2A_S[0,:]))) * 180 / math.pi
            LK_FL = np.arccos(np.dot(LH2K_S[0,:], LK2A_S[0,:]) / (np.linalg.norm(LH2K_S[0,:])*np.linalg.norm(LK2A_S[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(RH2K_S[i,:], RK2A_S[i,:]) / (np.linalg.norm(RH2K_S[i,:])*np.linalg.norm(RK2A_S[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(LH2K_S[i,:], LK2A_S[i,:]) / (np.linalg.norm(LH2K_S[i,:])*np.linalg.norm(LK2A_S[i,:]))) * 180 / math.pi
                RK_FL = np.vstack([RK_FL, X])
                LK_FL = np.vstack([LK_FL, Y])
            RK_FL = RK_FL[:,0]
            LK_FL = LK_FL[:,0]
            #       Transverse plane (internal-external)
            RK_INT = np.arccos(np.dot(RH2K_T[0,:], RK2A_T[0,:]) / (np.linalg.norm(RH2K_T[0,:])*np.linalg.norm(RK2A_T[0,:]))) * 180 / math.pi
            LK_INT = np.arccos(np.dot(LH2K_T[0,:], LK2A_T[0,:]) / (np.linalg.norm(LH2K_T[0,:])*np.linalg.norm(LK2A_T[0,:]))) * 180 / math.pi
            for i in range(1,len(RH2K)):
                X = np.arccos(np.dot(RH2K_T[i,:], RK2A_T[i,:]) / (np.linalg.norm(RH2K_T[i,:])*np.linalg.norm(RK2A_T[i,:]))) * 180 / math.pi
                Y = np.arccos(np.dot(LH2K_T[i,:], LK2A_T[i,:]) / (np.linalg.norm(LH2K_T[i,:])*np.linalg.norm(LK2A_T[i,:]))) * 180 / math.pi
                RK_INT = np.vstack([RK_INT, X])
                LK_INT = np.vstack([LK_INT, Y])
            RK_INT = 180-RK_INT[:,0]
            LK_INT = 180-LK_INT[:,0]

            if trial[0:3] == 'DVJ':

                #   Knee-to-ankle separation ratio (KASR)
                KASR = np.absolute(RKJ[:,1] - LKJ[:,1]) / np.absolute(RAJ[:,1] - LAJ[:,1])

                #   Performance variables and event detection

                #       Maximal height reached during jump + Event detection (when set to 1)
                #           Generate figure for manual selection
                fig = plt.figure()
                fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (vertical trajectory -> z-axis)')
                if event_detection == 0:
                    fig.suptitle('Select maximal height reached during DVJ\n\nTip -> use RAJ trajectory (if complete), use RHL otherwise (orange)', fontsize=ft_size)
                else:
                    fig.suptitle('Select box jump, primary landing, take off, and secondary landing (in this order)\n\nTip -> use RAJ trajectory (if complete), use RHL otherwise (orange)', fontsize=ft_size)
                ax = fig.add_subplot(111)
                ax.plot(RAJ[:,2], label='RAJ')
                ax.plot(RFT[:,2], '--', label='RFT')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Height [pixels]')
                ax.legend()
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
                #           Points selection
                if event_detection == 0:
                    points = plt.ginput(1, show_clicks=True)
                    points = np.array(points[:]).astype(int)
                    max_idx = points[0,0]

                #       Event detection (DVJ)
                else:
                    #       Select landmarks for event detection
                    points = plt.ginput(4, show_clicks=True)
                    points = np.array(points[:]).astype(int)
                    idx = np.round(points[:,0],0)
                    #       Find peaks in negative signal
                    peaks,_ = find_peaks(-RAJz)
                    #       Box jump
                    # find actual local maxima closest to the selected point
                    R_boxjump = np.argmax(RAJz[idx[0]-event_window:idx[0]+event_window]) + idx[0]-event_window
                    #       Landings
                    # find actual local minima closest to the selected point
                    R_landing1 = np.argmin(RAJz[idx[1]-event_window:idx[1]+event_window]) + idx[1]-event_window
                    # find actual local minima closest to the selected point
                    R_landing2 = np.argmin(RAJz[idx[3]-event_window:idx[3]+event_window]) + idx[3]-event_window
                    #       Take-off
                    R_takeoff = idx[2]
                    #       Max jump
                    R_maxjump = np.argmax(RAJz[R_takeoff:R_landing2]) + R_takeoff
                    max_idx = R_maxjump
                    #       Combine events indices in one array
                    DVJ_events = [R_boxjump, R_landing1, R_takeoff, R_maxjump, R_landing2]

                    print('-> Events detected')

                #       Calculate maximal jump height
                min_height = np.nanmin(RAJ[:,2])
                max_height = np.nanmax(RAJ[max_idx-10:max_idx+10,2])
                Max_jump = max_height - min_height
                plt.close(fig)

            #   Maximal distance reached during single-legged hop test (no 3D data with Curv pipeline -> cannot calculate distance)
            if trial[1:5] == 'Dist':
                prehop_pos = 0
                posthop_pos = 0
                Dist_reached = 0

            #   Event detection (Dist)
            if event_detection == 1:
                if trial[0:5] == 'RDist':
                    #       Generate figure for manual selection
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (vertical trajectory -> z-axis)')
                    fig.suptitle('Select take off and landing\n (in this order)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(RAJ[:,2], label='RAJ')
                    ax.plot(RFT[:,2], '--', label='RFT')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Distance [pixels]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Select landing and take off (in this order)
                    points = plt.ginput(2, show_clicks=True)
                    points = np.array(points[:])
                    idx = points[:,0].astype(int)
                    #       Find peaks in negative signal
                    peaks,_ = find_peaks(-RAJz)
                    #       Take-off
                    # find actual local minima closest to the selected point
                    takeoff = np.argmin(RAJz[idx[0]-event_window:idx[0]+event_window]) + idx[0]-event_window
                    #       Landing
                    # find actual local minima closest to the selected point
                    landing = np.argmin(RAJz[idx[1]-event_window:idx[1]+event_window]) + idx[1]-event_window
                    #       Combine events indices in one array
                    Dist_events = [takeoff, landing]
                    plt.close(fig)
                if trial[0:5] == 'LDist':
                    #       Generate figure for manual selection
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (vertical trajectory -> z-axis)')
                    fig.suptitle('Select take off and landing\n (in this order)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(LAJ[:,2], label='LAJ')
                    ax.plot(LFT[:,2], '--', label='LFT')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Distance [pixels]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Select landing and take off (in this order)
                    points = plt.ginput(2, show_clicks=True)
                    points = np.array(points[:]).astype(int)
                    idx = points[:,0].astype(int)
                    #       Find peaks in negative signal
                    peaks,_ = find_peaks(-LAJz)
                    #       Take-off
                    # find actual local minima closest to the selected point
                    takeoff = np.argmin(LAJz[idx[0]-event_window:idx[0]+event_window]) + idx[0]-event_window
                    #       Landing
                    # find actual local minima closest to the selected point
                    landing = np.argmin(LAJz[idx[1]-event_window:idx[1]+event_window]) + idx[1]-event_window
                    #       Combine events indices in one array
                    Dist_events = [takeoff, landing]
                    plt.close(fig)

                print('-> Events detected')

            #   Time to reach 2.5 meters during single-legged hop test (no 3D data with Curv pipeline -> cannot calculate distance)(+ event detection)
            #       Right
            if trial[1:6] == 'Timed':
                fr_2_5m = np.array([np.nanmax(fr)])[0]
                dist_2_5m_t = 0
                Timed_dist = 0
                Time_reached = 0

            #   Event detection (Timed)
            if event_detection == 1:
                if trial[0:6] == 'RTimed':
                    #       Generate figure for manual selection
                    plt.clf()
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (forward trajectory -> z-axis)')
                    fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(RAJ[:,2], label='RAJ')
                    ax.plot(RFT[:,2], '--', label='RFT')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Height [pixels]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Starting point selection
                    point = plt.ginput(1, show_clicks=True)
                    point = np.array(point[:])
                    select_prehop_pos = int(point[0,1])
                    prehop_idx = int(point[0,0])
                    prehop_t = t[prehop_idx]
                    plt.close(fig)

                    print('-> Events detected')

                if trial[0:6] == 'LTimed':
                    #       Generate figure for manual selection
                    plt.clf()
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (forward trajectory -> z-axis)')
                    fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(LAJ[:,2], label='LAJ')
                    ax.plot(LFT[:,2], '--', label='LFT')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Height [pixels]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Starting point selection
                    point = plt.ginput(1, show_clicks=True)
                    point = np.array(point[:])
                    select_prehop_pos = int(point[0,1])
                    prehop_idx = int(point[0,0])
                    prehop_t = t[prehop_idx]
                    plt.close(fig)

                    print('-> Events detected')

            print('-> Biomechanical variables calculated')

# BIOMECHANICAL VARIABLES INTERPOLATION

            if interpolation == 1:
                #   Interpolate for missing observations using splines
                for bvar in ['TR_3D', 'TR_AB', 'TR_FL', 'TR_INT', 'RH_3D', 'RH_AB', 'RH_FL', 'RH_INT', 'LH_3D', 'LH_AB', 'LH_FL', 'LH_INT', 'RK_3D', 'RK_AB', 'RK_FL', 'RK_INT', 'LK_3D', 'LK_AB', 'LK_FL', 'LK_INT']:
                    exec(bvar + ' = cubic_spline_fill(' + bvar + ', mode=interpolation_mode)')

                print('-> Biomechanical variables interpolated')

# RESULTS EXPORT

            #   Frames
            f_frame = pd.MultiIndex.from_product([['Frame'], ['index']])
            data_frame = np.transpose(fr)
            df_frame = pd.DataFrame(data_frame, columns=f_frame)

            #   Time
            f_t = pd.MultiIndex.from_product([['Time'], ['secs']])
            data_t = np.transpose(t)
            df_t = pd.DataFrame(data_t, columns=f_t)

            #   Joints position, velocity and acceleration
            #       Position
            f_joints = pd.MultiIndex.from_product([['Right Hip', 'Right Knee', 'Right Ankle', 'Right Shoulder', 'Left Hip', 'Left Knee', 'Left Ankle', 'Left Shoulder'], ['X [pixels]','Y [pixels]', 'Z [pixels]']])
            data_joints = np.hstack([RHJ, RKJ, RAJ, RSJ, LHJ, LKJ, LAJ, LSJ])
            df_joints = pd.DataFrame(data_joints, columns=f_joints)
            #       Velocity
            f_joints_vel = pd.MultiIndex.from_product([['Vel: Right Hip', 'Vel: Right Knee', 'Vel: Right Ankle', 'Vel: Right Shoulder', 'Vel: Left Hip', 'Vel: Left Knee', 'Vel: Left Ankle', 'Vel: Left Shoulder'], ['X [pixels/s]', 'Y [pixels/s]', 'Z [pixels/s]']])
            data_joints_vel = np.hstack([RHJ_vel, RKJ_vel, RAJ_vel, RSJ_vel, LHJ_vel, LKJ_vel, LAJ_vel, LSJ_vel])
            df_joints_vel = pd.DataFrame(data_joints_vel, columns=f_joints_vel)
            #       Acceleration
            f_joints_acc = pd.MultiIndex.from_product([['Acc: Right Hip', 'Acc: Right Knee', 'Acc: Right Ankle', 'Acc:Right Shoulder', 'Acc: Left Hip', 'Acc: Left Knee', 'Acc: Left Ankle', 'Acc: Left Shoulder'], ['X [pixels/s2]', 'Y [pixels/s2]', 'Z [pixels/s2]']])
            data_joints_acc = np.hstack([RHJ_acc, RKJ_acc, RAJ_acc, RSJ_acc, LHJ_acc, LKJ_acc, LAJ_acc, LSJ_acc])
            df_joints_acc = pd.DataFrame(data_joints_acc, columns=f_joints_acc)
            #      Concatenation
            df_joints = pd.concat([df_frame, df_t, df_joints, df_joints_vel, df_joints_acc], axis=1)

            #    Biomechanical variables
            #       Trunk angles
            f_trunk = pd.MultiIndex.from_product([['TRUNK'], ['TOTAL', 'AB-AD', 'EX-FL', 'INT-EXT']])
            data_trunk = np.transpose(np.vstack([TR_3D, TR_AB, TR_FL, TR_INT]))
            df_trunk = pd.DataFrame(data_trunk, columns=f_trunk)
            #       Hip angles
            f_hip = pd.MultiIndex.from_product([['Right HIP', 'Left HIP'], ['TOTAL', 'AB-AD', 'EX-FL', 'INT-EXT']])
            data_hip = np.transpose(np.vstack([RH_3D, RH_AB, RH_FL, RH_INT, LH_3D, LH_AB, LH_FL, LH_INT]))
            df_hip = pd.DataFrame(data_hip, columns=f_hip)
            #       Knee angles
            f_knee = pd.MultiIndex.from_product([['Right KNEE', 'Left KNEE'], ['TOTAL', 'AB-AD', 'EX-FL', 'INT-EXT']])
            data_knee = np.transpose(np.vstack([RK_3D, RK_AB, RK_FL, RK_INT, LK_3D, LK_AB, LK_FL, LK_INT]))
            df_knee = pd.DataFrame(data_knee, columns=f_knee)
            #       KASR
            if trial[0:3] == 'DVJ':
                f_kasr = pd.MultiIndex.from_product([['KASR'], ['value']])
                data_kasr = np.transpose(KASR)
                df_kasr = pd.DataFrame(data_kasr, columns=f_kasr)
            #       Max height reached during DVJ
            if trial[0:3] == 'DVJ':
                f_Max_height = pd.MultiIndex.from_product([['Height'],['pixels']])
                data_height = np.transpose(np.array([min_height, max_height, Max_jump]))
                df_Max_height = pd.DataFrame(data_height, columns=f_Max_height)
            #       Distance reached during single-legged hop test
            if trial[1:5] == 'Dist':
                f_Dist_reached = pd.MultiIndex.from_product([['Distance'],['pixels']])
                data_dist = np.transpose(np.array([prehop_pos, posthop_pos, Dist_reached]))
                df_Dist_reached = pd.DataFrame(data_dist, columns=f_Dist_reached)
            #       Time to reach 2.5 meters during single-legged hop test
            if trial[1:6] == 'Timed':
                f_Time_reached = pd.MultiIndex.from_product([['Time'],['secs']])
                data_time = np.transpose(np.array([0, dist_2_5m_t, Time_reached]))
                df_Time_reached = pd.DataFrame(data_time, columns=f_Time_reached)
            #   Events
            if event_detection == 1:
                #       DVJ
                if trial[0:3] == 'DVJ':
                    f_DVJ_fr = pd.MultiIndex.from_product([['Events'], ['frame']])
                    DVJ_events_fr = np.transpose(np.array([DVJ_events]))
                    df_DVJ_events_fr = pd.DataFrame(DVJ_events_fr, columns=f_DVJ_fr)
                    f_DVJ_t = pd.MultiIndex.from_product([['Events'], ['secs']])
                    DVJ_events_t = np.transpose(np.array([t[DVJ_events]]))
                    df_DVJ_events_t = pd.DataFrame(DVJ_events_t, columns=f_DVJ_t)
                #       Dist
                if trial[1:5] == 'Dist':
                    f_Dist_fr = pd.MultiIndex.from_product([['Events'], ['frame']])
                    Dist_events_fr = np.transpose(np.array([Dist_events]))
                    df_Dist_events_fr = pd.DataFrame(Dist_events_fr, columns=f_Dist_fr)
                    f_Dist_t = pd.MultiIndex.from_product([['Events'], ['secs']])
                    Dist_events_t = np.transpose(np.array([t[Dist_events]]))
                    df_Dist_events_t = pd.DataFrame(Dist_events_t, columns=f_Dist_t)
                #       Timed
                if trial[1:6] == 'Timed':
                    f_Timed_fr = pd.MultiIndex.from_product([['Events'], ['frame']])
                    Timed_events_fr = np.transpose(np.array([prehop_idx, fr_2_5m, Timed_dist]))
                    df_Timed_events_fr = pd.DataFrame(Timed_events_fr, columns=f_Timed_fr)
            #       Concatenation
            if event_detection == 0:
                if trial[0:3] == 'DVJ':
                    df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_kasr, df_Max_height], axis=1)
                elif trial[1:5] == 'Dist':
                    df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Dist_reached], axis=1)
                elif trial[1:6] == 'Timed':
                    df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Time_reached], axis=1)
            else:
                if trial[0:3] == 'DVJ':
                    df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_kasr, df_Max_height, df_DVJ_events_fr, df_DVJ_events_t], axis=1)
                elif trial[1:5] == 'Dist':
                    df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Dist_reached, df_Dist_events_fr, df_Dist_events_t], axis=1)
                elif trial[1:6] == 'Timed':
                    df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Time_reached, df_Timed_events_fr], axis=1)

            #   Export
            if joint_center_trajectories == 1:
                df_joints.to_csv(export_jc_path, index=False)
            if biomechanical_variables == 1:
                df_processed.to_csv(export_bvar_path, index=False)

            print('-> Results exported')

# PLOTTING

            if plot == 1:

                #   Generate time variable
                t = np.array([np.ones(len(key_points))*i for i in range(int(len(data)))]).flatten()

                #   Define relative height threshold for plot axes definition
                height = np.max(z)

                #   Generate new data frame
                df_plot = pd.DataFrame({"time": t ,"x" : x, "y" : y, "z" : z})

                #   Function to update graph for every frame
                def update_graph(num):
                    dat = df_plot[df_plot['time']==num]
                    graph._offsets3d = (dat.x, dat.y, dat.z)
                    title.set_text('Time = {} secs'.format(round(num/sf, 2)))

                #   Figure settings
                fig = plt.figure()
                fig.canvas.set_window_title('2D Animated Trial')
                ax = fig.add_subplot(111, projection='3d')
                title = ax.set_title('2D Animated Trial')
                ax.set_xlim([-1, 1])
                ax.set_xlabel('X')
                ax.set_ylim([int(np.nanmean(y))-height/2, int(np.nanmean(y))+height/2])
                ax.set_ylabel('Y')
                ax.set_zlim3d([0.0, height])
                ax.set_zlabel('Z')
                ax.view_init(elev=elevation, azim=azimuth)

                #       Data scatter
                dat = df_plot[df_plot['time']==0]
                graph = ax.scatter(dat.x, dat.y, dat.z)

                #       Animation
                ani = mpl.animation.FuncAnimation(fig, update_graph, int(len(data))-1, interval=1, blit=False)

                #       Save/Plot
                if save_ani == 1:
                    ani.save(export_vid_path, writer=writer)
                    print('-> Animation exported')
                else:
                    plt.show()
                    print('-> Animation generated')

print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')



