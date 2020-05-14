# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

from scipy import signal                    # for filtering
from scipy import interpolate               # for interpolation
from scipy.signal import find_peaks
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

def glob2loc2glob(O, X, Y):
    '''
    Calculates the rotation and translation matrices allowing the transformations of
    a data set from a global to a local coordinate system (and vice versa)
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

def trans_glob2loc(Mglob, Rglob2loc, tglob2loc):
    '''
    Transforms a data set from a global to a local coordinate system
    Input:
        M: nx3 array corresponding to the data set to be transformed (n = number of points, columns = X, Y and Z coordinates)
        Rglob2loc: rotation matrix from global to local coordinate system
        tglob2loc: translation vector from global to local coordinate system
    Output:
        Mloc: nx3 array corresponding to the transformed data set
    '''
    Mloc = np.transpose(np.dot(Rglob2loc, np.transpose(Mglob)) + np.transpose(np.tile(tglob2loc, np.shape(Mglob)[0]).reshape(np.shape(Mglob)[0],3)) )

    return Mloc

def trans_loc2glob(Mloc, Rloc2glob, tloc2glob):
    '''
    Transforms a data set from a local to a global coordinate system
    Input:
        Mglob: nx3 array corresponding to the data set to be transformed (n = number of points, columns = X, Y and Z coordinates)
        Rloc2glob: rotation matrix from local to global coordinate system
        tloc2glob: translation vector from local to global coordinate system
    Output:
        Mglob: nx3 array corresponding to the transformed data set
    '''
    Mglob = np.transpose( np.dot(Rloc2glob, np.transpose(Mloc - np.tile(tloc2glob, np.shape(Mloc)[0]).reshape(np.shape(Mloc)[0],3)) ))

    return Mglob

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

# SETTINGS

#   Main Directory and Data type
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/ACL Injury Screening/Data"
data_type =  'Vicon'

#   List of participants
#       Note: full list for this pipeline -> ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']

#   Trials
#       Note: full list -> ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']
trials = ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']

#   List of joint centers (in the same order as they appear on the CSV file)
joints = ['RHJ', 'RKJ', 'RAJ', 'RSJ', 'LHJ', 'LKJ', 'LAJ', 'LSJ']

#   List of biomechanical variables (in the same order as they appear on the CSV file)
bvars = ['RKA', 'LKA', 'RHA', 'LHA']

#   Data processing
joint_center_trajectories = 0       # set to 1 of you want to export the processed results into CSV format
coronal_plane_orientation = 0       # set to 1 of you want to export the processed results into CSV format
biomechanical_variables = 0         # set to 1 of you want to export the processed results into CSV format
custom_jc = 0           # use custom-made pipeline to calculate the trajectories of the dynamic joint centers (hip, knee and ankle)
v3d_bvars = 1           # import biomechanical variables generated by Visual3D? (1=Yes - 0=No) -> please set to 0 if custom_jc = 1
event_detection = 1     # use manual event detection? (1:Yes - 0:No)
interpolation = 1       # interpolate missing observations? (1=Yes - 0=No)
interpolation_mode = 1  # select mode to deal with edge effects during cubic spline interpolation
                            # 0: set edges to zero
                            # 1: set to edge value
                            # 2: set edges to mean
                            # 3: set edges to NaN [default]
                            # 4: set edges to reflected signal

#   List of participants for each data collection period (data sets may differ slightly, especially the sync file, which had 2 seperate pulses in 2017, and a single pulse in 2019)
participants_2017 = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
participants_2019 = ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']

#   Sampling rate
sr = 100

#   Plotting
ft_size = 12            # define font size (for axes titles, etc.)
height = 2.5            # height of the plotting area (in m)
elevation = 10          # elevation for 3D visualization
azimuth = 360           # azimuth for 3D visualization
plot = 0                # generate plots? (1=Yes - 0=No)
save_animation = 0      # save the animated dynamic trial? (1=Yes - 0=No) Note: animated_trial must be 1 to work

#   Set up formatting for the movie files
Writer = mpl.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#   Event detection parameters and Initialization
height_threshold_factor = 80    # height threshold factor for the detection of peaks (in %)
                                # e.g. a value of X means that only peaks that have a y-value of more than X% of the max height of the signal will be found
event_window = 2                # number of frames that will be included to detect box jump, landings and take-off during DVJ
DVJ_events = np.zeros(5)
Dist_events = np.zeros([2])
Timed_events = np.array([])

#   Extensions
ext_data = '.csv'       # extension for data files
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
            #       Kinematics data
            #           Path for joints centers file (generated by Visual3D)
            jc_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/Visual3D/' + participant + '/Raw from Visual3D (reorganized)/' + data_type + '-' + participant + '_' + trial + '_raw' + ext_data)
            #           Path for markers file (from Vicon processed with Nexus)
            m_path = os.path.join(DATADIR, 'Raw\\' + data_type + '\\' + 'Nexus/' + '\\' + participant + '\\' + data_type + '-' + participant + '_' + trial + ext_data)
            #           Path for analog file (to sync markers data)
            a_path = os.path.join(DATADIR, 'Raw\\' + data_type + '\\' + 'Nexus/' + '\\' + participant + '\\' + data_type + '-' + participant + '_' + trial + '.A' + ext_data)
            #           Path for static calibration file (for custom calculation of dynamic joint centers trajectories)
            static = os.path.join(DATADIR, 'Raw\\' + data_type + '\\' + 'Nexus/' + '\\' + participant + '\\' + data_type + '-' + participant + '_SCal' + ext_data)
            #       Biomechanical data
            #           Path for biomechanical variables (generated by Visual3D)
            bvars_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/Visual3D/' + participant + '/Raw from Visual3D (reorganized)/' + data_type + '-' + participant + '_' + trial + '_raw' + ext_data)
            #           Path for orientation data (generated by Visual3D)
            orientation_path = os.path.join(DATADIR, 'Processed/Orientation/' + data_type + '/' + participant + '/Rotation matrices (Visual3D)/' + data_type + '-' + participant + '_' + trial + '_knee_cs_glob2loc' + ext_data)

        #   Export
            if custom_jc == 0:
                if interpolation == 1:
                    #       Path for joint centers data
                    export_jc_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/Visual3D/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_v3d_i_jc' + ext_data)
                    #       Path for biomechanical variables
                    if v3d_bvars == 0:
                        export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/Custom/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_v3d_cust_i_bvars' + ext_data)
                    else:
                        export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/Visual3D/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_v3d_i_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '-' + participant + '_' + trial + '_v3d_i' + ext_vid)
                else:
                    #       Path for joint centers data
                    export_jc_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/Visual3D/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_v3d_jc' + ext_data)
                    #       Path for biomechanical variables
                    if v3d_bvars == 0:
                        export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/Custom/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_v3d_cust_bvars' + ext_data)
                    else:
                        export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/Visual3D/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_v3d_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '-' + participant + '_' + trial + '_v3d' + ext_vid)
            else:
                if interpolation == 1:
                    #       Path for joint centers data
                    export_jc_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/Custom/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_cust_i_jc' + ext_data)
                    #       Path for biomechanical variables
                    export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/Custom/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_cust_i_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '-' + participant + '_' + trial + '_cust_i' + ext_vid)
                else:
                    #       Path for joint centers data
                    export_jc_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/Custom/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_cust_jc' + ext_data)
                    #       Path for biomechanical variables
                    export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/Custom/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_cust_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '-' + participant + '_' + trial + '_cust' + ext_vid)
                                #       Path for local frontal/coronal plane orientation
            export_orientation_path = os.path.join(DATADIR, 'Processed/Orientation/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_orientation' + ext_data)

# DATA IMPORT

    #   Import

        #       Joint centers data
            #           Read the .csv file and skip the first 2 rows (text and headers)
            jc_data = pd.read_csv(jc_path, skiprows=1)
            #           Convert to numpy array (to allow operations)
            jc_data = np.array(jc_data)
            #           Second column = time
            time = np.array(jc_data[:,1])
            #           Crop the frame and time columns
            jc_data = jc_data[:,2:]

        #       Markers data
            #           Read the .csv file and skip the first 6 rows (text and headers + first data row because error if first row included (not sure why))
            m_data = pd.read_csv(m_path, skiprows=5)
            #           Convert to numpy array (to allow operations)
            m_data = np.array(m_data)
            #           Remove first rows (no acceleration data before at least 2 frames) and columns (Frame and SubFrame) and convert from mm to m
            m_data = m_data[1:,2:]/1000
            #           First 3rd of the data = markers position
            markers = m_data[:,0:int(np.shape(m_data)[1]/3)]

        #       Analog data
            #           Read the .csv file and skip the first 3 rows (Sampling rate and first row of headers)
            a_data = pd.read_csv(a_path, skiprows=2)
            a_data = np.array(a_data)
            #           Find index of the 1st pulse analog channel (add 1 because 2nd column of the table has no header)
            idx1 = int(np.array(np.where(a_data[0,:] == '1'))) + 1
            #           Find index of the 2nd pulse analog channel
            idx2 = int(np.array(np.where(a_data[0,:] == '2'))) + 1
            #           Read the .csv file (again) and skip the first 5 rows (text and headers, avoid data to be as string)
            a_data = pd.read_csv(a_path, skiprows=5)
            a_data = np.array(a_data)
            #           Calculate the difference in the first column to identify the indexes of each non-analog frame
            diff_a = np.diff(a_data[:,0])
            #           Calculate the indexes of each non-analog frame (analog rate = 1000, non-analog rate = 100)
            idx100 = np.array(np.where(diff_a == 1)) + 1
            #           Resample the analog data to 100 Hz (same as non-analog data)
            a_data = np.array(a_data[idx100,:])
            #           Keep the two columns that have the pulse data
            a_data = a_data[0,1:,idx1-1:idx2]

        #       Static calibration data
            #           Read the .csv file and skip the first 6 rows (text and headers + first data row because error if first row included (not sure why))
            s_data = pd.read_csv(static, skiprows=5)
            #           Convert to numpy array (to allow operations)
            s_data = np.array(s_data)
            #           Remove first rows (no acceleration data before at least 2 frames) and columns (Frame and SubFrame) and convert from mm to m
            s_data = s_data[1:,2:]/1000
            #           First 3rd of the data = position
            pos_s = s_data[:,0:int(np.shape(s_data)[1]/3)]

        #       Orientation data
            #           Read the .csv file and skip the first 2 rows (text and headers)
            o_data = pd.read_csv(orientation_path, skiprows=1)
            #           Convert to numpy array (to allow operations)
            o_data = np.array(o_data)
            #           Crop the frame and time columns
            o_data = o_data[:,2:]
            #           Separate right from left knee orientation data
            r_o_data = o_data[:,0:9]
            l_o_data = o_data[:,9:]

        #       Biomechanical variables
            #           Read the .csv file and skip the first 2 rows (text and headers)
            bvars_data = pd.read_csv(bvars_path, skiprows=1)
            #           Convert to numpy array (to allow operations)
            bvars_data = np.array(bvars_data)
            #           Crop the frame and time columns
            bvars_data = bvars_data[:,2:]

            print('-> Data imported')

# DATA SYNC

            #   Crop the imported markers data to match with the other motion capture techniques
            #   NOTE: the joint centers and biomechanical variable data were already synced before being imported here
            #       -> only apply to static calibration and markers data

            #   Exception
            if participant == '02':
                jc_data = jc_data[0:len(markers),:]
                o_data = o_data[0:len(markers),:]
                r_o_data = r_o_data[0:len(markers),:]
                l_o_data = l_o_data[0:len(markers),:]
                bvars_data = bvars_data[0:len(markers),:]
                print('- Exception: Markers data unsynced')

            else:
                #   Find the indexes of the start and end pulses
                if participant in participants_2017:
                    pulse_1 = np.max(np.where(np.absolute(np.diff(a_data[:,0])) > np.max(a_data[:,0])/2))
                    pulse_2 = np.min(np.where(np.absolute(np.diff(a_data[:,1])) > np.max(a_data[:,1])/2))
                    start_pulse = np.min([pulse_1, pulse_2])
                    end_pulse = np.max([pulse_1, pulse_2])
                elif participant in participants_2019:
                    signal_1 = np.diff(a_data[:,0])
                    signal_2 = np.diff(a_data[:,1])
                    if np.shape(np.nonzero(signal_1))[1] > 0:
                        pulse_1 = np.where(signal_1 == np.max(signal_1))
                        pulse_2 = np.where(signal_1 == np.min(signal_1))
                    elif np.shape(np.nonzero(signal_2))[1] > 0:
                        pulse_1 = np.where(signal_2 == np.max(signal_2))
                        pulse_2 = np.where(signal_2 == np.min(signal_2))
                    start_pulse = np.min([pulse_1, pulse_2])
                    end_pulse = np.max([pulse_1, pulse_2])
                #   Crop the static calibration and dynamic data using the start and end pulses
                s_data = s_data[start_pulse:end_pulse,:]
                markers = markers[start_pulse:end_pulse,:]
                #   Ensure that the length of the joint centers, orientation and biomechanical variables data is the same as the markers data
                jc_data = jc_data[0:len(markers),:]
                o_data = o_data[0:len(markers),:]
                r_o_data = r_o_data[0:len(markers),:]
                l_o_data = l_o_data[0:len(markers),:]
                bvars_data = bvars_data[0:len(markers),:]

                print('-> Markers data synced')

# DATA INTERPOLATION

            if interpolation == 1:
                #   Interpolate for missing observations using splines
                for i in range(0,np.shape(jc_data)[1]):
                    jc_data[:,i] = cubic_spline_fill(jc_data[:,i], mode=interpolation_mode)
                for i in range(0,np.shape(markers)[1]):
                    markers[:,i] = cubic_spline_fill(markers[:,i], mode=interpolation_mode)
                for i in range(0,np.shape(bvars_data)[1]):
                    bvars_data[:,i] = cubic_spline_fill(bvars_data[:,i], mode=interpolation_mode)

                print('-> Data interpolated')

# JOINT CENTERS IDENTIFICATION

            for i in range(0, len(joints)):
                exec(joints[i] + ' = jc_data[:,3*' + str(i) + ':3*(' + str(i) + '+1)]')

            print('-> Joint centers identified')

# MARKERS IDENTIFICATION

            for i in range(1, int(np.shape(markers)[1]/3+1)):
                exec('pm' + str(i) + ' = markers[:,3*' + str(i) + ':3*(' + str(i) + '+1)]')

            print('-> Markers identified')

# BIOMECHANICAL VARIABLES IDENTIFICATION

            if v3d_bvars == 1:
                for i in range(0, len(bvars)):
                    exec(bvars[i] + ' = bvars_data[:,3*' + str(i) + ':3*(' + str(i) + '+1)]')

                print('-> Biomechanical variables identified')

# STATIC CALIBRATION JOINT CENTERS IDENTIFICATION

            if custom_jc == 1:
                #   Identify markers in the static calibration data
                for i in range(1, int(int(np.shape(s_data)[1]/3)/3)+1):
                    exec ('ps_m' + str(i) + '=' + str('np.array(pos_s[:,') + str(3*(i-1)) + ':' + str(3*i) + '])')
                #   Right hip joint (RHJ): marker RGT (#35)
                RHJ = ps_m35
                #   Left hip joint (RHJ): marker LGT (#36)
                LHJ = ps_m36
                #   Right knee joint (RKJ): centroid of markers RLK (#40) and RMK (#41)
                RKJ = np.mean(np.array([ps_m40[0,:], ps_m41[0,:]]), axis=0)
                for i in range(1,int(len(pos_s))):
                    X = np.mean(np.array([ps_m40[i,:], ps_m41[i,:]]), axis=0)
                    RKJ = np.vstack([RKJ, X])
                #   Left knee joint (LKJ): centroid of markers LLK (#42) and LMK (#43)
                LKJ = np.mean(np.array([ps_m42[0,:], ps_m43[0,:]]), axis=0)
                for i in range(1,int(len(pos_s))):
                    X = np.mean(np.array([ps_m42[i,:], ps_m43[i,:]]), axis=0)
                    LKJ = np.vstack([LKJ, X])
                #   Right ankle joint (RAJ): centroid of markers RLA (#18-dynamic) and RMA (#44)
                RAJ = np.mean(np.array([ps_m18[0,:], ps_m44[0,:]]), axis=0)
                for i in range(1,int(len(pos_s))):
                    X = np.mean(np.array([ps_m18[i,:], ps_m44[i,:]]), axis=0)
                    RAJ = np.vstack([RAJ, X])
                #   Left ankle joint (LAJ): centroid of markers LLA (#29-dynamic) and LMA (#47)
                LAJ = np.mean(np.array([ps_m29[0,:], ps_m47[0,:]]), axis=0)
                for i in range(1,int(len(pos_s))):
                    X = np.mean(np.array([ps_m29[i,:], ps_m47[i,:]]), axis=0)
                    LAJ = np.vstack([LAJ, X])

                print('-> Static calibration joint centers identified')

# DYNAMIC JOINT CENTERS TRAJECTORIES

            if custom_jc == 1:
                #   RHJ
                #       CS: Origin = O = RTH1 = m9 | X = RTH2 = m10 | Y = RTH4 = m12
                RHJ = find_joint_center(ps_m9, ps_m10, ps_m12, RHJ, pm9[0,:], pm10[0,:], pm12[0,:])
                for i in range(1, int(len(markers))):
                    X = find_joint_center(ps_m9, ps_m10, ps_m12, RHJ, pm9[i,:], pm10[i,:], pm12[i,:])
                    RHJ = np.vstack([RHJ, X])
                #   RKJ
                #       CS: Origin = O = RTH1 = m9 | X = RTH2 = m10 | Y = RTH4 = m12
                RKJ = find_joint_center(ps_m9, ps_m10, ps_m12, RKJ, pm9[0,:], pm10[0,:], pm12[0,:])
                for i in range(1, int(len(markers))):
                    X = find_joint_center(ps_m9, ps_m10, ps_m12, RKJ, pm9[i,:], pm10[i,:], pm12[i,:])
                    RKJ = np.vstack([RKJ, X])
                #   RAJ
                #       CS: Origin = O = RSH1 = m13 | X = RSH2 = m14 | Y = RSH4 = m16
                RAJ = find_joint_center(ps_m13, ps_m14, ps_m16, RAJ, pm13[0,:], pm14[0,:], pm16[0,:])
                for i in range(1, int(len(markers))):
                    X = find_joint_center(ps_m13, ps_m14, ps_m16, RAJ, pm13[i,:], pm14[i,:], pm16[i,:])
                    RAJ = np.vstack([RAJ, X])
                #   LHJ
                #       CS: Origin = O = LTH1 = m20 | X = LTH2 = m21 | Y = LTH4 = m23
                LHJ = find_joint_center(ps_m20, ps_m21, ps_m23, LHJ, pm20[0,:], pm21[0,:], pm23[0,:])
                for i in range(1, int(len(markers))):
                    X = find_joint_center(ps_m20, ps_m21, ps_m23, LHJ, pm20[i,:], pm21[i,:], pm23[i,:])
                    LHJ = np.vstack([LHJ, X])
                #   LKJ
                #       CS: Origin = O = LTH1 = m20 | X = LTH2 = m21 | Y = LTH4 = m23
                LKJ = find_joint_center(ps_m20, ps_m21, ps_m23, LKJ, pm20[0,:], pm21[0,:], pm23[0,:])
                for i in range(1, int(len(markers))):
                    X = find_joint_center(ps_m20, ps_m21, ps_m23, LKJ, pm20[i,:], pm21[i,:], pm23[i,:])
                    LKJ = np.vstack([LKJ, X])
                #   LAJ
                #       CS: Origin = O = LSH1 = m24 | X = LSH2 = m25 | Y = LSH4 = m27
                LAJ = find_joint_center(ps_m24, ps_m25, ps_m27, LAJ, pm24[0,:], pm25[0,:], pm27[0,:])
                for i in range(1, int(len(markers))):
                    X = find_joint_center(ps_m24, ps_m25, ps_m27, LAJ, pm24[i,:], pm25[i,:], pm27[i,:])
                    LAJ = np.vstack([LAJ, X])

                print('-> Dynamic joint centers trajectories calculated')

# JOINT CENTERS INTERPOLATION

            if custom_jc == 1:
                if interpolation == 1:
                    jc_list = ['RHJ', 'RKJ', 'RAJ', 'LHJ', 'LKJ', 'LAJ']
                    #    Interpolate for missing observations using splines
                    for jc in jc_list:
                        exec(jc + '= cubic_spline_fill_3D(' + jc + ', mode=interpolation_mode)')

                    print('-> Joint centers trajectories interpolated')

# DATA ALIGNMENT

            #   Initialize arrays for joint centers
            xj = [RAJ[0,0], RKJ[0,0], RHJ[0,0], RSJ[0,0], LHJ[0,0], LKJ[0,0], LAJ[0,0], LSJ[0,0]]
            yj = [RAJ[0,1], RKJ[0,1], RHJ[0,1], RSJ[0,1], LHJ[0,1], LKJ[0,1], LAJ[0,1], LSJ[0,1]]
            zj = [RAJ[0,2], RKJ[0,2], RHJ[0,2], RSJ[0,2], LHJ[0,2], LKJ[0,2], LAJ[0,2], LSJ[0,2]]

            #   Initialize arrays for dynamic markers
            xm = [pm1[0,0], pm2[0,0], pm3[0,0], pm4[0,0], pm5[0,0], pm6[0,0], pm7[0,0], pm8[0,0], pm9[0,0], pm10[0,0], pm11[0,0], pm12[0,0], pm13[0,0], pm14[0,0], pm15[0,0], pm16[0,0], pm17[0,0], pm18[0,0], pm19[0,0], pm20[0,0], pm21[0,0], pm22[0,0], pm23[0,0], pm24[0,0], pm25[0,0], pm26[0,0], pm27[0,0], pm28[0,0], pm29[0,0], pm30[0,0]]
            ym = [pm1[0,1], pm2[0,1], pm3[0,1], pm4[0,1], pm5[0,1], pm6[0,1], pm7[0,1], pm8[0,1], pm9[0,1], pm10[0,1], pm11[0,1], pm12[0,1], pm13[0,1], pm14[0,1], pm15[0,1], pm16[0,1], pm17[0,1], pm18[0,1], pm19[0,1], pm20[0,1], pm21[0,1], pm22[0,1], pm23[0,1], pm24[0,1], pm25[0,1], pm26[0,1], pm27[0,1], pm28[0,1], pm29[0,1], pm30[0,1]]
            zm = [pm1[0,2], pm2[0,2], pm3[0,2], pm4[0,2], pm5[0,2], pm6[0,2], pm7[0,2], pm8[0,2], pm9[0,2], pm10[0,2], pm11[0,2], pm12[0,2], pm13[0,2], pm14[0,2], pm15[0,2], pm16[0,2], pm17[0,2], pm18[0,2], pm19[0,2], pm20[0,2], pm21[0,2], pm22[0,2], pm23[0,2], pm24[0,2], pm25[0,2], pm26[0,2], pm27[0,2], pm28[0,2], pm29[0,2], pm30[0,2]]

            #   Generate arrays for joint centers and markers
            for i in range(1, int(len(jc_data))):
                k = [RAJ[i,0], RKJ[i,0], RHJ[i,0], RSJ[i,0], LHJ[i,0], LKJ[i,0], LAJ[i,0], LSJ[i,0]]
                l = [RAJ[i,1], RKJ[i,1], RHJ[i,1], RSJ[i,1], LHJ[i,1], LKJ[i,1], LAJ[i,1], LSJ[i,1]]
                m = [RAJ[i,2], RKJ[i,2], RHJ[i,2], RSJ[i,2], LHJ[i,2], LKJ[i,2], LAJ[i,2], LSJ[i,2]]
                xj = np.vstack([xj, k])
                yj = np.vstack([yj, l])
                zj = np.vstack([zj, m])
            for i in range(1, int(len(markers))):
                x = [pm1[i,0], pm2[i,0], pm3[i,0], pm4[i,0], pm5[i,0], pm6[i,0], pm7[i,0], pm8[i,0], pm9[i,0], pm10[i,0], pm11[i,0], pm12[i,0], pm13[i,0], pm14[i,0], pm15[i,0], pm16[i,0], pm17[i,0], pm18[i,0], pm19[i,0], pm20[i,0], pm21[i,0], pm22[i,0], pm23[i,0], pm24[i,0], pm25[i,0], pm26[i,0], pm27[i,0], pm28[i,0], pm29[i,0], pm30[i,0]]
                y = [pm1[i,1], pm2[i,1], pm3[i,1], pm4[i,1], pm5[i,1], pm6[i,1], pm7[i,1], pm8[i,1], pm9[i,1], pm10[i,1], pm11[i,1], pm12[i,1], pm13[i,1], pm14[i,1], pm15[i,1], pm16[i,1], pm17[i,1], pm18[i,1], pm19[i,1], pm20[i,1], pm21[i,1], pm22[i,1], pm23[i,1], pm24[i,1], pm25[i,1], pm26[i,1], pm27[i,1], pm28[i,1], pm29[i,1], pm30[i,1]]
                z = [pm1[i,2], pm2[i,2], pm3[i,2], pm4[i,2], pm5[i,2], pm6[i,2], pm7[i,2], pm8[i,2], pm9[i,2], pm10[i,2], pm11[i,2], pm12[i,2], pm13[i,2], pm14[i,2], pm15[i,2], pm16[i,2], pm17[i,2], pm18[i,2], pm19[i,2], pm20[i,2], pm21[i,2], pm22[i,2], pm23[i,2], pm24[i,2], pm25[i,2], pm26[i,2], pm27[i,2], pm28[i,2], pm29[i,2], pm30[i,2]]
                xm = np.vstack([xm, x])
                ym = np.vstack([ym, y])
                zm = np.vstack([zm, z])

            #   Flatten joint centers and markers arrays
            xj = xj.flatten()
            yj = yj.flatten()
            zj = zj.flatten()
            xm = xm.flatten()
            ym = ym.flatten()
            zm = zm.flatten()

            #   Calculate means and min
            mean_x = np.nanmean(xm)
            mean_y = np.nanmean(ym)
            min_z = np.nanmin(zm)

            #   Align data around 0 for x and y, and translates z to have only positive values
            for i in range(1,31):
                exec ('pm' + str(i) + '[:,0] = pm' + str(i) + '[:,0] - mean_x')
                exec ('pm' + str(i) + '[:,1] = pm' + str(i) + '[:,1] - mean_y')
                exec ('pm' + str(i) + '[:,2] = pm' + str(i) + '[:,2] - min_z')
            for i in range(0, len(joints)):
                exec(joints[i] + '[:,0] = ' + joints[i] + '[:,0] - mean_x')
                exec(joints[i] + '[:,1] = ' + joints[i] + '[:,1] - mean_y')
                exec(joints[i] + '[:,2] = ' + joints[i] + '[:,2] - min_z')

            print('-> Data aligned')

# JOINTS VELOCITIES & ACCELERATIONS

            if joint_center_trajectories == 1:

                #   Velocity: calculated as velocity(frame_i) = [position(frame_i) - position(frame_i-1)] * sampling_rate
                RHJ_vel = np.array(np.diff(RHJ, axis=0)*sr)
                RKJ_vel = np.array(np.diff(RKJ, axis=0)*sr)
                RAJ_vel = np.array(np.diff(RAJ, axis=0)*sr)
                RSJ_vel = np.array(np.diff(RSJ, axis=0)*sr)
                LHJ_vel = np.array(np.diff(LHJ, axis=0)*sr)
                LKJ_vel = np.array(np.diff(LKJ, axis=0)*sr)
                LAJ_vel = np.array(np.diff(LHJ, axis=0)*sr)
                LSJ_vel = np.array(np.diff(LSJ, axis=0)*sr)

                #   Acceleration: calculated as acceleration(frame_i) = [velocity(frame_i) - velocity(frame_i-1)] * sampling_rate
                RHJ_acc = np.array(np.diff(RHJ_vel, axis=0)*sr)
                RKJ_acc = np.array(np.diff(RKJ_vel, axis=0)*sr)
                RAJ_acc = np.array(np.diff(RAJ_vel, axis=0)*sr)
                RSJ_acc = np.array(np.diff(RSJ_vel, axis=0)*sr)
                LHJ_acc = np.array(np.diff(LHJ_vel, axis=0)*sr)
                LKJ_acc = np.array(np.diff(LKJ_vel, axis=0)*sr)
                LAJ_acc = np.array(np.diff(LHJ_vel, axis=0)*sr)
                LSJ_acc = np.array(np.diff(LSJ_vel, axis=0)*sr)

                empty_row = np.zeros(3)
                empty_row[:] = np.nan

                #   Note: first frame cannot have a velocity (as 2 observations are needed), so to maintain the same shape
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

# LOCAL FRONTAL/CORONAL PLANE ORIENTATION

            if coronal_plane_orientation == 1:

                #   Initialization
                R_adjLocalCoronalPlaneNormal = np.zeros(3)
                L_adjLocalCoronalPlaneNormal = np.zeros(3)

                for i in range(0,len(o_data)):
                    #   Reshape orientation data to a (3x3) rotation matrix, which represents the rotation of the knee coordinate system with respect to the global coordinate system
                    r_mat = np.reshape(r_o_data[i,:], (3,3))
                    l_mat = np.reshape(l_o_data[i,:], (3,3))
                    #   Transform global to local coordinate system (= knee coordinate system -> involves translating global coordinate system to the knee joint centers)
                    r_knee_local_cs = ((1,0,0), (0,1,0), (0,0,1))*r_mat + RKJ[i,:]
                    l_knee_local_cs = l_mat*((1,0,0), (0,1,0), (0,0,1)) + LKJ[i,:]
                    #   Calculate coordinates of local frontal/coronal planes normals (= x-axis of local knee coordinate system)
                    #       Non-normalized
                    R_adjLocalCoronalPlaneNormal_temp = r_knee_local_cs[0,:]
                    L_adjLocalCoronalPlaneNormal_temp = l_knee_local_cs[0,:]
                    #       Normalized
                    R_adjLocalCoronalPlaneNormal_temp = R_adjLocalCoronalPlaneNormal_temp / np.linalg.norm(R_adjLocalCoronalPlaneNormal_temp)
                    L_adjLocalCoronalPlaneNormal_temp = L_adjLocalCoronalPlaneNormal_temp / np.linalg.norm(L_adjLocalCoronalPlaneNormal_temp)
                    #       Combine each frame
                    R_adjLocalCoronalPlaneNormal = np.vstack([R_adjLocalCoronalPlaneNormal, R_adjLocalCoronalPlaneNormal_temp])
                    L_adjLocalCoronalPlaneNormal = np.vstack([L_adjLocalCoronalPlaneNormal, L_adjLocalCoronalPlaneNormal_temp])
                #       Remove first row of zeros from initialization
                R_adjLocalCoronalPlaneNormal = R_adjLocalCoronalPlaneNormal[1:,:]
                L_adjLocalCoronalPlaneNormal = L_adjLocalCoronalPlaneNormal[1:,:]

                print('-> Local frontal/coronal plane orientation calculated')

# BIOMECHANICAL VARIABLES CALCULATION & EVENT DETECTION

        #   Define frame and time vectors
            fr = np.array(range(0,int(len(jc_data))))
            t = fr/sr

            if biomechanical_variables == 1:

            #   Generate artificial landmarks (to match with Kinect and Curv data)
                SB = (RHJ + LHJ)/2           # spine base (mean of the Right and Left hip joints)
                SHM = (RSJ + LSJ)/2          # landmark between shoulders on the spine (mean of the Right and Left shoulder joints)

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
                if v3d_bvars == 0:
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
                    RH_FL = RH_FL[:,0]
                    LH_FL = LH_FL[:,0]
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
                else:
                    #       Identification of each sub-angle (generated by Visual3D)
                    RH_AB = RHA[:,0]
                    RH_FL = RHA[:,1]
                    RH_INT = RHA[:,2]+90
                    LH_AB = LHA[:,0]
                    LH_FL = LHA[:,1]
                    LH_INT = LHA[:,2]-90

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
                if v3d_bvars == 0:
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
                else:
                    #       Identification of each sub-angle (generated by Visual3D)
                    RK_AB = -RKA[:,0]
                    RK_FL = RKA[:,1]
                    RK_INT = RKA[:,2]
                    LK_AB = LKA[:,0]
                    LK_FL = LKA[:,1]
                    LK_INT = LKA[:,2]

                #   Identification of specific markers for plotting
                RHL = pm17
                RLA = pm18
                RFT = pm19
                LHL = pm28
                LLA = pm29
                LFT = pm30

                #   Isolate z-axis trajectories for event detection
                RAJz = RAJ[:,2]
                LAJz = LAJ[:,2]

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
                    ax.plot(RHL[:,2], '--', label='RHL')
                    ax.plot(RLA[:,2], '--', label='RLA')
                    ax.plot(RFT[:,2], '--', label='RFT')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Height [meters]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #           Points selection
                    if event_detection == 0:
                        points = plt.ginput(1, show_clicks=True)
                        points = np.array(points[:]).astype(int)
                        max_idx = points[0,0]

                    #       Event detection
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
                    min_height_RAJ = np.nanmin(RAJ[:,2])
                    max_height_RAJ = np.nanmax(RAJ[max_idx-10:max_idx+10,2])
                    min_height_RHL = np.nanmin(RHL[:,2])
                    max_height_RHL = np.nanmax(RHL[max_idx-10:max_idx+10,2])
                    Max_jump_RAJ = max_height_RAJ - min_height_RAJ
                    Max_jump_RHL = max_height_RHL - min_height_RHL
                    if np.max([Max_jump_RAJ, Max_jump_RHL]) == Max_jump_RAJ:
                        min_height = min_height_RAJ
                        max_height = max_height_RAJ
                        Max_jump = Max_jump_RAJ
                    elif np.max([Max_jump_RAJ, Max_jump_RHL]) == Max_jump_RHL:
                        min_height = min_height_RHL
                        max_height = max_height_RHL
                        Max_jump = Max_jump_RHL
                    plt.close(fig)

            #   Distance reached during single-legged hop test
                #       Right
                if trial[0:5] == 'RDist':
                    #       Generate figure for manual selection
                    plt.clf()
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (forward trajectory -> x-axis)')
                    fig.suptitle('Select position of ankle pre- and post-jump for distance calculation\n\nTip -> stick to one curve, preferably RHL (most consistent one)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(RAJ[:,0], label='RAJ')
                    ax.plot(RHL[:,0], '--', label='RHL')
                    ax.plot(RLA[:,0], '--', label='RLA')
                    ax.plot(RFT[:,0], '--', label='RFT')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Distance [meters]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Points selection
                    points = plt.ginput(2, show_clicks=True)
                    points = np.array(points[:])
                    prehop_idx = int(points[0,0])
                    posthop_idx = int(points[1,0])
                    select_prehop_pos = points[0,1]
                    select_posthop_pos = points[1,1]
                    #       Find closest curve to point selected (RAJ, RHL, RLA or RFT)
                    RAJ_prehop_dist = np.abs(RAJ[prehop_idx,0] - select_prehop_pos)
                    RHL_prehop_dist = np.abs(RHL[prehop_idx,0] - select_prehop_pos)
                    RLA_prehop_dist = np.abs(RLA[prehop_idx,0] - select_prehop_pos)
                    RFT_prehop_dist = np.abs(RFT[prehop_idx,0] - select_prehop_pos)
                    RAJ_posthop_dist = np.abs(RAJ[posthop_idx,0] - select_posthop_pos)
                    RHL_posthop_dist = np.abs(RHL[posthop_idx,0] - select_posthop_pos)
                    RLA_posthop_dist = np.abs(RLA[posthop_idx,0] - select_posthop_pos)
                    RFT_posthop_dist = np.abs(RFT[posthop_idx,0] - select_posthop_pos)
                    min_dist_prehop = np.nanmin([RAJ_prehop_dist, RHL_prehop_dist, RLA_prehop_dist, RFT_prehop_dist])
                    min_dist_posthop = np.nanmin([RAJ_posthop_dist, RHL_posthop_dist, RLA_posthop_dist, RFT_posthop_dist])
                    if min_dist_prehop == RAJ_prehop_dist and min_dist_posthop == RAJ_posthop_dist:
                        prehop_pos = RAJ[prehop_idx,0]
                        posthop_pos = RAJ[posthop_idx,0]
                    elif min_dist_prehop == RHL_prehop_dist and min_dist_posthop == RHL_posthop_dist:
                        prehop_pos = RHL[prehop_idx,0]
                        posthop_pos = RHL[posthop_idx,0]
                    elif min_dist_prehop == RLA_prehop_dist and min_dist_posthop == RLA_posthop_dist:
                        prehop_pos = RLA[prehop_idx,0]
                        posthop_pos = RLA[posthop_idx,0]
                    elif min_dist_prehop == RFT_prehop_dist and min_dist_posthop == RFT_posthop_dist:
                        prehop_pos = RFT[prehop_idx,0]
                    Dist_reached = np.abs(prehop_pos - posthop_pos)
                    plt.close(fig)

                    #   Event detection
                    if event_detection == 1:
                        #       Generate figure for manual selection
                        fig = plt.figure()
                        fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (vertical trajectory -> z-axis)')
                        fig.suptitle('Select take off and landing (in this order)\n\nTip -> stick to one curve, preferably RHL (most consistent one)', fontsize=ft_size)
                        ax = fig.add_subplot(111)
                        ax.plot(RAJ[:,2], label='RAJ')
                        ax.plot(RHL[:,2], '--', label='RHL')
                        ax.plot(RLA[:,2], '--', label='RLA')
                        ax.plot(RFT[:,2], '--', label='RFT')
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Distance [meters]')
                        ax.legend()
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        #       Select landing and take off (in this order)
                        points = plt.ginput(2, show_clicks=True)
                        points = np.array(points[:])
                        idx = points[:,0].astype(int)
                        plt.close(fig)
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

                        print('-> Events detected')

                #       Left
                if trial[0:5] == 'LDist':
                    #       Generate figure for manual selection
                    plt.clf()
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (forward trajectory -> x-axis)')
                    fig.suptitle('Select position of ankle pre- and post-jump for distance calculation\n\nTip -> stick to one curve, preferably RHL (most consistent one)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(LAJ[:,0], label='LAJ')
                    ax.plot(LHL[:,0], '--', label='LHL')
                    ax.plot(LLA[:,0], '--', label='LLA')
                    ax.plot(LFT[:,0], '--', label='LFT')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Distance [meters]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Points selection
                    points = plt.ginput(2, show_clicks=True)
                    points = np.array(points[:])
                    prehop_idx = int(points[0,0])
                    posthop_idx = int(points[1,0])
                    select_prehop_pos = points[0,1]
                    select_posthop_pos = points[1,1]
                    #       Find closest curve to point selected (LAJ, LHL, LLA or LFT)
                    LAJ_prehop_dist = np.abs(LAJ[prehop_idx,0] - select_prehop_pos)
                    LHL_prehop_dist = np.abs(LHL[prehop_idx,0] - select_prehop_pos)
                    LLA_prehop_dist = np.abs(LLA[prehop_idx,0] - select_prehop_pos)
                    LFT_prehop_dist = np.abs(LFT[prehop_idx,0] - select_prehop_pos)
                    LAJ_posthop_dist = np.abs(LAJ[posthop_idx,0] - select_posthop_pos)
                    LHL_posthop_dist = np.abs(LHL[posthop_idx,0] - select_posthop_pos)
                    LLA_posthop_dist = np.abs(LLA[posthop_idx,0] - select_posthop_pos)
                    LFT_posthop_dist = np.abs(LFT[posthop_idx,0] - select_posthop_pos)
                    min_dist_prehop = np.nanmin([LAJ_prehop_dist, LHL_prehop_dist, LLA_prehop_dist, LFT_prehop_dist])
                    min_dist_posthop = np.nanmin([LAJ_posthop_dist, LHL_posthop_dist, LLA_posthop_dist, LFT_posthop_dist])
                    if min_dist_prehop == LAJ_prehop_dist and min_dist_posthop == LAJ_posthop_dist:
                        prehop_pos = LAJ[prehop_idx,0]
                        posthop_pos = LAJ[posthop_idx,0]
                    elif min_dist_prehop == LHL_prehop_dist and min_dist_posthop == LHL_posthop_dist:
                        prehop_pos = LHL[prehop_idx,0]
                        posthop_pos = LHL[posthop_idx,0]
                    elif min_dist_prehop == LLA_prehop_dist and min_dist_posthop == LLA_posthop_dist:
                        prehop_pos = LLA[prehop_idx,0]
                        posthop_pos = LLA[posthop_idx,0]
                    elif min_dist_prehop == LFT_prehop_dist and min_dist_posthop == LFT_posthop_dist:
                        prehop_pos = LFT[prehop_idx,0]
                        posthop_pos = LFT[posthop_idx,0]
                    Dist_reached = np.abs(prehop_pos - posthop_pos)
                    plt.close(fig)

                    #   Event detection
                    if event_detection == 1:
                        #       Generate figure for manual selection
                        fig = plt.figure()
                        fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (vertical trajectory -> z-axis)')
                        fig.suptitle('Select take off and landing (in this order)\n\nTip -> stick to one curve, preferably RHL (most consistent one)', fontsize=ft_size)
                        ax = fig.add_subplot(111)
                        ax.plot(LAJ[:,2], label='LAJ')
                        ax.plot(LHL[:,2], '--', label='LHL')
                        ax.plot(LLA[:,2], '--', label='LLA')
                        ax.plot(LFT[:,2], '--', label='LFT')
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Distance [meters]')
                        ax.legend()
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        #       Select landing and take off (in this order)
                        points = plt.ginput(2, show_clicks=True)
                        points = np.array(points[:]).astype(int)
                        idx = points[:,0].astype(int)
                        plt.close(fig)
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

                        print('-> Events detected')

            #   Time to reach 2.5 meters during single-legged hop test (+ event detection)
                #       Right
                if trial[0:6] == 'RTimed':
                    if np.max(RAJ[:,0]) - np.min(RAJ[:,0]) <= 2.5 and np.max(RHL[:,0]) - np.min(RHL[:,0]) <= 2.5 and np.max(RLA[:,0]) - np.min(RLA[:,0]) <= 2.5 and np.max(RFT[:,0]) - np.min(RFT[:,0]) <= 2.5:
                        print('     Warning! Total distance captured is shorter than 2.5 meters')
                        prehop_idx = 0
                        fr_2_5m = np.array([np.nanmax(fr)])
                        prehop_t = 0
                        dist_2_5m_t = 0
                        Timed_dist = np.nanmax([(np.max(RAJ[:,0]) - np.min(RAJ[:,0])), (np.max(RHL[:,0]) - np.min(RHL[:,0])), (np.max(RLA[:,0]) - np.min(RLA[:,0])), (np.max(RFT[:,0]) - np.min(RFT[:,0]))])
                        Time_reached = 0
                    else:
                        #       Generate figure for manual selection
                        plt.clf()
                        fig = plt.figure()
                        fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (forward trajectory -> x-axis)')
                        fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)\nTip -> preferably use RHL (most consistent one)', fontsize=ft_size)
                        ax = fig.add_subplot(111)
                        ax.plot(RAJ[:,0], label='RAJ')
                        ax.plot(RHL[:,0], '--', label='RHL')
                        ax.plot(RLA[:,0], '--', label='RLA')
                        ax.plot(RFT[:,0], '--', label='RFT')
                        ax.hlines(np.min([RAJ[0,0], RHL[0,0], RLA[0,0], RFT[0,0]]) + 2.5, 0, np.max(fr), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Distance [meters]')
                        ax.legend()
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        #       Starting point selection
                        point = plt.ginput(1, show_clicks=True)
                        point = np.array(point[:])
                        select_prehop_pos = point[0,1]
                        prehop_idx = int(point[0,0])
                        #   First look at RAJ
                        prehop_pos = RAJ[prehop_idx,0]
                        if prehop_pos+2.5 > np.nanmax(RAJ[:,0]):
                            print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RAJ)')
                            #   If RAJ doesn't go far enough, look at RHL
                            prehop_pos = RHL[prehop_idx,0]
                            if prehop_pos+2.5 > np.nanmax(RHL[:,0]):
                                print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RHL)')
                                #   If RHL doesn't go far enough, look at RLA
                                prehop_pos = RLA[prehop_idx,0]
                                if prehop_pos+2.5 > np.nanmax(RLA[:,0]):
                                    print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RLA)')
                                    #   If RLA doesn't go far enough, look at RFT
                                    prehop_pos = RFT[prehop_idx,0]
                                    if prehop_pos+2.5 > np.nanmax(RFT[:,0]):
                                        print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RFT)')
                                        fr_2_5m = np.array([np.nanmax(fr)])
                                        Timed_dist = RFT[fr_2_5m[0],0] - RFT[prehop_idx,0]
                                    else:
                                        fr_2_5m = np.where(RFT[:,0]>=prehop_pos+2.5)[0]
                                        Timed_dist = 2.5
                                else:
                                    fr_2_5m = np.where(RLA[:,0]>=prehop_pos+2.5)[0]
                                    Timed_dist = 2.5
                            else:
                                fr_2_5m = np.where(RHL[:,0]>=prehop_pos+2.5)[0]
                                Timed_dist = 2.5
                        else:
                            fr_2_5m = np.where(RAJ[:,0]>=prehop_pos+2.5)[0]
                            Timed_dist = 2.5
                        prehop_t = t[prehop_idx]
                        dist_2_5m_t = t[fr_2_5m][0]
                        Time_reached = dist_2_5m_t - prehop_t
                        plt.close(fig)

                    print('-> Events detected')

                #       Left
                if trial[0:6] == 'LTimed':
                    if np.max(LAJ[:,0]) - np.min(LAJ[:,0]) <= 2.5 and np.max(LHL[:,0]) - np.min(LHL[:,0]) <= 2.5 and np.max(LLA[:,0]) - np.min(LLA[:,0]) <= 2.5 and np.max(LFT[:,0]) - np.min(LFT[:,0]) <= 2.5:
                        print('     Warning! Total distance captured is shorter than 2.5 meters')
                        prehop_idx = 0
                        fr_2_5m = np.array([np.nanmax(fr)])
                        prehop_t = 0
                        dist_2_5m_t = 0
                        Timed_dist = np.nanmax([(np.max(LAJ[:,0]) - np.min(LAJ[:,0])), (np.max(LHL[:,0]) - np.min(LHL[:,0])), (np.max(LLA[:,0]) - np.min(LLA[:,0])), (np.max(LFT[:,0]) - np.min(LFT[:,0]))])
                        Time_reached = 0
                    else:
                        #       Generate figure for manual selection
                        plt.clf()
                        fig = plt.figure()
                        fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (forward trajectory -> x-axis)')
                        fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)\nTip -> preferably use LHL (most consistent one)', fontsize=ft_size)
                        ax = fig.add_subplot(111)
                        ax.plot(LAJ[:,0], label='LAJ')
                        ax.plot(LHL[:,0], '--', label='LHL')
                        ax.plot(LLA[:,0], '--', label='LLA')
                        ax.plot(LFT[:,0], '--', label='LFT')
                        ax.hlines(np.min([LAJ[0,0], LHL[0,0], LLA[0,0], LFT[0,0]]) + 2.5, 0, np.max(fr), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Distance [meters]')
                        ax.legend()
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        #       Starting point selection
                        point = plt.ginput(1, show_clicks=True)
                        point = np.array(point[:])
                        select_prehop_pos = point[0,1]
                        prehop_idx = int(point[0,0])
                        #   First look at LAJ
                        prehop_pos = LAJ[prehop_idx,0]
                        if prehop_pos+2.5 > np.nanmax(LAJ[:,0]):
                            print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (LAJ)')
                            #   If LAJ doesn't go far enough, look at RHL
                            prehop_pos = LHL[prehop_idx,0]
                            if prehop_pos+2.5 > np.nanmax(LHL[:,0]):
                                print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters LHL)')
                                #   If LHL doesn't go far enough, look at LLA
                                prehop_pos = LLA[prehop_idx,0]
                                if prehop_pos+2.5 > np.nanmax(LLA[:,0]):
                                    print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (LLA)')
                                    #   If LLA doesn't go far enough, look at LFT
                                    prehop_pos = LFT[prehop_idx,0]
                                    if prehop_pos+2.5 > np.nanmax(LFT[:,0]):
                                        print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (LFT)')
                                        fr_2_5m = np.array([np.nanmax(fr)])
                                        Timed_dist = LFT[fr_2_5m[0],0] - LFT[prehop_idx,0]
                                    else:
                                        fr_2_5m = np.where(LFT[:,0]>=prehop_pos+2.5)[0]
                                        Timed_dist = 2.5
                                else:
                                    fr_2_5m = np.where(LLA[:,0]>=prehop_pos+2.5)[0]
                                    Timed_dist = 2.5
                            else:
                                fr_2_5m = np.where(LHL[:,0]>=prehop_pos+2.5)[0]
                                Timed_dist = 2.5
                        else:
                            fr_2_5m = np.where(LAJ[:,0]>=prehop_pos+2.5)[0]
                            Timed_dist = 2.5
                        prehop_t = t[prehop_idx]
                        dist_2_5m_t = t[fr_2_5m][0]
                        Time_reached = dist_2_5m_t - prehop_t
                        plt.close(fig)

                    print('-> Events detected')

                print('-> Biomechanical variables calculated')

# RESULTS EXPORT

        #    Frames
            fr = np.array(range(0,int(len(jc_data))))
            f_frame = pd.MultiIndex.from_product([['Frame'], ['index']])
            data_frame = np.transpose(fr)
            df_frame = pd.DataFrame(data_frame, columns=f_frame)

        #    Time
            t = fr/sr
            f_t = pd.MultiIndex.from_product([['Time'], ['[secs]']])
            data_t = np.transpose(t)
            df_t = pd.DataFrame(data_t, columns=f_t)

        #   Joints position, velocity and acceleration
            if joint_center_trajectories == 1:
                #       Position
                f_joints = pd.MultiIndex.from_product([['Right Hip', 'Right Knee', 'Right Ankle', 'Right Shoulder', 'Left Hip', 'Left Knee', 'Left Ankle', 'Left Shoulder'], ['X [m]', 'Y [m]', 'Z [m]']])
                data_joints = np.hstack([RHJ, RKJ, RAJ, RSJ, LHJ, LKJ, LAJ, LSJ])
                df_joints = pd.DataFrame(data_joints, columns=f_joints)
                #       Velocity
                f_joints_vel = pd.MultiIndex.from_product([['Vel: Right Hip', 'Vel: Right Knee', 'Vel: Right Ankle', 'Vel: Right Shoulder', 'Vel: Left Hip', 'Vel: Left Knee', 'Vel: Left Ankle', 'Vel: Left Shoulder'], ['X [m/s]', 'Y [m/s]', 'Z [m/s]']])
                data_joints_vel = np.hstack([RHJ_vel, RKJ_vel, RAJ_vel, RSJ_vel, LHJ_vel, LKJ_vel, LAJ_vel, LSJ_vel])
                df_joints_vel = pd.DataFrame(data_joints_vel, columns=f_joints_vel)
                #       Acceleration
                f_joints_acc = pd.MultiIndex.from_product([['Acc: Right Hip', 'Acc: Right Knee', 'Acc: Right Ankle', 'Acc: Right Shoulder', 'Acc: Left Hip', 'Acc: Left Knee', 'Acc: Left Ankle', 'Acc: Left Shoulder'], ['X [m/s2]', 'Y [m/s2]', 'Z [m/s2]']])
                data_joints_acc = np.hstack([RHJ_acc, RKJ_acc, RAJ_acc, RSJ_acc, LHJ_acc, LKJ_acc, LAJ_acc, LSJ_acc])
                df_joints_acc = pd.DataFrame(data_joints_acc, columns=f_joints_acc)
                #       Concatenation
                df_joints = pd.concat([df_frame, df_t, df_joints, df_joints_vel, df_joints_acc], axis=1)

        #   Local frontal/coronal plane orientation
            if coronal_plane_orientation == 1:
                #       Right knee
                f_r_knee_plane = pd.MultiIndex.from_product([['Right Knee'], ['X [m]','Y [m]', 'Z [m]']])
                df_r_knee_plane = pd.DataFrame(R_adjLocalCoronalPlaneNormal, columns=f_r_knee_plane)
                #       Left knee
                f_l_knee_plane = pd.MultiIndex.from_product([['Left Knee'], ['X [m]','Y [m]', 'Z [m]']])
                df_l_knee_plane = pd.DataFrame(L_adjLocalCoronalPlaneNormal, columns=f_l_knee_plane)
                #       Concatenation
                df_orientation = pd.concat([df_frame, df_t, df_r_knee_plane, df_l_knee_plane], axis=1)

        #   Biomechanical variables
            if biomechanical_variables == 1:
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
                    f_Max_height = pd.MultiIndex.from_product([['Height'],['meters']])
                    data_height = np.transpose(np.array([min_height, max_height, Max_jump]))
                    df_Max_height = pd.DataFrame(data_height, columns=f_Max_height)
                #       Distance reached during single-legged hop test
                if trial[1:5] == 'Dist':
                    f_Dist_reached = pd.MultiIndex.from_product([['Distance'],['meters']])
                    data_dist = np.transpose(np.array([prehop_pos, posthop_pos, Dist_reached]))
                    df_Dist_reached = pd.DataFrame(data_dist, columns=f_Dist_reached)
                #       Time to reach 2.5 meters during single-legged hop test
                if trial[1:6] == 'Timed':
                    f_Time_reached = pd.MultiIndex.from_product([['Time'],['secs']])
                    data_time = np.transpose(np.array([prehop_t, dist_2_5m_t, Time_reached]))
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
                        Timed_events_fr = np.transpose(np.array([prehop_idx, fr_2_5m[0], Timed_dist]))
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

        #       Export
            if joint_center_trajectories == 1:
                df_joints.to_csv(export_jc_path, index=False)
            if coronal_plane_orientation == 1:
                df_orientation.to_csv(export_orientation_path, index=False)
            if biomechanical_variables == 1:
                df_processed.to_csv(export_bvar_path, index=False)

            print('-> Results exported')

# 3D PLOTTING

            if plot == 1:

                #   Time variables
                tj = np.array([np.ones(8)*i for i in range(int(len(jc_data)))]).flatten()
                tm = np.array([np.ones(30)*i for i in range(int(len(markers)))]).flatten()

                #   Normalize data around 0 for x and y, and translates z to have only positive values
                #       Note: (-) sign in front of a and b to change the direction and match with Vicon
                xj = xj - np.nanmean(xm)
                yj = yj - np.nanmean(ym)
                zj = zj - np.nanmin(zm)
                xm = xm - np.nanmean(xm)
                ym = ym - np.nanmean(ym)
                zm = zm - np.nanmin(zm)

                #   Joint CoR and markers variables (DataFrame format)
                dfj = pd.DataFrame({"time": tj ,"x" : xj, "y" : yj, "z" : zj})
                dfm = pd.DataFrame({"time": tm ,"x" : xm, "y" : ym, "z" : zm})

                #   Function to update graph for every frame
                def update_graph(num):
                    datam = dfm[dfm['time']==num]
                    graphm._offsets3d = (datam.x, datam.y, datam.z)
                    dataj = dfj[dfj['time']==num]
                    graphj._offsets3d = (dataj.x, dataj.y, dataj.z)

                    title.set_text('Time = {} secs'.format(num/sr))

                #   Figure settings
                fig = plt.figure()
                fig.canvas.set_window_title('3D Animated Dynamic Trial')
                ax = fig.add_subplot(111, projection='3d')
                title = ax.set_title('3D Test')
                ax.set_xlim3d([int(np.nanmean(xm))-height/2, int(np.nanmean(xm))+height/2])
                ax.set_xlabel('X')
                ax.set_ylim3d([int(np.nanmean(ym))-height/2, int(np.nanmean(ym))+height/2])
                ax.set_ylabel('Y')
                ax.set_zlim3d([0.0, height])
                ax.set_zlabel('Z')
                ax.view_init(elev=elevation, azim=azimuth)

                #   Data scatter
                datam = dfm[dfm['time']==0]
                graphm = ax.scatter(datam.x, datam.y, datam.z)
                dataj = dfj[dfj['time']==0]
                graphj = ax.scatter(dataj.x, dataj.y, dataj.z, marker='+')

                #   Animation
                ani = matplotlib.animation.FuncAnimation(fig, update_graph, int(len(jc_data))-1, interval=1, blit=False)
                plt.show()

                print('-> Animations generated')

                #   Save/Plot
                if save_animation == 1:
                    ani.save(export_vid_path, writer=writer)

                    print('-> Animations exported')

print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')
