# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

from scipy import interpolate               # for interpolation
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

# DEPENDENCIES (FUNCTIONS)

def cubic_spline_resample(time, ym, sf):
    '''
    Fits xm cubic spline to the data and resamples the corresponding time series
    to the desired sampling rate.
    Input:
        time: nx1 array corresponding to the time (in secs)
        ym: nx1 array: tested time series (e.g. xm coordinates of xm marker)
        sf: resampling frequency (Hz)
    Output:
        y_resampled: array corresponding to resampled time series
            Note: Keeps the original values of the time series
        time_resampled: array corresponding to resampled time
    '''
    duration = time[-1]
    # calculate number of frames based on total duration and resampling rate
    num_fr = int(np.round(duration * sf, 0))
    if num_fr > len(time):
        # generate an array that contains the indices of all frames included in the original sampling rate
        fr = np.round(time * sf, 0).astype(int)
        # generate an array that contains the indices of all frames after resampling
        fr_resampled = np.arange(0,num_fr+1)
        # find the equation of the cubic spline that best fits time series
        cs = interpolate.CubicSpline(fr, ym)
        # generate new time array based on resampling rate
        time_resampled = fr_resampled / sf
        # apply cubic spline equation to obtained resampled time series
        y_resampled = cs(fr_resampled)
        # use original data to fill value that match the original sampling rate
        y_resampled[fr]= ym
    else:
        # generate an array that contains the indices of all frames included in the original sampling rate
        _, fr_resampled = np.unique(np.round(time * sf, 0).astype(int), return_index=True)
        time_resampled = time[fr_resampled]
        y_resampled = ym[fr_resampled]

    return y_resampled, time_resampled

def cubic_spline_resample_3D(time, ym, sf):
    '''
    Applies cubic_spline_resample to each dimension of xm 3D time series.
    Input:
        time: nx1 array corresponding to the time (in secs)
        ym: nx3 array: tested time series (e.g. xm coordinates of xm marker)
        sf: resampling frequency (Hz)
    Output:
        Y_resampled: array corresponding to resampled time series
            Note: Keeps the original values of the time series
        time_resampled: array corresponding to resampled time
    Dependencies:
        cubic_spline_resample
    '''
    xm, time_resampled = cubic_spline_resample(time, ym[:,0], sf)
    ym, _ = cubic_spline_resample(time, ym[:,1], sf)
    zm, _ = cubic_spline_resample(time, ym[:,2], sf)
    Y_resampled = np.transpose(np.array([xm, ym, zm]))

    return Y_resampled, time_resampled

# SETTINGS

#    Main Directory and Data type
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/ACL Injury Screening/Data"
data_type =  'Kinect'

#   List of participants
#       Note: full list for this pipeline -> ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']

#   Trials
#       Note: full list -> ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']
trials = ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']

#   Method
#       Choose between 'Original' (i.e. read biomechanical variables calculated by the Kinect software) and 'Custom' (i.e. code written to calculate biomechanical variables using custom method)
method = 'Original'
brace = 0                       # was the participant wearing knee braces? (1=Yes - 0=No)

#    Data processing
joint_center_trajectories = 0   # set to 1 of you want to export the processed results into CSV format
coronal_plane_orientation = 0   # set to 1 of you want to export the processed results into CSV format
biomechanical_variables = 0     # set to 1 of you want to export the processed results into CSV format
num_seg = 25                    # number of segments
resampling = 0                  # resample data to resampling frequency? (1=Yes - 0=No)
sf = 100                        # resampling frequency
event_detection = 1             # use manual event detection? (1:Yes - 0:No)

#    Plotting
ft_size = 12                    # define font size (for axes titles, etc.)
height = 2.5                    # height of the plotting area (in m)
elevation = 10.                 # elevation for 3D visualization
azimuth = 360                   # azimuth for 3D visualization
plot = 0                        # generate plots? (1=Yes - 0=No)
save_ani = 0                    # save animated trial? (1=Yes - 0=No)

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
ext_data = '.csv'               # extension for data files
ext_vid = '.mp4'                # extension for exported video file

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

# TIME FORMAT
            #       Define time format (which randomly changes between participants)
            if participant == '05' or participant == '06' or participant == '12' or participant == '15':
                format = 'xm.hh:mm:ss.ms'
            else:
                format = 'hh:mm:ss'

# PATHS DEFINITION

        #   Import
            #       Path for participant folder
            if brace == 0:
                path = os.path.join(DATADIR, 'Raw\\' + data_type + '\\' + participant)
            else:
                path = os.path.join(DATADIR, 'Raw\\' + data_type + '\\' + participant + '\\Brace')
            #       Path for trial file
            file_path = os.path.join(path, data_type + '-' + participant + '_' + trial + ext_data)

        #   Export
            if brace == 0:
                if resampling == 1:
                    #       Path for joint centers data
                    export_joints_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_' + str(sf) + 'Hz_jc' + ext_data)
                    #       Path for local frontal/coronal plane orientation
                    export_orientation_path = os.path.join(DATADIR, 'Processed/Orientation/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_' + str(sf) + 'Hz_orientation' + ext_data)
                    #       Path for biomechanical variables
                    export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/' + method + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_' + str(sf) + 'Hz_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '- ' + participant + '_' + trial + '_' + str(sf) + 'Hz' + ext_vid)
                else:
                    #       Path for joint centers data
                    export_joints_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_jc' + ext_data)
                    #       Path for local frontal/coronal plane orientation
                    export_orientation_path = os.path.join(DATADIR, 'Processed/Orientation/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_orientation' + ext_data)
                    #       Path for biomechanical variables
                    export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/' + method + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '-' + participant + '_' + trial + ext_vid)
            else:
                if resampling == 1:
                    #       Path for joint centers data
                    export_joints_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_' + str(sf) + 'Hz_brace_jc' + ext_data)
                    #       Path for local frontal/coronal plane orientation
                    export_orientation_path = os.path.join(DATADIR, 'Processed/Orientation/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_' + str(sf) + 'Hz_brace_orientation' + ext_data)
                    #       Path for biomechanical variables
                    export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/' + method + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_' + str(sf) + 'Hz_brace_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '- ' + participant + '_' + trial + '_' + str(sf) + 'Hz_brace' + ext_vid)
                else:
                    #       Path for joint centers data
                    export_joints_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_brace_jc' + ext_data)
                    #       Path for local frontal/coronal plane orientation
                    export_orientation_path = os.path.join(DATADIR, 'Processed/Orientation/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_brace_orientation' + ext_data)
                    #       Path for biomechanical variables
                    export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/' + method + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + '_brace_bvars' + ext_data)
                    #       Path for video
                    export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '-' + participant + '_' + trial + '_brace' + ext_vid)

# DATA IMPORT

        #   Import
            #       Read the .csv file and skip the first 2 rows (text and headers)
            data = pd.read_csv(file_path, skiprows=1)
            #       Convert to numpy array (to allow operations)
            data = np.array(data)
            #       Isolate first column which represents the date and time of recording for each frame (-10 frames to avoid problems with potential missing frames at the edge of the data set)
            time = data[0:len(data)-10,1]
            #       Remove first 2 columns (SourceFile and Time)
            data = data[0:len(data)-10,2:]

        #   Structure
            pos = data[:,0:num_seg*3]                   # first portion of the data = position
            bvar = data[:,175:]

            print('-> Data imported')

# LANDMARKS IDENTIFICATION

            #   Create one variable for each landmark (e.g. l1 has position data for the SpineBase landmark)
            #       Note: each landmark variable is an array of shape (1, m, 3), where m = number of frames
            #       Note: the corresponding generated variables won't show in the Outline but are still in the memory
            for i in range(1,num_seg*3+1):
                exec ('l' + str(i) + '=' + str('np.array(pos[:,') + str(3*(i-1)) + ':' + str(3*i) + '])')

            print('-> Landmarks identified')

# SAMPLING RATE CALCULATION

            #       Note: The sampling rate of the kinect data is non-linear (i.e. different sampling rates between seconds of recording)

            #   Extract time information
            if format == 'hh:mm:ss':
                #       Hours
                hrs = np.zeros(len(time))           # initialization
                for i in range(0,len(time)):
                    hrs[i] = str(time[i])[0:2]      # save relevant information
                hrs = np.double(hrs)                # convert string to double
                #       Minutes
                mins = np.zeros(len(time))
                for i in range(0,len(time)):
                    mins[i] = str(time[i])[3:5]
                mins = np.double(mins)
                #       Seconds
                secs = np.zeros(len(time))
                for i in range(0,len(time)):
                    secs[i] = str(time[i])[6:]
                secs = np.double(secs)
            elif format == 'xm.hh:mm:ss.ms':
                #        Hours
                hrs = np.zeros(len(time))           # initialization
                for i in range(0,len(time)):
                    hrs[i] = str(time[i])[2:4]      # save relevant information
                hrs = np.double(hrs)                # convert string to double
                #        Minutes
                mins = np.zeros(len(time))
                for i in range(0,len(time)):
                    mins[i] = str(time[i])[5:7]
                mins = np.double(mins)
                #        Seconds
                secs = np.zeros(len(time))
                for i in range(0,len(time)):
                    secs[i] = str(time[i])[8:]
                secs = np.double(secs)

            #   Convert HR:MIN:SEC format to SEC
            time_secs = 3600*hrs + 60*mins + secs

            #   Start time at zero
            time_secs = time_secs-time_secs[0]

            #   Identify when each new second of recording is done
            idx_sec = np.diff(np.round(time_secs,0))

            #   Calculate the sampling rate for each second window
            sr_sec = np.diff(np.nonzero(idx_sec))[0,:]

            print('-> Sampling rate calculated')

# DATA RESAMPLING

            #   Vicon data are sampled the 100 Hz. To facilitate the comparison, the kinect data is resampled to 100 Hz (linear) using xm cubic spline interpolation
            if resampling == 1:
                for i in range(1,num_seg+1):
                    exec ('l' + str(i) + ', resampled_time = cubic_spline_resample_3D(time_secs, l' + str(i) + ', sf)')

                print('-> Data resampled to ' + str(sf) + ' Hz')

            else:
                resampled_time = time_secs

# DATA ALIGNMENT

            #   The coordinate system of the Kinect is rotated 90 degrees compared to Vicon
            #   (i.e. x_vicon = z_kinect - y_vicon = x_kinect - z_vicon = y_kinect)
            #   To facilitate the comparison, a 90 degrees rotation is applied to the data
            for i in range(1,num_seg+1):
                exec ('x = -l' + str(i) + '[:,2]')
                exec ('y = -l' + str(i) + '[:,0]')
                exec ('z = l' + str(i) + '[:,1]')
                exec ('l' + str(i) + ' = np.transpose(np.array([x, y, z]))')

            #   Initialize arrays for X, Y, Z coordinates
            xm = [l1[0,0], l2[0,0], l3[0,0], l4[0,0], l5[0,0], l6[0,0], l7[0,0], l8[0,0], l9[0,0], l10[0,0], l11[0,0], l12[0,0], l13[0,0], l14[0,0], l15[0,0], l16[0,0], l17[0,0], l18[0,0], l19[0,0], l20[0,0], l21[0,0], l22[0,0], l23[0,0], l24[0,0], l25[0,0]]
            ym = [l1[0,1], l2[0,1], l3[0,1], l4[0,1], l5[0,1], l6[0,1], l7[0,1], l8[0,1], l9[0,1], l10[0,1], l11[0,1], l12[0,1], l13[0,1], l14[0,1], l15[0,1], l16[0,1], l17[0,1], l18[0,1], l19[0,1], l20[0,1], l21[0,1], l22[0,1], l23[0,1], l24[0,1], l25[0,1]]
            zm = [l1[0,2], l2[0,2], l3[0,2], l4[0,2], l5[0,2], l6[0,2], l7[0,2], l8[0,2], l9[0,2], l10[0,2], l11[0,2], l12[0,2], l13[0,2], l14[0,2], l15[0,2], l16[0,2], l17[0,2], l18[0,2], l19[0,2], l20[0,2], l21[0,2], l22[0,2], l23[0,2], l24[0,2], l25[0,2]]

            #   Generate arrays for X, Y, Z coordinates
            for i in range(1, int(len(resampled_time))):
                x = [l1[i,0], l2[i,0], l3[i,0], l4[i,0], l5[i,0], l6[i,0], l7[i,0], l8[i,0], l9[i,0], l10[i,0], l11[i,0], l12[i,0], l13[i,0], l14[i,0], l15[i,0], l16[i,0], l17[i,0], l18[i,0], l19[i,0], l20[i,0], l21[i,0], l22[i,0], l23[i,0], l24[i,0], l25[i,0]]
                y = [l1[i,1], l2[i,1], l3[i,1], l4[i,1], l5[i,1], l6[i,1], l7[i,1], l8[i,1], l9[i,1], l10[i,1], l11[i,1], l12[i,1], l13[i,1], l14[i,1], l15[i,1], l16[i,1], l17[i,1], l18[i,1], l19[i,1], l20[i,1], l21[i,1], l22[i,1], l23[i,1], l24[i,1], l25[i,1]]
                z = [l1[i,2], l2[i,2], l3[i,2], l4[i,2], l5[i,2], l6[i,2], l7[i,2], l8[i,2], l9[i,2], l10[i,2], l11[i,2], l12[i,2], l13[i,2], l14[i,2], l15[i,2], l16[i,2], l17[i,2], l18[i,2], l19[i,2], l20[i,2], l21[i,2], l22[i,2], l23[i,2], l24[i,2], l25[i,2]]
                xm = np.vstack([xm, x])
                ym = np.vstack([ym, y])
                zm = np.vstack([zm, z])

            #   Calculate means and min
            mean_x = np.nanmean(xm)
            mean_y = np.nanmean(ym)
            min_z = np.nanmin(zm)

            #   Normalize data around 0 for xm and ym, and translates zm to have only positive values
            for i in range(1,num_seg+1):
                exec ('l' + str(i) + '[:,0] = l' + str(i) + '[:,0] - mean_x')
                exec ('l' + str(i) + '[:,1] = l' + str(i) + '[:,1] - mean_y')
                exec ('l' + str(i) + '[:,2] = l' + str(i) + '[:,2] - min_z')

            print('-> Data aligned')

# LANDMARKS LABELING

            #   Select landmarks
            SB = l1        # spine base
            SHM = l21      # landmark between shoulders on the spine
            RHJ = l17      # right hip joint
            RKJ = l18      # right knee joint
            RAJ = l19      # right ankle joint
            RFT = l20      # right foot
            RSJ = l9       # right shoulder joint
            LHJ = l13      # left hip joint
            LKJ = l14      # left knee joint
            LAJ = l15      # left ankle joint
            LFT = l16      # left foot
            LSJ = l5       # left shoulder joint

            #   Isolate z-axis trajectories for event detection
            RAJz = RAJ[:,2]
            LAJz = LAJ[:,2]

            print('-> Landmarks labelled')

# JOINTS VELOCITIES & ACCELERATIONS

            if joint_center_trajectories == 1:

                #   Velocity: calculated as velocity(frame_i) = [position(frame_i) - position(frame_i-1)] * sampling_rate
                RHJ_vel = np.array(np.diff(RHJ, axis=0)*sf)
                RKJ_vel = np.array(np.diff(RKJ, axis=0)*sf)
                RAJ_vel = np.array(np.diff(RAJ, axis=0)*sf)
                RSJ_vel = np.array(np.diff(RSJ, axis=0)*sf)
                LHJ_vel = np.array(np.diff(LHJ, axis=0)*sf)
                LKJ_vel = np.array(np.diff(LKJ, axis=0)*sf)
                LAJ_vel = np.array(np.diff(LHJ, axis=0)*sf)
                LSJ_vel = np.array(np.diff(LSJ, axis=0)*sf)

                #   Acceleration: calculated as acceleration(frame_i) = [velocity(frame_i) - velocity(frame_i-1)] * sampling_rate
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

# LOCAL FRONTAL/CORONAL PLANE ORIENTATION

            if coronal_plane_orientation == 1:

                #   Initialization
                R_adjLocalCoronalPlaneNormal = np.zeros(3)
                L_adjLocalCoronalPlaneNormal = np.zeros(3)

                #   Step 1 – Define normalized tibial axes
                R_tibialAxis = (RAJ - RKJ) / np.linalg.norm(RAJ - RKJ)
                L_tibialAxis = (LAJ - LKJ) / np.linalg.norm(LAJ - LKJ)

                #   Step 2 – Define local frontal/coronal planes normals
                for i in range(0,len(R_tibialAxis)):
                    R_localCoronalPlaneNormal = np.cross((RHJ[i,:] - RKJ[i,:]).tolist(), (SB[i,:] - RKJ[i,:]).tolist())
                    L_localCoronalPlaneNormal = np.cross((LHJ[i,:] - LKJ[i,:]).tolist(), (SB[i,:] - LKJ[i,:]).tolist())

                    #   Step 3 – Define points that belongs to local sagittal planes (parallel to global sagittal plane passing by defined points), such as the knee joint center translated by the local frontal/coronal plane normal
                    R_pointInSagittalPlane = RKJ[i,:] + R_localCoronalPlaneNormal
                    L_pointInSagittalPlane = LKJ[i,:] + L_localCoronalPlaneNormal

                    #   Step 4 – Define normalized local sagittal planes normals
                    #       Non-normalized
                    R_localSagittalPlaneNormal = np.cross((RHJ[i,:] - RKJ[i,:]).tolist(), (R_pointInSagittalPlane - RKJ[i,:]).tolist())
                    L_localSagittalPlaneNormal = np.cross((LHJ[i,:] - LKJ[i,:]).tolist(), (L_pointInSagittalPlane - LKJ[i,:]).tolist())
                    #       Normalized
                    R_localSagittalPlaneNormal = R_localSagittalPlaneNormal / np.linalg.norm(R_localSagittalPlaneNormal)
                    L_localSagittalPlaneNormal = L_localSagittalPlaneNormal / np.linalg.norm(L_localSagittalPlaneNormal)

                    #   Step 5 – Calculate adjusted local frontal/coronal planes normals
                    #       Non-normalized
                    R_adjLocalCoronalPlaneNormal_temp = np.cross(R_localSagittalPlaneNormal.tolist(), R_tibialAxis[i,:].tolist())
                    L_adjLocalCoronalPlaneNormal_temp = np.cross(L_localSagittalPlaneNormal.tolist(), L_tibialAxis[i,:].tolist())
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
            fr = np.array(range(0,int(len(resampled_time))))
            t = resampled_time

            if biomechanical_variables == 1:

        #   INJURY FORTUNE TELLER 2.0 PIPELINE (knee joint angles = imported from IFT 2.0, trunk and hip angles calculated as projections onto global coordinate system)

                if method == 'Original':

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

                #   Knee angles
                    RK_3D = 180-bvar[:,10]
                    LK_3D = 180-bvar[:,4]
                    RK_AB = 180-bvar[:,9]
                    LK_AB = -bvar[:,3]-180
                    RK_FL = 180-bvar[:,7]
                    LK_FL = 180-bvar[:,1]
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
                        KASR = bvar[:,12]

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
                        ax.set_ylabel('Height [meters]')
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

                    if trial[1:5] == 'Dist':
                        #   Maximal distance reached during single-legged hop test
                        prehop_pos = 0
                        posthop_pos = data[0,191]
                        if posthop_pos == 'not detected':
                            posthop_pos = 0
                            Dist_reached = 0
                        else:
                            Dist_reached = data[0,191]

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
                                ax.set_ylabel('Distance [meters]')
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
                                ax.set_ylabel('Distance [meters]')
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

                    if trial[1:6] == 'Timed':
                        #   Time to reach 2.5 meters during single-legged hop test
                        dist_2_5m_t = data[0,191]
                        if dist_2_5m_t == 'not detected':
                            prehop_idx = 0
                            fr_2_5m = 0
                            dist_2_5m_t = 0
                            Time_reached = 0
                        else:
                            Time_reached = data[0,191]

                        #   Event detection (Timed)
                        if event_detection == 1:
                            if trial[0:6] == 'RTimed':
                                if np.max(RAJ[:,0]) - np.min(RAJ[:,0]) <= 2.5 and np.max(RFT[:,0]) - np.min(RFT[:,0]) <= 2.5:
                                    print('     Warning! Total distance captured is shorter than 2.5 meters')
                                    prehop_idx = 0
                                    fr_2_5m = np.array([np.nanmax(fr)])
                                    Timed_dist = np.nanmax([(np.max(RAJ[:,0]) - np.min(RAJ[:,0])), (np.max(RFT[:,0]) - np.min(RFT[:,0]))])
                                else:
                                    #       Generate figure for manual selection
                                    plt.clf()
                                    fig = plt.figure()
                                    fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (forward trajectory -> x-axis)')
                                    fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)', fontsize=ft_size)
                                    ax = fig.add_subplot(111)
                                    ax.plot(RAJ[:,0], label='RAJ')
                                    ax.plot(RFT[:,0], '--', label='RFT')
                                    ax.hlines(np.min([RAJ[0,0], RFT[0,0]]) + 2.5, 0, np.max(fr), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                                    ax.set_xlabel('Frame')
                                    ax.set_ylabel('Distance [meters]')
                                    ax.legend()
                                    mng = plt.get_current_fig_manager()
                                    mng.window.state('zoomed')
                                    #       Starting point selection
                                    point = plt.ginput(1, show_clicks=True)
                                    point = np.array(point[:])
                                    select_prehop_pos = int(point[0,1])
                                    prehop_idx = int(point[0,0])
                                    #       Find closest curve to point selected (RAJ or RFT)
                                    RAJ_prehop_dist = np.abs(RAJ[prehop_idx,0] - select_prehop_pos)
                                    RFT_prehop_dist = np.abs(RFT[prehop_idx,0] - select_prehop_pos)
                                    min_dist_prehop = np.nanmin([RAJ_prehop_dist, RFT_prehop_dist])
                                    if min_dist_prehop == RAJ_prehop_dist:
                                        prehop_pos = RAJ[prehop_idx,0]
                                        if prehop_pos+2.5 > np.nanmax(RAJ[:,0]):
                                            print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RAJ)')
                                            fr_2_5m = np.array([np.nanmax(fr)])
                                            Timed_dist = RAJ[fr_2_5m[0],0] - RAJ[prehop_idx,0]
                                        else:
                                            fr_2_5m = np.where(RAJ[:,0]>=prehop_pos+2.5)[0]
                                            Timed_dist = 2.5
                                    elif min_dist_prehop == RFT_prehop_dist:
                                        prehop_pos = RFT[prehop_idx,0]
                                        if prehop_pos+2.5 > np.nanmax(RFT[:,0]):
                                            print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RFT)')
                                            fr_2_5m = np.array([np.nanmax(fr)])
                                            Timed_dist = RFT[fr_2_5m[0],0] - RFT[prehop_idx,0]
                                        else:
                                            fr_2_5m = np.where(RFT[:,0]>=prehop_pos+2.5)[0]
                                            Timed_dist = 2.5
                                    plt.close(fig)
                            elif trial[0:6] == 'LTimed':
                                if np.max(LAJ[:,0]) - np.min(LAJ[:,0]) <= 2.5 and np.max(LFT[:,0]) - np.min(LFT[:,0]) <= 2.5:
                                    print('     Warning! Total distance captured is shorter than 2.5 meters')
                                    prehop_idx = 0
                                    fr_2_5m = np.array([np.nanmax(fr)])
                                    Timed_dist = np.nanmax([(np.max(LAJ[:,0]) - np.min(LAJ[:,0])), (np.max(LFT[:,0]) - np.min(LFT[:,0]))])
                                else:
                                    #       Generate figure for manual selection
                                    plt.clf()
                                    fig = plt.figure()
                                    fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (forward trajectory -> x-axis)')
                                    fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)', fontsize=ft_size)
                                    ax = fig.add_subplot(111)
                                    ax.plot(LAJ[:,0], label='LAJ')
                                    ax.plot(LFT[:,0], '--', label='LFT')
                                    ax.hlines(np.min([LAJ[0,0], LFT[0,0]]) + 2.5, 0, np.max(fr), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                                    ax.set_xlabel('Frame')
                                    ax.set_ylabel('Distance [meters]')
                                    ax.legend()
                                    mng = plt.get_current_fig_manager()
                                    mng.window.state('zoomed')
                                    #       Starting point selection
                                    point = plt.ginput(1, show_clicks=True)
                                    point = np.array(point[:])
                                    select_prehop_pos = int(point[0,1])
                                    prehop_idx = int(point[0,0])
                                    #       Find closest curve to point selected (LAJ or LFT)
                                    LAJ_prehop_dist = np.abs(LAJ[prehop_idx,0] - select_prehop_pos)
                                    LFT_prehop_dist = np.abs(LFT[prehop_idx,0] - select_prehop_pos)
                                    min_dist_prehop = np.nanmin([LAJ_prehop_dist, LFT_prehop_dist])
                                    if min_dist_prehop == LAJ_prehop_dist:
                                        prehop_pos = LAJ[prehop_idx,0]
                                        if prehop_pos+2.5 > np.nanmax(LAJ[:,0]):
                                            print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (LAJ)')
                                            fr_2_5m = np.array([np.nanmax(fr)])
                                            Timed_dist = LAJ[fr_2_5m[0],0] - LAJ[prehop_idx,0]
                                        else:
                                            fr_2_5m = np.where(LAJ[:,0]>=prehop_pos+2.5)[0]
                                            Timed_dist = 2.5
                                    elif min_dist_prehop == LFT_prehop_dist:
                                        prehop_pos = LFT[prehop_idx,0]
                                        if prehop_pos+2.5 > np.nanmax(LFT[:,0]):
                                            print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (LFT)')
                                            fr_2_5m = np.array([np.nanmax(fr)])
                                            Timed_dist = LFT[fr_2_5m[0],0] - LFT[prehop_idx,0]
                                        else:
                                            fr_2_5m = np.where(LFT[:,0]>=prehop_pos+2.5)[0]
                                            Timed_dist = 2.5
                                    plt.close(fig)

                            print('-> Events detected')

                    print('-> Biomechanical variables identification and calculation completed')

        #   CUSTOM PIPELINE (joint angles = projections onto global coordinate system)

                elif method == 'Custom':

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
                        ax.set_ylabel('Height [meters]')
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

                    #   Maximal distance reached during single-legged hop test
                    #       Right
                    if trial[0:5] == 'RDist':
                        #       Generate figure for manual selection
                        fig = plt.figure()
                        fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (forward trajectory -> x-axis)')
                        fig.suptitle('Select position of ankle pre- and post-jump for distance calculation\n\nTip -> stick to one curve (most consistent one)', fontsize=ft_size)
                        ax = fig.add_subplot(111)
                        ax.plot(RAJ[:,0], label='RAJ')
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
                        #       Find closest curve to point selection (RAJ or RFT)
                        RAJ_prehop_dist = np.abs(RAJ[prehop_idx,0] - select_prehop_pos)
                        RFT_prehop_dist = np.abs(RFT[prehop_idx,0] - select_prehop_pos)
                        RAJ_posthop_dist = np.abs(RAJ[posthop_idx,0] - select_posthop_pos)
                        RFT_posthop_dist = np.abs(RFT[posthop_idx,0] - select_posthop_pos)
                        min_dist_prehop = np.min([RAJ_prehop_dist, RFT_prehop_dist])
                        min_dist_posthop = np.min([RAJ_posthop_dist, RFT_posthop_dist])
                        if min_dist_prehop == RAJ_prehop_dist and min_dist_posthop == RAJ_posthop_dist:
                            prehop_pos = RAJ[prehop_idx,0]
                            posthop_pos = RAJ[posthop_idx,0]
                        elif min_dist_prehop == RFT_prehop_dist and min_dist_posthop == RFT_posthop_dist:
                            prehop_pos = RFT[prehop_idx,0]
                            posthop_pos = RFT[posthop_idx,0]
                        Dist_reached = np.abs(prehop_pos - posthop_pos)
                        plt.close(fig)
                    #       Left
                    if trial[0:5] == 'LDist':
                        #       Generate figure for manual selection
                        fig = plt.figure()
                        fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (forward trajectory -> x-axis)')
                        fig.suptitle('Select position of ankle pre- and post-jump for distance calculation\n\nTip -> stick to one curve (most consistent one)', fontsize=ft_size)
                        ax = fig.add_subplot(111)
                        ax.plot(LAJ[:,0], label='LAJ')
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
                        #       Find closest curve to point selection (LAJ or LFT)
                        LAJ_prehop_dist = np.abs(LAJ[prehop_idx,0] - select_prehop_pos)
                        LFT_prehop_dist = np.abs(LFT[prehop_idx,0] - select_prehop_pos)
                        LAJ_posthop_dist = np.abs(LAJ[posthop_idx,0] - select_posthop_pos)
                        LFT_posthop_dist = np.abs(LFT[posthop_idx,0] - select_posthop_pos)
                        min_dist_prehop = np.min([LAJ_prehop_dist, LFT_prehop_dist])
                        min_dist_posthop = np.min([LAJ_posthop_dist, LFT_posthop_dist])
                        if min_dist_prehop == LAJ_prehop_dist and min_dist_posthop == LAJ_posthop_dist:
                            prehop_pos = LAJ[prehop_idx,0]
                            posthop_pos = LAJ[posthop_idx,0]
                        elif min_dist_prehop == LFT_prehop_dist and min_dist_posthop == LFT_posthop_dist:
                            prehop_pos = LFT[prehop_idx,0]
                            posthop_pos = LFT[posthop_idx,0]
                        Dist_reached = np.abs(prehop_pos - posthop_pos)
                        plt.close(fig)

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
                            ax.set_ylabel('Distance [meters]')
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
                            ax.set_ylabel('Distance [meters]')
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

                    #   Time to reach 2.5 meters during single-legged hop test (+ event detection)
                    #       Right
                    if trial[0:6] == 'RTimed':
                        if np.max(RAJ[:,0]) - np.min(RAJ[:,0]) <= 2.5 and np.max(RFT[:,0]) - np.min(RFT[:,0]) <= 2.5:
                            print('     Warning! Total distance captured is shorter than 2.5 meters')
                            prehop_idx = 0
                            fr_2_5m = np.array([np.nanmax(fr)])
                            prehop_t = 0
                            dist_2_5m_t = 0
                            Timed_dist = np.nanmax([(np.max(RAJ[:,0]) - np.min(RAJ[:,0])), (np.max(RFT[:,0]) - np.min(RFT[:,0]))])
                            Time_reached = 0
                        else:
                            #       Generate figure for manual selection
                            plt.clf()
                            fig = plt.figure()
                            fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (forward trajectory -> x-axis)')
                            fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)', fontsize=ft_size)
                            ax = fig.add_subplot(111)
                            ax.plot(RAJ[:,0], label='RAJ')
                            ax.plot(RFT[:,0], '--', label='RFT')
                            ax.hlines(np.min([RAJ[0,0], RFT[0,0]]) + 2.5, 0, np.max(fr), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                            ax.set_xlabel('Frame')
                            ax.set_ylabel('Distance [meters]')
                            ax.legend()
                            mng = plt.get_current_fig_manager()
                            mng.window.state('zoomed')
                            #       Starting point selection
                            point = plt.ginput(1, show_clicks=True)
                            point = np.array(point[:])
                            select_prehop_pos = int(point[0,1])
                            prehop_idx = int(point[0,0])
                            #       Find closest curve to point selected (RAJ or RFT)
                            RAJ_prehop_dist = np.abs(RAJ[prehop_idx,0] - select_prehop_pos)
                            RFT_prehop_dist = np.abs(RFT[prehop_idx,0] - select_prehop_pos)
                            min_dist_prehop = np.nanmin([RAJ_prehop_dist, RFT_prehop_dist])
                            if min_dist_prehop == RAJ_prehop_dist:
                                prehop_pos = RAJ[prehop_idx,0]
                                if prehop_pos+2.5 > np.nanmax(RAJ[:,0]):
                                    print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RAJ)')
                                    fr_2_5m = np.array([np.nanmax(fr)])
                                    Timed_dist = RAJ[fr_2_5m[0],0] - RAJ[prehop_idx,0]
                                else:
                                    fr_2_5m = np.where(RAJ[:,0]>=prehop_pos+2.5)[0]
                                    Timed_dist = 2.5
                            elif min_dist_prehop == RFT_prehop_dist:
                                prehop_pos = RFT[prehop_idx,0]
                                if prehop_pos+2.5 > np.nanmax(RFT[:,0]):
                                    print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (RFT)')
                                    fr_2_5m = np.array([np.nanmax(fr)])
                                    Timed_dist = RFT[fr_2_5m[0],0] - RFT[prehop_idx,0]
                                else:
                                    fr_2_5m = np.where(RFT[:,0]>=prehop_pos+2.5)[0]
                                    Timed_dist = 2.5
                            prehop_t = t[prehop_idx]
                            dist_2_5m_t = t[fr_2_5m][0]
                            Time_reached = dist_2_5m_t - prehop_t
                            plt.close(fig)

                        print('-> Events detected')

                    #       Left
                    if trial[0:6] == 'LTimed':
                        if np.max(LAJ[:,0]) - np.min(LAJ[:,0]) <= 2.5 and np.max(LFT[:,0]) - np.min(LFT[:,0]) <= 2.5:
                            print('     Warning! Total distance captured is shorter than 2.5 meters')
                            prehop_idx = 0
                            fr_2_5m = np.array([np.nanmax(fr)])
                            prehop_t = 0
                            dist_2_5m_t = 0
                            Timed_dist = np.nanmax([(np.max(LAJ[:,0]) - np.min(LAJ[:,0])), (np.max(LFT[:,0]) - np.min(LFT[:,0]))])
                            Time_reached = 0
                        else:
                            #       Generate figure for manual selection
                            plt.clf()
                            fig = plt.figure()
                            fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (forward trajectory -> x-axis)')
                            fig.suptitle('Select beginning of single-legged hop test\n(when foot/curve first takes off)', fontsize=ft_size)
                            ax = fig.add_subplot(111)
                            ax.plot(LAJ[:,0], label='LAJ')
                            ax.plot(LFT[:,0], '--', label='LFT')
                            ax.hlines(np.min([LAJ[0,0], LFT[0,0]]) + 2.5, 0, np.max(fr), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                            ax.set_xlabel('Frame')
                            ax.set_ylabel('Distance [meters]')
                            ax.legend()
                            mng = plt.get_current_fig_manager()
                            mng.window.state('zoomed')
                            #       Starting point selection
                            point = plt.ginput(1, show_clicks=True)
                            point = np.array(point[:])
                            select_prehop_pos = int(point[0,1])
                            prehop_idx = int(point[0,0])
                            #       Find closest curve to point selected (RAJ or RFT)
                            LAJ_prehop_dist = np.abs(LAJ[prehop_idx,0] - select_prehop_pos)
                            LFT_prehop_dist = np.abs(LFT[prehop_idx,0] - select_prehop_pos)
                            min_dist_prehop = np.nanmin([LAJ_prehop_dist, LFT_prehop_dist])
                            if min_dist_prehop == LAJ_prehop_dist:
                                prehop_pos = LAJ[prehop_idx,0]
                                if prehop_pos+2.5 > np.nanmax(LAJ[:,0]):
                                    print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (LAJ)')
                                    fr_2_5m = np.array([np.nanmax(fr)])
                                    Timed_dist = LAJ[fr_2_5m[0],0] - LAJ[prehop_idx,0]
                                else:
                                    fr_2_5m = np.where(LAJ[:,0]>=prehop_pos+2.5)[0]
                                    Timed_dist = 2.5
                            elif min_dist_prehop == LFT_prehop_dist:
                                prehop_pos = LFT[prehop_idx,0]
                                if prehop_pos+2.5 > np.nanmax(LFT[:,0]):
                                    print('     Warning! Maximal distance capture after primary take-off is shorter than 2.5 meters (LFT)')
                                    fr_2_5m = np.array([np.nanmax(fr)])
                                    Timed_dist = LFT[fr_2_5m[0],0] - LFT[prehop_idx,0]
                                else:
                                    fr_2_5m = np.where(LFT[:,0]>=prehop_pos+2.5)[0]
                                    Timed_dist = 2.5
                            prehop_t = t[prehop_idx]
                            dist_2_5m_t = t[fr_2_5m][0]
                            Time_reached = dist_2_5m_t - prehop_t
                            plt.close(fig)

                        print('-> Events detected')

                    print('-> Biomechanical variables calculated')

# RESULTS EXPORT

        #   Frames
            f_frame = pd.MultiIndex.from_product([['Frame'], ['index']])
            data_frame = np.transpose(fr)
            df_frame = pd.DataFrame(data_frame, columns=f_frame)

        #   Time
            f_t = pd.MultiIndex.from_product([['Time'], ['[secs]']])
            data_t = np.transpose(t)
            df_t = pd.DataFrame(data_t, columns=f_t)

        #   Joints position, velocity and acceleration
            if joint_center_trajectories == 1:
                #        Position
                f_joints = pd.MultiIndex.from_product([['Right Hip', 'Right Knee', 'Right Ankle', 'Right Shoulder', 'Left Hip', 'Left Knee', 'Left Ankle', 'Left Shoulder'], ['X [m]','Y [m]', 'Z [m]']])
                data_joints = np.hstack([RHJ, RKJ, RAJ, RSJ, LHJ, LKJ, LAJ, LSJ])
                df_joints = pd.DataFrame(data_joints, columns=f_joints)
                #        Velocity
                f_joints_vel = pd.MultiIndex.from_product([['Vel: Right Hip', 'Vel: Right Knee', 'Vel: Right Ankle', 'Vel: Right Shoulder', 'Vel: Left Hip', 'Vel: Left Knee', 'Vel: Left Ankle', 'Vel: Left Shoulder'], ['X [m/s]', 'Y [m/s]', 'Z [m/s]']])
                data_joints_vel = np.hstack([RHJ_vel, RKJ_vel, RAJ_vel, RSJ_vel, LHJ_vel, LKJ_vel, LAJ_vel, LSJ_vel])
                df_joints_vel = pd.DataFrame(data_joints_vel, columns=f_joints_vel)
                #        Acceleration
                f_joints_acc = pd.MultiIndex.from_product([['Acc: Right Hip', 'Acc: Right Knee', 'Acc: Right Ankle', 'Acc:Right Shoulder', 'Acc: Left Hip', 'Acc: Left Knee', 'Acc: Left Ankle', 'Acc: Left Shoulder'], ['X [m/s2]', 'Y [m/s2]', 'Z [m/s2]']])
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
                        Timed_events_fr = np.transpose(np.array([prehop_idx, fr_2_5m[0], Timed_dist]))
                        df_Timed_events_fr = pd.DataFrame(Timed_events_fr, columns=f_Timed_fr)
                #       Concatenation
                if event_detection == 0:
                    if trial[0:3] == 'DVJ':
                        df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_kasr, df_Max_height], axis=1)
                    elif trial[1:5] == 'Dist':
                        df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Dist_reached], axis=1)
                    elif trial[1:6] == 'Timed':
                        df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Time_reached], axis=1)
                else:
                    if trial[0:3] == 'DVJ':
                        df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_kasr, df_Max_height, df_DVJ_events_fr, df_DVJ_events_t], axis=1)
                    elif trial[1:5] == 'Dist':
                        df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Dist_reached, df_Dist_events_fr, df_Dist_events_t], axis=1)
                    elif trial[1:6] == 'Timed':
                        df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Time_reached, df_Timed_events_fr], axis=1)

        #    Export
            if joint_center_trajectories == 1:
                df_joints.to_csv(export_joints_path, index=False)
            if coronal_plane_orientation == 1:
                df_orientation.to_csv(export_orientation_path, index=False)
            if biomechanical_variables == 1:
                df_bvars.to_csv(export_bvar_path, index=False)

            print('-> Results exported')

# 3D PLOTTING

            if plot == 1:

                #   Generate time variables
                t = np.array([np.ones(num_seg)*i for i in range(int(len(resampled_time)))]).flatten()

                #   Flatten key points array
                xm = xm.flatten()
                ym = ym.flatten()
                zm = zm.flatten()

                #   Normalize data around 0 for xm and ym, and translates zm to have only positive values
                #       Note: (-) sign in front of xm and ym to change the direction and match with Vicon
                xm = -(xm - np.mean(xm))
                ym = -(ym - np.mean(ym))
                zm = zm - np.min(zm)

                #   Generate data frame for key points
                df = pd.DataFrame({"time": t ,"x" : xm, "y" : ym, "z" : zm})

                #   Function to update graph for every frame
                def update_graph(num):
                    dat = df[df['time']==num]
                    graph._offsets3d = (dat.x, dat.y, dat.z)
                    title.set_text('Time = {} secs'.format(round(num/sf,2)))

                #   Figure settings
                fig = plt.figure()
                fig.canvas.set_window_title('3D Animated Trial')
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
                dat = df[df['time']==0]
                graph = ax.scatter(dat.x, dat.y, dat.z)

                #   Animation
                ani = mpl.animation.FuncAnimation(fig, update_graph, int(len(resampled_time))-1, interval=1, blit=False)

                #   Save/Plot
                if save_ani == 1:
                    ani.save(export_vid_path, writer=writer)
                    print('-> Animations exported')
                else:
                    plt.show()

                print('-> Animations generated')

print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')
