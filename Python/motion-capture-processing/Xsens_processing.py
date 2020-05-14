# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

from mpl_toolkits.mplot3d import Axes3D                 # for 3D visualization

# SETTINGS

#   Main Directory and Data type
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/ACL Injury Screening/Data"
data_type =  'Xsens'

#   List of participants
#       Note: full list for this pipeline -> ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']

#   Trials
#       Note: full list -> ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']
trials = ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']

#   Sampling rate
sr = 100

#   Plotting
ft_size = 12            # define font size (for axes titles, etc.)
height = 2.5            # height of the plotting area (in m)
elevation = 10.         # elevation for 3D visualization
azimuth = 360           # azimuth for 3D visualization
plot = 0                # generate plots? (1=Yes - 0=No)
save_animation = 0      # save the animated trial? (Yes: 1 - No: 0) Note: plot must be 1 to work

#   Set up formatting for the movie files
Writer = mpl.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#   Extensions
ext_data = '.npy'       # extension for data files
ext_export = '.csv'     # extension for data export
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
            #       Path for raw data (.csv format)
            export_IMU_path = path + '/' + data_type + '-' + participant + '_' + trial + ext_export
            #       Path for joint centers data
            export_joints_path = os.path.join(DATADIR, 'Processed/Joint centers/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + ext_export)
            #       Path for biomechanical variables
            export_bvar_path = os.path.join(DATADIR, 'Processed/Biomechanical variables/' + data_type + '/' + participant + '/' + data_type + '-' + participant + '_' + trial + ext_export)
            #       Path for video
            export_vid_path = os.path.join(DATADIR, 'Processed/Visualization/Videos/' + data_type + '/' + data_type + '-' + participant + '_' + trial + ext_vid)

# DATA IMPORT

            #   Read the .npy file and skip the first 5 rows (text and headers)
            data = np.load(file_path)
            #   Convert to numpy array (to allow operations)
            data = np.array(data)

            print('-> Data imported')

# IMUs IDENTIFICATION

            #   Create one variable for each IMU
            #       Note: each IMU variable is an array of shape (m, 3), where m = number of frames
            #       Note: the corresponding generated variables won't show in the Outline but are still in the memory
            num_IMUs = int(np.shape(data)[1])
            for i in range(1,num_IMUs+1):
                exec ('IMU' + str(i) + '=' + str('np.array(data[:,') + str(i-1) + ',:])')

            print('-> IMUs identified')

# DATA ALIGNMENT

            #    Change the orientation of the data to align the frontal plane with the x-axis
            #    Note: based on orientation of the torso during the first recorded frame

            #        Combine torso IMUs in arrays (1 for x- and 1 for y-coordinates) - first frame only
            torso_x = [IMU1[0,0], IMU2[0,0], IMU3[0,0],  IMU4[0,0],  IMU5[0,0],  IMU6[0,0], IMU8[0,0], IMU9[0,0], IMU12[0,0], IMU13[0,0], IMU16[0,0], IMU20[0,0]]
            torso_y = [IMU1[0,1], IMU2[0,1], IMU3[0,1],  IMU4[0,1],  IMU5[0,1],  IMU6[0,1], IMU8[0,1], IMU9[0,1], IMU12[0,1], IMU13[0,1], IMU16[0,1], IMU20[0,1]]

            #        Calculate trend line equation to know the global direction of the torso IMUs
            trend = np.polyfit(torso_x, torso_y, 1)
            equ_trend = np.poly1d(trend)
            inter_x = -equ_trend[0]/equ_trend[1]    # Intersection with x-axis
            inter_y = equ_trend[0]                  # Intersection with y-axis

            #        Calculate angle between trendline and x-axis
            theta = -math.atan(inter_y/inter_x)

            #        Generate a rotation matrix along z-axis to align the data with the x-axis
            Rz = np.array([[math.cos(theta), -math.sin(theta), 0],[math.sin(theta), math.cos(theta), 0],[0, 0, 1]])

            #        Apply rotation matrix to data
            for i in range(1,num_IMUs+1):
                exec ('IMU' + str(i) + '=' + str('np.transpose(np.dot(Rz, np.transpose(IMU') + str(i) + ')))')

            print('-> Data aligned')

# LANDMARKS SELECTION AND LABELING

            #   Select landmarks
            SB = IMU1        # spine base
            SHM = IMU5       # landmark between shoulders on the spine
            RHJ = IMU16      # right hip joint
            RKJ = IMU17      # right knee joint
            RAJ = IMU18      # right ankle joint
            RFT = IMU19      # right foot
            RSJ = IMU9       # right shoulder joint
            LHJ = IMU20      # left hip joint
            LKJ = IMU21      # left knee joint
            LAJ = IMU22      # left ankle joint
            LFT = IMU23      # left foot
            LSJ = IMU13      # left shoulder joint

            print('-> Landmarks selection and labeling completed')

# JOINTS VELOCITIES & ACCELERATIONS

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
            t = fr/sr

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

                #   Maximal height reached during jump
                #       Generate figure for manual selection
                fig = plt.figure()
                fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (vertical trajectory -> z-axis)')
                fig.suptitle('Select maximal height reached during DVJ\n\nTip -> use RAJ trajectory (if complete)', fontsize=ft_size)
                ax = fig.add_subplot(111)
                ax.plot(RAJ[:,2], label='RAJ')
                ax.plot(RFT[:,2], '--', label='RFT')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Height [meters]')
                ax.legend()
                mng = plt.get_current_fig_manager()
                mng.window.state('zoomed')
                #       Points selection
                points = plt.ginput(1, show_clicks=True)
                points = np.array(points[:])
                selection_idx = points[0,0]
                min_height_RAJ = np.nanmin(RAJ[:,2])
                max_height_RAJ = np.nanmax(RAJ[int(selection_idx)-10:int(selection_idx)+10,2])
                min_height_RFT = np.nanmin(RFT[:,2])
                max_height_RFT = np.nanmax(RFT[int(selection_idx)-10:int(selection_idx)+10,2])
                Max_jump_RAJ = max_height_RAJ - min_height_RAJ
                Max_jump_RFT = max_height_RFT - min_height_RFT
                if np.max([Max_jump_RAJ, Max_jump_RFT]) == Max_jump_RAJ:
                    min_height = min_height_RAJ
                    max_height = max_height_RAJ
                    Max_jump = Max_jump_RAJ
                elif np.max([Max_jump_RAJ, Max_jump_RFT]) == Max_jump_RFT:
                    min_height = min_height_RFT
                    max_height = max_height_RFT
                    Max_jump = Max_jump_RFT

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

            #   Time to reach 2.5 meters during single-legged hop test
            #       Right
            if trial[0:6] == 'RTimed':
                if np.max(RAJ[:,0]) - np.min(RAJ[:,0]) >= 2.5:
                    #       Generate figure for manual selection
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Right ankle joint (forward trajectory -> x-axis)')
                    fig.suptitle('Select beginning of single-legged hop test\n\n(when foot/curve first takes off)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(t, RAJ[:,0], label='RAJ')
                    ax.hlines(np.min(RAJ[:,0]) + 2.5, 0, np.max(t), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                    ax.set_xlabel('Time [secs]')
                    ax.set_ylabel('Distance [meters]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Starting point selection
                    point = plt.ginput(1, show_clicks=True)
                    point = np.array(point[:])
                    #       Time calculation
                    prehop_t = point[0,0]
                    prehop_pos = point[0,1]
                    fr_2_5m = np.where(RAJ[:,0]>=prehop_pos+2.5)[0]
                    dist_2_5m_t = t[fr_2_5m][0]
                    Time_reached = dist_2_5m_t - prehop_t
                elif np.max(RFT[:,0]) - np.min(RFT[:,0]) >= 2.5:
                    #       Generate figure for manual selection
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Right heel (forward trajectory -> x-axis)')
                    fig.suptitle('Select beginning of single-legged hop test\n\n(when foot/curve first takes off)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(t, RFT[:,0], label='RFT')
                    ax.hlines(np.min(RFT[:,0]) + 2.5, 0, np.max(t), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                    ax.set_xlabel('Time [secs]')
                    ax.set_ylabel('Distance [meters]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Starting point selection
                    point = plt.ginput(1, show_clicks=True)
                    point = np.array(point[:])
                    #       Time calculation
                    prehop_t = point[0,0]
                    prehop_pos = point[0,1]
                    fr_2_5m = np.where(RFT[:,0]>=prehop_pos+2.5)[0]
                    dist_2_5m_t = t[fr_2_5m][0]
                    Time_reached = dist_2_5m_t - prehop_t
                else:
                    print('Warning! Distance captured is shorter than 2.5 meters')
                    prehop_t = 0
                    dist_2_5m_t = 0
                    Time_reached = 0
            #       Left
            if trial[0:6] == 'LTimed':
                if np.max(LAJ[:,0]) - np.min(LAJ[:,0]) >= 2.5:
                    #       Generate figure for manual selection
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Left ankle joint (forward trajectory -> x-axis)')
                    fig.suptitle('Select beginning of single-legged hop test\n\n(when foot/curve first takes off)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(t, LAJ[:,0], label='LAJ')
                    ax.hlines(np.min(LAJ[:,0]) + 2.5, 0, np.max(t), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                    ax.set_xlabel('Time [secs]')
                    ax.set_ylabel('Distance [meters]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Starting point selection
                    point = plt.ginput(1, show_clicks=True)
                    point = np.array(point[:])
                    #       Time calculation
                    prehop_t = point[0,0]
                    prehop_pos = point[0,1]
                    fr_2_5m = np.where(LAJ[:,0]>=prehop_pos+2.5)[0]
                    dist_2_5m_t = t[fr_2_5m][0]
                    Time_reached = dist_2_5m_t - prehop_t
                elif np.max(LFT[:,0]) - np.min(LFT[:,0]) >= 2.5:
                    #       Generate figure for manual selection
                    fig = plt.figure()
                    fig.canvas.set_window_title(participant + '-' + trial + ': Left heel (forward trajectory -> x-axis)')
                    fig.suptitle('Select beginning of single-legged hop test\n\n(when foot/curve first takes off)', fontsize=ft_size)
                    ax = fig.add_subplot(111)
                    ax.plot(t, LFT[:,0], label='LFT')
                    ax.hlines(np.min(LFT[:,0]) + 2.5, 0, np.max(t), colors='k', linestyles='dashed', label='2.5 meters from beginning')
                    ax.set_xlabel('Time [secs]')
                    ax.set_ylabel('Distance [meters]')
                    ax.legend()
                    mng = plt.get_current_fig_manager()
                    mng.window.state('zoomed')
                    #       Starting point selection
                    point = plt.ginput(1, show_clicks=True)
                    point = np.array(point[:])
                    #       Time calculation
                    prehop_t = point[0,0]
                    prehop_pos = point[0,1]
                    fr_2_5m = np.where(LFT[:,0]>=prehop_pos+2.5)[0]
                    dist_2_5m_t = t[fr_2_5m][0]
                    Time_reached = dist_2_5m_t - prehop_t
                else:
                    print('Warning! Distance captured is shorter than 2.5 meters')
                    prehop_t = 0
                    dist_2_5m_t = 0
                    Time_reached = 0

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
            #       Concatenation
            if trial[0:3] == 'DVJ':
                df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_kasr, df_Max_height], axis=1)
            elif trial[1:5] == 'Dist':
                df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Dist_reached], axis=1)
            elif trial[1:6] == 'Timed':
                df_processed = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Time_reached], axis=1)

            #    Export
            df_joints.to_csv(export_joints_path, index=False)
            df_processed.to_csv(export_bvar_path, index=False)

            print('-> Results exported')

# 3D PLOTTING

            if plot == 1:

                #   First Frame

                #       IMU data
                x = [IMU1[0,0], IMU2[0,0], IMU3[0,0], IMU4[0,0], IMU5[0,0], IMU6[0,0], IMU7[0,0], IMU8[0,0], IMU9[0,0], IMU10[0,0], IMU11[0,0], IMU12[0,0], IMU13[0,0], IMU14[0,0], IMU15[0,0], IMU16[0,0], IMU17[0,0], IMU18[0,0], IMU19[0,0], IMU20[0,0], IMU21[0,0], IMU22[0,0], IMU23[0,0]]
                y = [IMU1[0,1], IMU2[0,1], IMU3[0,1], IMU4[0,1], IMU5[0,1], IMU6[0,1], IMU7[0,1], IMU8[0,1], IMU9[0,1], IMU10[0,1], IMU11[0,1], IMU12[0,1], IMU13[0,1], IMU14[0,1], IMU15[0,1], IMU16[0,1], IMU17[0,1], IMU18[0,1], IMU19[0,1], IMU20[0,1], IMU21[0,1], IMU22[0,1], IMU23[0,1]]
                z = [IMU1[0,2], IMU2[0,2], IMU3[0,2], IMU4[0,2], IMU5[0,2], IMU6[0,2], IMU7[0,2], IMU8[0,2], IMU9[0,2], IMU10[0,2], IMU11[0,2], IMU12[0,2], IMU13[0,2], IMU14[0,2], IMU15[0,2], IMU16[0,2], IMU17[0,2], IMU18[0,2], IMU19[0,2], IMU20[0,2], IMU21[0,2], IMU22[0,2], IMU23[0,2]]

                #       Figure settings
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, color='blue', marker='^')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_xlim(int(np.nanmean(x))-height/2, int(np.nanmean(x))+height/2)
                ax.set_ylim(int(np.nanmean(x))-height/2, int(np.nanmean(x))+height/2)
                ax.set_zlim(0, height)
                ax.view_init(elev=elevation, azim=azimuth)
                fig.canvas.set_window_title('Firt Frame')

                plt.show(block=False)

                #   3D Animated Trial

                #       Frames
                num_fr = int(np.shape(data)[0])

                #       Time
                t = np.array([np.ones(num_IMUs)*i for i in range(num_fr)]).flatten()

                #       Loop
                a = x
                b = y
                c = z
                for i in range(1, num_fr):
                    x = [IMU1[i,0], IMU2[i,0], IMU3[i,0], IMU4[i,0], IMU5[i,0], IMU6[i,0], IMU7[i,0], IMU8[i,0], IMU9[i,0], IMU10[i,0], IMU11[i,0], IMU12[i,0], IMU13[i,0], IMU14[i,0], IMU15[i,0], IMU16[i,0], IMU17[i,0], IMU18[i,0], IMU19[i,0], IMU20[i,0], IMU21[i,0], IMU22[i,0], IMU23[i,0]]
                    y = [IMU1[i,1], IMU2[i,1], IMU3[i,1], IMU4[i,1], IMU5[i,1], IMU6[i,1], IMU7[i,1], IMU8[i,1], IMU9[i,1], IMU10[i,1], IMU11[i,1], IMU12[i,1], IMU13[i,1], IMU14[i,1], IMU15[i,1], IMU16[i,1], IMU17[i,1], IMU18[i,1], IMU19[i,1], IMU20[i,1], IMU21[i,1], IMU22[i,1], IMU23[i,1]]
                    z = [IMU1[i,2], IMU2[i,2], IMU3[i,2], IMU4[i,2], IMU5[i,2], IMU6[i,2], IMU7[i,2], IMU8[i,2], IMU9[i,2], IMU10[i,2], IMU11[i,2], IMU12[i,2], IMU13[i,2], IMU14[i,2], IMU15[i,2], IMU16[i,2], IMU17[i,2], IMU18[i,2], IMU19[i,2], IMU20[i,2], IMU21[i,2], IMU22[i,2], IMU23[i,2]]
                    a = np.vstack([a, x])
                    b = np.vstack([b, y])
                    c = np.vstack([c, z])

                #       Data flattening
                a = a.flatten()
                b = b.flatten()
                c = c.flatten()

                #       DataFrame format
                df = pd.DataFrame({"time": t ,"x" : a, "y" : b, "z" : c})

                #       Function to update graph for every frame
                def update_graph(num):
                    datadf = df[df['time']==num]
                    graph._offsets3d = (datadf.x, datadf.y, datadf.z)
                    title.set_text('Time = {} secs'.format(num/sr))

                #       Figure settings
                fig = plt.figure()
                fig.canvas.set_window_title('3D Animated Trial')
                ax = fig.add_subplot(111, projection='3d')
                title = ax.set_title('3D Test')
                ax.set_xlim3d([int(np.nanmean(a))-height/2, int(np.nanmean(a))+height/2])
                ax.set_xlabel('X')
                ax.set_ylim3d([int(np.nanmean(b))-height/2, int(np.nanmean(b))+height/2])
                ax.set_ylabel('Y')
                ax.set_zlim3d([0.0, height])
                ax.set_zlabel('Z')
                ax.view_init(elev=elevation, azim=azimuth)

                #       Data scatter
                datadf = df[df['time']==0]
                graph = ax.scatter(datadf.x, datadf.y, datadf.z, color='blue', marker='^')

                #       Animation
                ani = matplotlib.animation.FuncAnimation(fig, update_graph, \
                                                         num_fr-1, interval=1, blit=False)

                #       Save/Plot
                if save_animation == 1:
                    ani.save(export_vid_path, writer=writer)
                else:
                    plt.show()

                print('-> Animations generated')

print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')
