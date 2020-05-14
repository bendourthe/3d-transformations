# LIBRARIES IMPORT

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# DEPENDENCIES (FUNCTIONS)

# SETTINGS

#    Paths to Main Directory and Sub-Directories
DATADIR = "C:/Users/bdour/Documents/Work/Toronto/Sunnybrook/Projects/ACL Injury Screening/Data"
import_participants_folder =  'Raw\\Kinect\\OpenSim\\Model 2 (translations disabled)\\OpenSim only\\'
export_participants_folder = 'Processed\\Biomechanical variables\\Kinect\\OpenSim\\Model 2 (translations disabled)\\OpenSim only\\'

#   List of participants
#       Note: full list for this pipeline -> ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
participants = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']

#   Trials
#       Note: full list -> ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']
trials = ['DVJ_0', 'DVJ_1', 'DVJ_2', 'RDist_0', 'RDist_1', 'RDist_2', 'LDist_0', 'LDist_1', 'LDist_2', 'RTimed_0', 'RTimed_1', 'RTimed_2', 'LTimed_0', 'LTimed_1', 'LTimed_2']

#    Extensions
import_ext_data = '.mot'               # extension for imported data
export_ext_data = '.csv'               # extension for exported data

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
            #       Path to participant folder
            path = os.path.join(DATADIR, import_participants_folder + participant)
            #       Path to trial file
            file_path = os.path.join(path, 'Kinect-' + participant + '_' + trial + import_ext_data)

        #   Export
            #       Path for biomechanical variables
            export_bvar_path = os.path.join(DATADIR, export_participants_folder + participant + '\\Kinect-' + participant + '_' + trial + '_bvars' + export_ext_data)

# DATA IMPORT

            #   Read the .csv file and skip the first 10 rows (not part of the actual table)
            data = pd.read_csv(file_path, sep='\t', lineterminator='\r', skiprows=10)

            print('-> Data imported')

# DATA FORMATING

            #   Remove last row (nan)
            data = data.iloc[:-1]

            #       Isolate the time column
            time = np.array(data['\ntime'].apply(lambda x: float(x.split()[0])))

            #   Isolate hip and knee joint angles columns
            angles_list = ['RH_ab', 'RH_fl', 'RH_int', 'LH_ab', 'LH_fl', 'LH_int', 'RK_ab', 'RK_fl', 'RK_int', 'LK_ab', 'LK_fl', 'LK_int']
            for angle in angles_list:
                exec(angle + " = np.array(data['" + angle + "'])")

            print('-> Data formatted')

# DATA ALIGNMENT

            #   Some angles calculates by OpenSim appear with an opposite sign compared to the angles calculated by the Injury Fortune Teller 2.0.
            RK_ab = -RK_ab
            LK_ab = -LK_ab
            RK_fl = -RK_fl
            LK_fl = -LK_fl

# RESULTS EXPORT

        #   Frames
            f_frame = pd.MultiIndex.from_product([['Frame'], ['index']])
            data_frame = np.transpose(np.arange(len(time)))
            df_frame = pd.DataFrame(data_frame, columns=f_frame)

        #   Time
            f_t = pd.MultiIndex.from_product([['Time'], ['[secs]']])
            data_t = np.transpose(time)
            df_t = pd.DataFrame(data_t, columns=f_t)

        #   Biomechanical variables

            #       Trunk angles
            f_trunk = pd.MultiIndex.from_product([['TRUNK'], ['TOTAL', 'AB-AD', 'EX-FL', 'INT-EXT']])
            data_trunk = np.transpose(np.vstack([np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time)), np.zeros(len(time))]))
            df_trunk = pd.DataFrame(data_trunk, columns=f_trunk)
            #       Hip angles
            f_hip = pd.MultiIndex.from_product([['Right HIP', 'Left HIP'], ['TOTAL', 'AB-AD', 'EX-FL', 'INT-EXT']])
            data_hip = np.transpose(np.vstack([np.zeros(len(time)), RH_ab, RH_fl, RH_int, np.zeros(len(time)), LH_ab, LH_fl, LH_int]))
            df_hip = pd.DataFrame(data_hip, columns=f_hip)
            #       Knee angles
            f_knee = pd.MultiIndex.from_product([['Right KNEE', 'Left KNEE'], ['TOTAL', 'AB-AD', 'EX-FL', 'INT-EXT']])
            data_knee = np.transpose(np.vstack([np.zeros(len(time)), RK_ab, RK_fl, RK_int, np.zeros(len(time)), LK_ab, LK_fl, LK_int]))
            df_knee = pd.DataFrame(data_knee, columns=f_knee)
            #       KASR (filled with zeros to match with data format from other sources, not calculated by OpenSim)
            if trial[0:3] == 'DVJ':
                f_kasr = pd.MultiIndex.from_product([['KASR'], ['value']])
                data_kasr = np.transpose(np.zeros(len(time)))
                df_kasr = pd.DataFrame(data_kasr, columns=f_kasr)
            #       Max jump and single-legged hop distance (filled with zeros to match with data format from other sources, not calculated by OpenSim)
            if trial[0:3] == 'DVJ':
                f_Max_height = pd.MultiIndex.from_product([['Height'],['meters']])
                data_height = np.transpose(np.zeros(3))
                df_Max_height = pd.DataFrame(data_height, columns=f_Max_height)
            if trial[1:5] == 'Dist':
                f_Dist_reached = pd.MultiIndex.from_product([['Distance'],['meters']])
                data_dist = np.transpose(np.zeros(3))
                df_Dist_reached = pd.DataFrame(data_dist, columns=f_Dist_reached)

        #       Events (filled with zeros to match with data format from other sources, not calculated by OpenSim)
            #           DVJ
            if trial[0:3] == 'DVJ':
                f_events_fr = pd.MultiIndex.from_product([['Events'], ['frame']])
                f_events_t = pd.MultiIndex.from_product([['Events'], ['secs']])
                events = np.transpose(np.zeros(5))
                df_events_fr = pd.DataFrame(events, columns=f_events_fr)
                df_events_t = pd.DataFrame(events, columns=f_events_t)
            #       Dist
            if trial[1:5] == 'Dist':
                f_events_fr = pd.MultiIndex.from_product([['Events'], ['frame']])
                f_events_t = pd.MultiIndex.from_product([['Events'], ['secs']])
                events = np.transpose(np.zeros(2))
                df_events_fr = pd.DataFrame(events, columns=f_events_fr)
                df_events_t = pd.DataFrame(events, columns=f_events_t)
            #       Timed
            if trial[1:6] == 'Timed':
                f_events_fr = pd.MultiIndex.from_product([['Events'], ['frame']])
                f_events_t = pd.MultiIndex.from_product([['Events'], ['secs']])
                events = np.transpose(np.zeros(3))
                df_events_fr = pd.DataFrame(events, columns=f_events_fr)
                df_events_t = pd.DataFrame(events, columns=f_events_t)


            #       Concatenation
            if trial[0:3] == 'DVJ':
                df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_kasr, df_Max_height, df_events_fr, df_events_t], axis=1)
            elif trial[1:5] == 'Dist':
                df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_Dist_reached, df_events_fr, df_events_t], axis=1)
            elif trial[1:6] == 'Timed':
                df_bvars = pd.concat([df_frame, df_t, df_trunk, df_hip, df_knee, df_events_fr, df_events_t], axis=1)

        #    Export
            df_bvars.to_csv(export_bvar_path, index=False)

            print('-> Results exported')

print('')
print('--------------')
print('CODE COMPLETED')
print('--------------')
