#!/usr/bin/env ipython
# author: stephanie hyland
# operating on data from 3_merged, interpolate lactate values for use in endpoints
# utility script of functions to be called from make_grid

import pandas as pd
import numpy as np
import pdb

bad_patients = open('bad_patients.csv', 'a')

def process_patient_single_lactate(p_df, measured_lactate_times, grid_size_in_minutes):
    print('patient with 1 lactate measurement!')
    if p_df.loc[measured_lactate_times[0], 'lactate_above_threshold']:
        temp_limit = int((3*60)/grid_size_in_minutes)     # max # entries on the time-grid corresponding to 3 hours
    else:
        # no limit
        temp_limit = None
    interpolated_lactate = p_df['lactate'].fillna(method='bfill', limit=temp_limit)
    interpolated_lactate.fillna(method='ffill', limit=temp_limit, inplace=True)
    return interpolated_lactate

def process_patient_multiple_lactate(p_df, measured_lactate_times, grid_size_in_minutes):
    interpolated_lactate = p_df['lactate'].copy()
    # EDGE CASES!
    for idx in range(1, len(measured_lactate_times)):
        previous_time = measured_lactate_times[idx-1]
        current_time = measured_lactate_times[idx]
        previous_status = p_df.loc[previous_time, 'lactate_above_threshold']
        current_status = p_df.loc[current_time, 'lactate_above_threshold']
        if previous_status == current_status:
            # happily linearly interpolate
            interpolated_lactate[previous_time:current_time] = interpolated_lactate.loc[previous_time:current_time].interpolate(method='linear')
        else:
            # forward, backward fill for max 3 hours
            minutes_elapsed = (current_time - previous_time)/np.timedelta64(1, 'm')
            if minutes_elapsed < 60*6:        # 6 hours
                # interpolate
                print('WARNING: under 6 hours between measurements - interpolating!')
                interpolated_lactate[previous_time:current_time] = interpolated_lactate.loc[previous_time:current_time].interpolate(method='linear')
            else:
                # more than 6 hours has elapsed - fill for max 3 hours in each direction
                temp_fill_limit = int((3*60)/grid_size_in_minutes)     # max # entries on the time-grid corresponding to 3 hours
                interpolated_lactate[previous_time:current_time] = interpolated_lactate.loc[previous_time:current_time].fillna(method='ffill', limit=temp_fill_limit).fillna(method='bfill', limit=temp_fill_limit)
    # now deal with the first and last regions
    # first
    first_lactate = measured_lactate_times[1]
    if p_df.loc[first_lactate, 'lactate_above_threshold']:
        temp_limit = int((3*60)/grid_size_in_minutes)     # max # entries on the time-grid corresponding to 3 hours
    else:
        temp_limit = None
    interpolated_lactate[p_df.index[0]:first_lactate] = interpolated_lactate[p_df.index[0]:first_lactate].fillna(method='bfill', limit=temp_limit)
    # last
    last_lactate = measured_lactate_times[-1]
    if p_df.loc[last_lactate, 'lactate_above_threshold']:
        temp_limit = int((3*60)/grid_size_in_minutes)     # max # entries on the time-grid corresponding to 3 hours
    else:
        temp_limit = None
    interpolated_lactate[last_lactate:p_df.index[-1]] = interpolated_lactate[last_lactate:p_df.index[-1]].fillna(method='ffill', limit=temp_limit)
    return interpolated_lactate

def interpolate_patient_lactate(p_df, pid, grid_size_in_minutes):
    # identify where there are lactate values
    measured_lactate_times = p_df['lactate'].dropna().index
    # we run through this list (we assume the list is not very long)
    if len(measured_lactate_times) == 0:
        # do nothing
        print('WARNING: patient has no lactate measurements at all.')
        bad_patients.write(str(int(pid)) + ',missing_lactate\n')
        p_df['interpolated_lactate'] = np.nan
        p_df['interpolated_lactate_above_threshold'] = np.nan
        return p_df
    if len(measured_lactate_times) == 1:
        interpolated_lactate = process_patient_single_lactate(p_df, measured_lactate_times, grid_size_in_minutes)
        p_df['interpolated_lactate'] = interpolated_lactate
    else:
        interpolated_lactate = process_patient_multiple_lactate(p_df, measured_lactate_times, grid_size_in_minutes)
    p_df['interpolated_lactate'] = interpolated_lactate
    p_df['interpolated_lactate_above_threshold'] = interpolated_lactate >= 2
    p_df.loc[p_df['interpolated_lactate'].isnull(), 'interpolated_lactate_above_threshold'] = np.nan
    return True
