#!/usr/bin/env ipython
# author: stephanie hyland
# based on pre-gridded data (containing all the relevant endpoint variables), identify endpoints!

# we can just load a batch of patients at once

import numpy as np
import pandas as pd
import ipdb

def check_window_for_endpoint(df_segment, MAP_ID, window_size_in_minutes, grid_size_in_minutes, interp=True):
    """
    Check if this window constitutes an endpoint, and label it.

    "interp" refers to whether we use the interpolated or the non-interpolated lactate to define the lactate criterion
    """
    assert df_segment.shape[0] == window_size_in_minutes/grid_size_in_minutes
    assert (df_segment.index[-1] - df_segment.index[0])/np.timedelta64(1, 's')/60 == (window_size_in_minutes - grid_size_in_minutes)       # implies 45 minutes covered using default values
    # check if there are any MAP measurements
    if df_segment[MAP_ID].isnull().mean()[0] == 1:
        print('[interp:', interp, '] WARNING: no MAP measurements in this segment')
        # there should not be any drugs ... right??
        try:
            assert df_segment.loc[:, ['level1_drugs_present', 'level2_drugs_present', 'level3_drugs_present']].any(axis=1).sum() == 0
        except AssertionError:
            print('WARNING: drugs found, but no MAP')
        return 'unknown'
    
    # check the MAP/drugs requirement
    MAP_or_drugs_segment = df_segment.loc[:, ['MAP_below_threshold', 'level1_drugs_present', 'level2_drugs_present', 'level3_drugs_present']]
    # get rows with any missing values
    MAP_or_drugs_missing = MAP_or_drugs_segment.isnull().any(axis=1)
    # get rows with any positive values
    MAP_or_drugs = MAP_or_drugs_segment.any(axis=1)
    MAP_or_drugs = df_segment.loc[:, ['MAP_below_threshold', 'level1_drugs_present', 'level2_drugs_present', 'level3_drugs_present']].any(axis=1)
    # rows _without_ positive values, but _with_ missing values, are set to NaN
    MAP_or_drugs[(~MAP_or_drugs) & (MAP_or_drugs_missing)] = np.nan
    
    MAP_or_drugs_fraction = MAP_or_drugs.mean()
    
    if MAP_or_drugs_fraction >= 2.0/3:
        # check most extreme status to formulate label
        drug_levels = df_segment.loc[:, ['level1_drugs_present', 'level2_drugs_present', 'level3_drugs_present']].sum(axis=0)
        if drug_levels['level3_drugs_present'] > 0:
            level = '3'
        elif drug_levels['level2_drugs_present'] > 0:
            level = '2'
        else:
            level = '1'
    else:
        # we've already failed to have an event
        return 'event 0'

    if interp:
       lactate_var = 'interpolated_lactate'
       lactate_threshold_var ='interpolated_lactate_above_threshold'
    else:
        lactate_var = 'lactate'
        lactate_threshold_var ='lactate_above_threshold'

    if not lactate_var in df_segment.columns:
        ipdb.set_trace()
    try:
        assert df_segment[lactate_var].isnull().sum() == df_segment[lactate_threshold_var].isnull().sum()
    except AssertionError:
        pdb.set_trace()
    if df_segment[lactate_threshold_var].isnull().mean() == 1:
        print('[interp:', interp, '] warning: no lactate measurements in this segment')
        # no lactate
        certainty = 'maybe'
    else:
        lactate_fraction = df_segment[lactate_threshold_var].mean()
        if lactate_fraction >= 2.0/3: 
            certainty = 'event'
        else:
            certainty = 'probably not'
    return certainty + ' ' + level

def find_endpoints(df, MAP_ID, window_size_in_minutes, grid_size_in_minutes):
    """
    For a single patient, find regions where endpoints are true.
    """
    # logic:
    # rolling 45 minute window (centered):
        # if MAP or drugs is true >=2/3% of the window:
            # if yes, check most extreme status, that's the label
            # check lactate status, record 'maybe/probably not/yes' depending on availability and value
    df['endpoint_status'] = 'unknown'
    # window is 45 minutes = 9 measurements (once every 5)
    # we will not try to be quick for now
    half_window_size = int(0.5*(window_size_in_minutes/grid_size_in_minutes - 1))
    for window_centre in range(half_window_size, df.shape[0]-half_window_size):
        df_segment = df.iloc[window_centre - half_window_size:window_centre + (half_window_size + 1), :]
        endpoint_status = check_window_for_endpoint(df_segment, MAP_ID, window_size_in_minutes, grid_size_in_minutes, interp=True)
        df.loc[df.index[window_centre], 'endpoint_status'] = endpoint_status
        endpoint_status_nointerp = check_window_for_endpoint(df_segment, MAP_ID, window_size_in_minutes, grid_size_in_minutes, interp=False)
        df.loc[df.index[window_centre], 'endpoint_status_nointerp'] = endpoint_status_nointerp

    # now we convert it to a more reasonable encoding,
    # somewhat consistent with what was before
    df['event1'] = 0
    df['event2'] = 0
    df['event3'] = 0
    df['maybe_event1'] = 0
    df['maybe_event2'] = 0
    df['maybe_event3'] = 0
    df['probably_not_event1'] = 0
    df['probably_not_event2'] = 0
    df['probably_not_event3'] = 0

    df.loc[df['endpoint_status'] == 'unknown', ['event1', 'event2', 'event3', 'maybe_event1', 'maybe_event2', 'maybe_event3',
        'probably_not_event1', 'probably_not_event2', 'probably_not_event3']] = np.nan
    df.loc[df['endpoint_status'] == 'event 1', 'event1'] = 1
    df.loc[df['endpoint_status'] == 'event 2', 'event2'] = 1
    df.loc[df['endpoint_status'] == 'event 3', 'event3'] = 1
    df.loc[df['endpoint_status'] == 'maybe 1', 'maybe_event1'] = 1
    df.loc[df['endpoint_status'] == 'maybe 2', 'maybe_event2'] = 1
    df.loc[df['endpoint_status'] == 'maybe 3', 'maybe_event3'] = 1
    df.loc[df['endpoint_status'] == 'probably not 1', 'probably_not_event1'] = 1
    df.loc[df['endpoint_status'] == 'probably not 2', 'probably_not_event2'] = 1
    df.loc[df['endpoint_status'] == 'probably not 3', 'probably_not_event3'] = 1
    # there should be nothing where event 0 is true
    assert not df.loc[df['endpoint_status'] == 'event 0', ['event1', 'event2', 'event3', 'maybe_event1', 'maybe_event2', 'maybe_event3',
        'probably_not_event1', 'probably_not_event2', 'probably_not_event3']].sum(axis=1).any()
    # there should be exactly 1 thing where event is known, and not event 0
    try:
        assert (df.loc[~((df['endpoint_status'] == 'unknown')|(df['endpoint_status'] == 'event 0')), ['event1', 'event2', 'event3', 
            'maybe_event1', 'maybe_event2', 'maybe_event3', 
            'probably_not_event1', 'probably_not_event2', 'probably_not_event3']].sum(axis=1) == 1).all()
    except AssertionError:
        pdb.set_trace()
    
    return True
