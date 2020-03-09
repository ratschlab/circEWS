#!/usr/bin/env ipython
# author: stephanie hyland

# get stats about endpoints and so on

# to report: for each patient
# -- # events of each level (unknown, 0, 1, 2, 3, maybe_1, etc.)
# -- average duration of each level
# -- time to first event of each type
# -- fraction of time spent in each level
# -- # deteriorations
# -- total LOS

# so I can do is create a long-form csv with columns:
#       pid, variable, value
# and "variable" takes the values of the various things we're reporting

import ipdb
import pandas as pd
import numpy as np
from multiprocessing import Pool
import itertools
import glob
import re

from get_statistics import get_relevant_patients

import paths

mimic = False
if mimic:
    endpoints_dir = paths.base_dir + '/external_validation/endpoints/181103/reduced/'
else:
    endpoints_dir = paths.endpoint_dir_reduced

events = ['event_0', 'event_1', 'event_2', 'event_3', 'maybe_1', 'maybe_2', 'maybe_3',
        'probably_not_1', 'probably_not_2', 'probably_not_3', 'unknown']
transitions = [x[0] + ':' + x[1] for x in itertools.product(events, repeat=2)]

# get the chunks
chunks = glob.glob(endpoints_dir + '/reduced_endpoints*.h5')
chunks = list(map(lambda x: x.split('/')[-1][:-3], chunks))
print(chunks)

def process_chunk(chunk_name):
    print('processing chunk', chunk_name)
    df = pd.read_hdf(endpoints_dir + chunk_name + '.h5')
    df.set_index('Datetime', inplace=True)
    patients = df['PatientID'].drop_duplicates()
    summary_file = open(endpoints_dir + chunk_name + '.summary.csv', 'w')
    summary_file.write('pid,event,variable,value\n')
    transitions_file = open(endpoints_dir + chunk_name + '.transitions.csv', 'w')
    transitions_file.write('pid,' + ','.join(transitions) + '\n')
    for (i, patient) in enumerate(patients):
        print(i, '\t patient', int(patient))
        p_df = df.loc[df['PatientID'] == patient, :]
        p_df.sort_index(inplace=True)
        summarise_patient(p_df, summary_file, transitions_file)
    summary_file.close()
    transitions_file.close()
    return True

def get_data_from_event_type(event_series):
    """
    """
    # for the purpose of this function, we assume that a NaN is a 0
    event_series = event_series.fillna(0)
    # where we go from 0 to 1 or NaN to 1
    starts = (event_series - event_series.shift() == 1)
    if event_series.iloc[0] == 1.0:
        # patient starts in an event
        starts.iloc[0] = True
    # events is just the number of starts
    number_of_events = starts.sum()
    if number_of_events == 0:
        average_duration = 'NA'
        range_of_durations = 'NA'
        time_to_first_event = 'NA'
        time_in_event = 0
    else:
        # now to get durations, we need ends too
        cut = False
        ends = (event_series - event_series.shift() == -1)
        start_times = event_series.loc[starts].index.values
        end_times = event_series.loc[ends].index.values
        if len(end_times) < len(start_times):
            assert event_series.iloc[-1] == 1
            print('looks like patient never left the final endpoint, adding synthetic after last positive case')
            # take the end as the last time it was 1, + 5
            synthetic_end = np.datetime64(event_series[event_series == 1].index[-1] + pd.Timedelta(5, 'm'))
            end_times = np.append(end_times, synthetic_end)
            # we have to drop one of the durations from consideration beacuse it's fake
            cut = True
        elif len(start_times) < len(end_times):
            assert event_series.iloc[0] == 1
            print('looks like the patient started in an endpoint')
            synthetic_start = np.datetime64(event_series.index[0])
            start_times = np.append(synthetic_start, start_times)
        assert len(end_times) == len(start_times)
        try:
            durations = ((end_times - start_times)/np.timedelta64(1, 'm'))
            if cut and durations[-1] < 10:
                # only exclude probably-fake durations
                durations = durations[:-1]
        except:
            ipdb.set_trace()
        if len(durations) > 0:
            try:
                assert np.all(durations > 0)
            except AssertionError:
                ipdb.set_trace()
            average_duration = np.mean(durations)
            range_of_durations = np.max(durations) - np.min(durations)
        else:
            average_duration = 'NA'
            range_of_durations = 'NA'
        # time to first event is just time until first start_time
        time_to_first_event = (start_times[0] - np.datetime64(event_series.index[0]))/np.timedelta64(1, 'm')
        time_in_event = np.sum(durations)
    return number_of_events, average_duration, range_of_durations, time_to_first_event, time_in_event

def get_transitions(endpoint_status, transitions_file, pid):
    """
    For all the types of event, count all the types of transitions.
    (but how are we going to save this...?)
    (in a very wide csv)
    """
    counts = dict(zip(transitions, [0]*len(transitions)))
    # just run through the endpoint_status recording what happens
    nrows = endpoint_status.shape[0]
    prev_status = endpoint_status.iloc[0]
    for idx in range(1, nrows):
        current_status = endpoint_status.iloc[idx]
        transition = re.sub(' ', '_', prev_status + ':' + current_status)
        try:
            assert transition in transitions
        except AssertionError:
            pdb.set_trace()
        counts[transition] += 1
        prev_status = current_status
    # save this to the file
    # TODO save to file
    transitions_file.write(pid + ',')
    for transition in transitions:
        count = counts[transition]
        transitions_file.write(str(count) + ',')
    transitions_file.write('\n')
    # now specifically look at deteriorations to return
    deteriorations = 0
    for transition in ['event_0:event_1', 'event_0:event_2', 'event_0:event_3',
            'event_1:event_2', 'event_1:event_3', 'event_2:event_3']:
        if transition in counts:
            deteriorations += counts[transition]
    return deteriorations

def summarise_unknown(pid, p_df, summary_file):
    # unknown event has to be treated differently because it's not a proper column... i should just make it one
    in_unknown = p_df.loc[p_df['endpoint_status'] == 'unknown', :]
    if in_unknown.empty:
        fraction_in_unknown = 0
        first_unknown = 'NA'
    else:
        fraction_in_unknown = in_unknown.shape[0]/p_df.shape[0]
        first_unknown = (in_unknown.index[0] - p_df.index[0])/np.timedelta64(1, 'm')
    summary_file.write(pid + ',unknown,fraction_in_event,' + str(fraction_in_unknown) + '\n')
    summary_file.write(pid + ',unknown,time_to_first_event,' + str(first_unknown) + '\n')
    return True

def summarise_patient(p_df, summary_file, transitions_file):
    """
    Get statistics about endpoints from a single patient.
    """
    pid = str(int(p_df['PatientID'].iloc[0]))
    LOS = (p_df.index[-1] - p_df.index[0])/np.timedelta64(1, 'm')
    # do it by event type
    for event in ['event1', 'event2', 'event3', 'maybe_event1', 'maybe_event2', 'maybe_event3', 'probably_not_event1', 'probably_not_event2', 'probably_not_event3']:
        number_of_events, average_duration, range_of_durations, time_to_first_event, time_in_event = get_data_from_event_type(p_df[event])
        # now write it to the file
        summary_file.write(pid + ',' + event + ',number_of_events,' + str(number_of_events) + '\n')
        summary_file.write(pid + ',' + event + ',average_duration,' + str(average_duration) + '\n')
        summary_file.write(pid + ',' + event + ',range_of_durations,' + str(range_of_durations) + '\n')
        summary_file.write(pid + ',' + event + ',time_to_first_event,' + str(time_to_first_event) + '\n')
        if LOS == 0:
            fraction_in_event = 'NA'
        else:
            fraction_in_event = time_in_event/LOS
        summary_file.write(pid + ',' + event + ',fraction_in_event,' + str(fraction_in_event) + '\n')
    summarise_unknown(pid, p_df, summary_file)
    # now get some event-independent statistics
    summary_file.write(pid + ',NA,length_of_stay,' + str(LOS) + '\n')
    number_of_deteriorations = get_transitions(p_df['endpoint_status'], transitions_file, pid)
    summary_file.write(pid + ',NA,number_of_deteriorations,' + str(number_of_deteriorations) + '\n')
    return True

def summarise_summary():
    """
    """
    df = pd.read_csv(endpoints_dir + '/summary/summary.csv')
    # assert we have all the patients
    if not mimic:
        pids = get_relevant_patients('split')
        print('Checking for statistics on patients...')
        try:
            assert set(pids) <= set(df['pid'].unique())
        except AssertionError:
            print('patient is missing from summary statistics...?')
            ipdb.set_trace()
        df = df.loc[df['pid'].isin(pids), :]
    
    df_event = df.loc[df['event'].isin({'event1', 'event2', 'event3'}), :]
    total_number_of_events = df_event.loc[df_event['variable'] == 'number_of_events', :].groupby('pid').apply(lambda x: x['value'].sum()).sum()
    ave_number_of_events = df_event.loc[df_event['variable'] == 'number_of_events', :].groupby('pid').apply(lambda x: x['value'].sum()).mean()
    patients_with_events = df_event.loc[df_event['variable'] == 'number_of_events', :].groupby('pid').apply(lambda x: x['value'].sum() > 0)
    df_with_events = df_event.loc[df_event['pid'].isin(patients_with_events.loc[patients_with_events].index), :]
    n_patients_with_events = patients_with_events.sum()
    
    ave_number_of_events = df_with_events.loc[df_with_events['variable'] == 'number_of_events', :].groupby('pid').apply(lambda x: x['value'].sum()).mean()
    ave_average_duration = df_with_events.loc[df_with_events['variable'] == 'average_duration', :].groupby('pid').apply(lambda x: x['value'].sum()).mean()
    ave_time_to_first_event = df_with_events.loc[df_with_events['variable'] == 'time_to_first_event', :].groupby('pid').apply(lambda x: x['value'].min()).mean()
    ave_fraction_in_event = df_with_events.loc[df_with_events['variable'] == 'fraction_in_event', :].groupby('pid').apply(lambda x: x['value'].sum()).mean()

    print('# of patients considered:', len(df['pid'].unique()))
    print('total # of events:', total_number_of_events)
    print('# patients with events:', n_patients_with_events)
    print('ave # of events:', ave_number_of_events)
    print('ave duration:', ave_average_duration)
    print('ave time to first:', ave_time_to_first_event)
    print('ave fraction in event:', ave_fraction_in_event)

