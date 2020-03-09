#!/usr/bin/env ipython
# author: stephanie hyland
# check statistics and so on

import pandas as pd
import numpy as np
import ipdb
import glob

import mimic_paths

from extract_data_from_mimic import load_reference_tables, collect_pre_merged, prep_table

def get_number_of_measurements():
    d_ref = load_reference_tables()
    itemids = d_ref['ITEMID'].unique()
    var_counts = dict()
    for itemid in itemids:
        var_counts[itemid] = 0
    for table in d_ref['table'].unique():
        print(table)
        try:
            var_idx, f = prep_table(table)
        except:
            continue
        header = f.readline()
        for i, line in enumerate(f):
            if i % 500000 == 0:
                print(i)
                print(var_counts)
            sl = line.strip('\n').split(',')
            try:
                var_counts[int(sl[var_idx])] += 1
            except KeyError:
                continue
    return var_counts

def get_sampling_interval(df, patient_identifier, time_identifier):
    # get mean sampling interval (in hours) over all patients
    intervs = df.groupby(patient_identifier).apply(lambda x:  ((pd.to_datetime(x.sort_values(time_identifier)[time_identifier]) - pd.to_datetime(x.sort_values(time_identifier)[time_identifier].shift()))/np.timedelta64(1, 'h')).dropna().mean())
    # take the median average sampling interval
    interv = intervs.median()
    return interv

def build_var_stats(merged=False):
    d_ref = load_reference_tables()
    if merged:
        df = pd.read_hdf(mimic_paths.merged_dir + 'merged.h5')
        patient_identifier = 'PatientID'
        time_identifier = 'Datetime'
        ipdb.set_trace()
    else:
        df = collect_pre_merged()
        patient_identifier = 'ICUSTAY_ID'
        time_identifier = 'CHARTTIME'

    mid_list = []
    var_list = []
    n_list = []
    n_pat = []
    per_pat = []
    interval = []

    for var in d_ref['mID'].unique():
        varname = d_ref.loc[d_ref['mID'] == var, 'varname'].iloc[0]
        if merged:
            df_v = df.loc[:, [patient_identifier, var, time_identifier]].dropna()
        else:
            df_v = df.loc[df['mID'] == var, :]
        df_v.dropna(inplace=True, how='all')
        n_measurements = df_v.shape[0]
        n_patients_with_measurements = len(df_v[patient_identifier].unique())
        measurements_per_patient = (df_v.groupby(patient_identifier).apply(lambda x: x.shape[0])).mean()
        sampling_interval = get_sampling_interval(df_v, patient_identifier, time_identifier)
        print(var, varname, n_measurements, n_patients_with_measurements, measurements_per_patient, sampling_interval)
        mid_list.append(var)
        var_list.append(varname)
        n_list.append(n_measurements)
        n_pat.append(n_patients_with_measurements)
        per_pat.append(measurements_per_patient)
        interval.append(sampling_interval)

    stats = pd.DataFrame({'mID': mid_list, 'varname': var_list,
        'n_measurements': n_list, 'n_patients_with_measurements': n_pat,
        'ave_measurements_per_patient': per_pat,
        'median_mean_sampling_interval': interval})
    stats.to_csv(mimic_paths.validation_dir + merged*'merged_' + 'stats.csv', index=False)
