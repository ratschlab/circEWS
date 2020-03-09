#!/usr/bin/env ipython
# author: stephanie hyland
# collect statistics about the patient cohort

import ipdb
import pandas as pd
import glob

import paths

def get_split_patients():
    # get all patients across all splits
    pids = pd.read_csv(paths.splits_file, sep='\t')
    return pids['pid']

def get_relevant_patients(stage):
    if stage == 'split':
        return get_split_patients()
    else:
        pid_tracking = pd.read_csv(paths.derived_dir + 'pids_tracking.csv')
        pid_tracking.set_index('PatientID', inplace=True)
        pids = pid_tracking.index[pid_tracking[stage]]
    return pids

def generaldata(pids=None):
    if pids is None:
        pids = get_relevant_patients('split')
    df = pd.read_csv(paths.csvs_0 + 'expot-generaldata.csv', sep=';', encoding='latin-1')
    df.set_index('PatientID', inplace=True)
    for pid in pids:
        assert pid in df.index
    df_subset = df.loc[pids]
    n_patients = len(pids)
    print(n_patients)
    if not n_patients == 36098:
        print('ERROR: WRONG NUMBER OF PATIENTS:', n_patients)
        return False
    assert df_subset.shape[0] == n_patients
    for characteristic in ['Country', 'MaritalState', 'Sex']:
        print(characteristic) 
        print('#:', df_subset[characteristic].value_counts())
        print('%:', 100*df_subset[characteristic].value_counts()/n_patients)
    df_subset['AdmissionTime'] = pd.to_datetime(df_subset['AdmissionTime'])
    age = df_subset['AdmissionTime'].apply(lambda x: x.year) - df_subset['birthYear']
    #print('Age:', age.mean(), age.std())
    print('Age:', age.mean(), age.std(), age.median(), age.ptp(), age.min(), age.max())
    return df_subset

def static(pids=None):
    if pids is None:
        pids = get_relevant_patients('split')
    df = pd.read_hdf(paths.hdf_clean_dir + 'static.h5')
    df.set_index('PatientID', inplace=True)
    for pid in pids:
        assert pid in df.index
    df_subset = df.loc[pids]
    n_patients = len(pids) 
    if not n_patients == 36098:
        print('ERROR: WRONG NUMBER OF PATIENTS:', n_patients)
        return False
    assert df_subset.shape[0] == n_patients
    for column in ['Discharge', 'Surgical', 'Emergency']:
        print(df_subset[column].value_counts())
        print(100*df_subset[column].value_counts()/n_patients)
    return df_subset

def endpoint_derived_stats(pids=None):
    if pids is None:
        pids = get_relevant_patients('split')
    df = pd.read_csv(paths.endpoints_dir_reduced + '/summary/summary.csv')
    los = df.loc[df['variable'] == 'length_of_stay', ['pid', 'value']]
    los.set_index('pid', inplace=True)
    minutes_per_day = 60*24
    print('median LOS:', los.median()/minutes_per_day)
    print('mean LOS:', los.mean()/minutes_per_day)
    print('std LOS:', los.std()/minutes_per_day)
    print('range LOS:', los.min()/minutes_per_day, los.max()/minutes_per_day)
    ipdb.set_trace()

def get_apache_stats(pids=None):
    if pids is None:
        pids = get_relevant_patients('split')

    apache_info = pd.read_hdf(paths.apache_info_path, 'table')
    apache_info.rename(columns={'patientid': 'PatientID'}, inplace=True)
    apache_info.set_index('PatientID', inplace=True)
    
    pid_info = pd.DataFrame({'pid': pids}, index=pids)
    # left join the meta groups
    pid_info = pid_info.join(apache_info['meta_group'])
    
    total_patients = len(pids.unique())
    print('Total patients:', total_patients)
    if not total_patients == 36098:
        print('Total IS NOT 36098 and SHOULD BE?')
    assert pid_info.shape[0] == total_patients

    print(pid_info['meta_group'].value_counts())
    print(100*pid_info['meta_group'].value_counts()/total_patients)

    print('Unknown:')
    print('#:', pid_info['meta_group'].isna().sum())
    print('%:', 100*pid_info['meta_group'].isna().sum()/total_patients)
