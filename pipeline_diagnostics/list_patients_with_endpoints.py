#!/usr/bin/env
# author: stephanie hyland
# make a list of PIDS with non-empty, non-all-unknown endpoint data

import pandas as pd
import glob
import ipdb

import paths

def compile_list():
    pids_with_endpoints = open(paths.derived_dir + 'patients_with_endpoints_v6b.txt', 'w')
    pids_without_endpoints = open(paths.derived_dir + 'patients_without_endpoints_v6b.txt', 'w')
    for f in glob.glob(paths.endpoints_dir_reduced + '*.h5'):
        df = pd.read_hdf(f, columns=['PatientID', 'endpoint_status'])
        df.drop_duplicates(inplace=True)
        has_endpoints = df.groupby('PatientID').apply(lambda x: (sum(x['endpoint_status'] != 'unknown') > 0) & (x.shape[0] > 0))
        print(sum(~has_endpoints), 'patients are missing endpoints in', f)
        for pid in has_endpoints.index[has_endpoints]:
            pids_with_endpoints.write(str(pid) + ',' + f + '\n')
        for pid in has_endpoints.index[~has_endpoints]:
            pids_without_endpoints.write(str(pid) + ',' + f + '\n')
    pids_with_endpoints.close()
    pids_without_endpoints.close()

def check_list():
    """
    slow but w/e
    """
    pids_with_endpoints = open(paths.derived_dir + 'patients_with_endpoints_v6b.txt', 'r')
    pids_without_endpoints = open(paths.derived_dir + 'patients_without_endpoints_v6b.txt', 'r')
    for line in pids_with_endpoints:
        pid, f = line.strip('\n').split(',')
        pdf = pd.read_hdf(f, where='PatientID == ' + str(pid), columns=['PatientID', 'endpoint_status'])
        assert pdf.shape[0] > 0
        if len(pdf['endpoint_status'].unique()) < 2:
            assert not 'unknown' in pdf['endpoint_status'].unique()
    for line in pids_with_endpoints:
        pid, f = line.strip('\n').split(',')
        pdf = pd.read_hdf(f, where='PatientID == ' + str(pid), columns=['PatientID', 'endpoint_status'])
        if not pdf.shape[0] == 0:
            assert len(pdf['endpoint_status'].unique()) == 1
            assert 'unknown' in pdf['endpoint_status'].unique()
    print('all good')
