#!/usr/bin/env ipython
# author: stephanie hyland
# get statistics on the different temporal splits
# number of patients
# number of training/test/validation examples
# start and end time

import pandas as pd
import glob
import ipdb

import paths

splits = pd.read_csv(paths.splits_file, sep='\t')
splits.set_index('pid', inplace=True)

def get_num_patients(split):
    """
    (special case of next function)
    """
    directory = paths.mlinput_dir + split + '/AllLabels_0.0_8.0/X'
    n = {'train': 0, 'val': 0, 'test': 0}
    for hdf_path in glob.glob(directory + '/batch*.h5'):
        print(hdf_path)
        store = pd.HDFStore(hdf_path)
        keys = store.keys()
        for key in keys:
            pid = int(key[1:])
            try:
                exp_split = splits.loc[pid, split]
                n[exp_split] += 1
            except:
                print('WARNING: patient', pid, 'is not in the splits file...?')
        store.close()
    return n

def get_data_size(split):
    directory = paths.mlinput_dir + split + '/AllLabels_0.0_8.0/X'
    n = {'train': 0, 'val': 0, 'test': 0}
    n_pat = {'train': 0, 'val': 0, 'test': 0}
    for hdf_path in glob.glob(directory + '/batch*.h5'):
        print(hdf_path)
        store = pd.HDFStore(hdf_path)
        keys = store.keys()
        for key in keys:
            nrows = store.get_storer(key).shape[0]
            pid = int(key[1:])
            # check where the patient is
            try:
                exp_split = splits.loc[pid, split]
                n[exp_split] += nrows
                n_pat[exp_split] += 1
            except:
                #                ipdb.set_trace()
                print('WARNING: patient', pid, 'is not in the splits file...?')
        store.close()
    return n, n_pat

def get_start_end(split):
    directory = paths.mlinput_dir + split + '/AllLabels_0.0_8.0/X'
    for hdf_path in glob.glob(directory + '/batch*.h5'):
        print(hdf_path)
        store = pd.HDFStore(hdf_path)
        keys = store.keys()
        store.close()
        for key in keys:
            df = pd.read_hdf(hdf_path, key)
            pid_min_time = df['AbsDatetime'].min()
            try:
                if pid_min_time < min_time:
                    min_time = pid_min_time
                    print('new min time:', min_time)
            except NameError:
                min_time = pid_min_time
            pid_max_time = df['AbsDatetime'].max()
            try:
                if pid_max_time > max_time:
                    max_time = pid_max_time
                    print('new max time:', max_time)
            except NameError:
                max_time = pid_max_time
    return min_time, max_time


def get_all(split):
    directory = paths.mlinput_dir + split + '/AllLabels_0.0_8.0/X'
    n = {'train': 0, 'val': 0, 'test': 0}
    n_pat = {'train': 0, 'val': 0, 'test': 0}
    for hdf_path in glob.glob(directory + '/batch*.h5'):
        if 'repacked' in hdf_path:
            # not sure what this is but it breaks my script
            continue
        print(hdf_path)
        store = pd.HDFStore(hdf_path)
        keys = store.keys()
        store.close()
        for key in keys:
            df = pd.read_hdf(hdf_path, key)
            nrows = df.shape[0]
            pid = int(key[1:])
            # check where the patient is
            try:
                exp_split = splits.loc[pid, split]
                n[exp_split] += nrows
                n_pat[exp_split] += 1
            except:
                #                ipdb.set_trace()
                print('WARNING: patient', pid, 'is not in the splits file...?')
            pid_min_time = df['AbsDatetime'].min()
            try:
                if pid_min_time < min_time:
                    min_time = pid_min_time
                    print('new min time:', min_time)
            except NameError:
                min_time = pid_min_time
            pid_max_time = df['AbsDatetime'].max()
            try:
                if pid_max_time > max_time:
                    max_time = pid_max_time
                    print('new max time:', max_time)
            except NameError:
                max_time = pid_max_time
    return min_time, max_time, n, n_pat

def get_split_test_start(split):
    print(splits.loc[splits[split] == 'test', :].adm_time.min())
    return True
