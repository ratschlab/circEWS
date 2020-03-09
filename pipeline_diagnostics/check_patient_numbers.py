#!/usr/bin/env ipython
# author: Stephanie Hyland
# purpose: check how many patients we have at different stages of the pipeline

import pandas as pd
import numpy as np
import gc
import re

import glob
import ipdb

import paths

PATTERN = re.compile(''';(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''')
#PATTERN = re.compile(r'''((?:[^;"']|"[^"]*"|'[^']*')+)''')

# what we will do is create a table of every patient ID as rows
# and whether or not they're present at different pipeline stages as columns
# then we can inspect it

version = 'v6b'

def get_csv_pids():
    """
    """
    pids_path = paths.derived_dir + version +'_csv_pids.txt'
    try:
        pids = set(map(int, np.loadtxt(pids_path)))
        print('Loaded from file')
        return pids
    except OSError:
        print('No file found, calculating...')
    pids = set()
    for csv in glob.glob(paths.csvs_0 + '*.csv'):
        print(csv)
        csv_file = open(csv, 'rb')
        header = csv_file.readline()
        csv_file.close()
        try:
            # assume separator is tab
            header_sl = header.decode('latin-1').strip('\n').split('\t')
            separator = '\t'
            assert len(header_sl) < len(header)
            assert len(header_sl) > 1
        except AssertionError:
            # otherwise it's a semicolon for some reason
            header_sl = header.decode('latin-1').strip('\n').split(';')
            separator = ';'
            assert len(header_sl) < len(header)
            assert len(header_sl) > 1
        try:
            pid_idx = header_sl.index('PatientID')
        except ValueError:
            continue
            # patientID isn't in this file
        # now, we know pid is in there, the separator, and which index we need
        print('processing', csv)

        # new bit
        df_chunks = pd.read_csv(csv, sep=separator, chunksize=5000, encoding='latin-1')
        for chunk in df_chunks:
            pids.update(chunk['PatientID'].unique())
            del chunk
            gc.collect()
        print(len(pids))
            
#        for i, line in enumerate(csv_file):
#            try:
#                #line_sl = line.decode('latin-1').strip('\n').split(separator)
#                line_sl = PATTERN.split(line.decode('latin-1').strip('\n'))
#            except UnicodeDecodeError:
#                ipdb.set_trace()
#            try:
#                if i % 500000 == 0:
#                    print(i, len(pids))
#                if len(line_sl) > pid_idx:
#                    try:
#                        pids.add(int(line_sl[pid_idx]))
#                    except ValueError:
#                        ipdb.set_trace()
#                else:
#                    # just skip this weirdo line
#                    pass
#            except IndexError:
#                ipdb.set_trace()
    np.savetxt(pids_path, list(pids))
    return pids

def get_hdf_pids(folder, label='', subfolders=False):
    pids_path = paths.derived_dir + version + '_' + label + '_pids.txt'
    try:
        pids = set(map(int, np.loadtxt(pids_path)))
        print('Loaded from file')
        return pids
    except OSError:
        print('No file found, calculating...')
    pids = set()
    for hdf5 in glob.glob(folder + '**/'*subfolders + '*.h5', recursive=subfolders):
        print(hdf5)
        try:
            df_chunks = pd.read_hdf(hdf5, columns=['PatientID'], chunksize=5000)
            for chunk in df_chunks:
                pids.update(chunk['PatientID'].unique())
                del chunk
                gc.collect()
            print(len(pids))
        except:
            print('No PatientID column found, skipping')
    np.savetxt(pids_path, list(pids))
    return pids

def get_all_pids():
    """
    """
    pids_csv = get_csv_pids()
    pids_hdf_consent = get_hdf_pids(paths.hdf_consent_dir, 'hdf_consent')
    pids_hdf_clean = get_hdf_pids(paths.hdf_clean_dir, 'hdf_clean', subfolders=True)
    pids_pivoted = get_hdf_pids(paths.pivoted_dir, 'pivoted', subfolders=True)
    pids_merged = get_hdf_pids(paths.merged_dir, 'merged')
    pids_merged_reduced = get_hdf_pids(paths.merged_dir_reduced, 'merged_reduced')
    pids_endpoints = get_hdf_pids(paths.endpoints_dir, 'endpoints')
    pids_imputed = get_hdf_pids(paths.imputed_dir, 'imputed', subfolders=True)
    pids_labels = get_hdf_pids(paths.labels_dir, 'labels', subfolders=True)
    # turn these all into a dataframe
    all_pids = set.union(*[pids_csv, pids_hdf_consent, pids_hdf_clean, pids_pivoted,
        pids_merged, pids_merged_reduced, pids_endpoints, pids_imputed, pids_labels])
    df = pd.DataFrame({'PatientID': list(all_pids)})
    df['csv'] = False
    df['hdf_consent'] = False
    df['hdf_clean'] = False
    df['pivoted'] = False
    df['merged'] = False
    df['merged_reduced'] = False
    df['endpoints'] = False
    df['imputed'] = False
    df['labels'] = False
    df.set_index('PatientID', inplace=True)
    df.loc[pids_csv, 'csv'] = True
    df.loc[pids_hdf_consent, 'hdf_consent'] = True
    df.loc[pids_hdf_clean, 'hdf_clean'] = True
    df.loc[pids_pivoted, 'pivoted'] = True
    df.loc[pids_merged, 'merged'] = True
    df.loc[pids_merged_reduced, 'merged_reduced'] = True
    df.loc[pids_endpoints, 'endpoints'] = True
    df.loc[pids_imputed, 'imputed'] = True
    df.loc[pids_labels, 'labels'] = True
    df.to_csv(paths.derived_dir + version + '_pids_tracking.csv', index=True, header=True)

def analyse_pids():
    """
    """
    df = pd.read_csv(paths.derived_dir + version + '_pids_tracking.csv')
    print(df.mean(axis=0))
    print(df.sum(axis=0))
    # need to make sure patients don't magically reappear
    for colnum in range(1, len(df.columns) - 1):
        assert ((df.iloc[:, colnum].astype(int) - df.iloc[:, colnum + 1].astype(int)) >= 0).mean() == 1
    return df
