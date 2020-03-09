#!/usr/bin/env ipython
# author: stephanie hyland
# chunk the merged file and also build the chunkfile

import numpy as np
import pandas as pd
import ipdb

import mimic_paths

n_splits = 50

def build_chunk_file(version='180817'):
    base_merged_dir = mimic_paths.merged_dir + version + '/reduced/'
    df = pd.read_hdf(base_merged_dir + 'merged_clean.h5', columns=['PatientID'])
    pids = sorted(df['PatientID'].unique())
    chunks = np.array_split(pids, n_splits)

    chunkfile = open(mimic_paths.chunks_file + '.' + version, 'w')
    chunkfile.write('PatientID,ChunkfileIndex\n')
    for chunk_idx, chunk in enumerate(chunks):
        for pid in chunk:
            chunkfile.write(str(int(pid)) + ',' + str(chunk_idx) + '\n')
    chunkfile.close()
    return True
   
def chunk_up_merged_file(version='180817'):
    """
    assumes the chunkfile already exists
    """
    base_merged_dir = mimic_paths.merged_dir + version + '/reduced/'
    df = pd.read_hdf(base_merged_dir + '/merged_clean.h5')
    chunks = pd.read_csv(mimic_paths.chunks_file + '.' + version)
    for chunk_idx in chunks['ChunkfileIndex'].unique():
        pids = set(chunks.loc[chunks['ChunkfileIndex'] == chunk_idx, 'PatientID'])
        idx_start = min(pids)
        idx_stop = max(pids)
        chunk_name = 'reduced_fmat_' + str(chunk_idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'
        chunk_df = df.loc[df['PatientID'].isin(pids), :]
        print(chunk_name, ';', chunk_df.shape)
        chunk_df.to_hdf(base_merged_dir + chunk_name, key='merged_clean', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
