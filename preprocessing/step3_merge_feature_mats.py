#!/usr/bin/env python
import pandas as pd
import numpy as np
import os

import sys
sys.path.append('../utils')
import preproc_utils

def get_feature_mat(tbl_name, patient_id):
    input_path = os.path.join(preproc_utils.datapath, '2_pivoted', version, tbl_name)
    filename = [f for f in os.listdir(input_path) if '_%d_'%index_chunk in f][0]
    df = pd.read_hdf(os.path.join(input_path, filename), where='PatientID=%d'%patient_id, mode='r')
        
    df.set_index(['PatientID', 'Datetime'], inplace=True)
    return df

def merge_single_patient_feature_mats(patient_id):
    df_mat = get_feature_mat('monvals', patient_id)
    df_mat = [df_mat] + [get_feature_mat(tbl_name, patient_id) for tbl_name in ['dervals', 'observrec', 'pharmarec', 'labres']]
    df_merged = pd.concat(df_mat, join='outer', axis=1)

    datetime_list = [set([x[1] for x in df_tmp.index]) for df_tmp in df_mat]
    datetime_set = set()
    for x in datetime_list:
        datetime_set |= x
    try:        
        assert(len(df_merged) == len(datetime_set))
    except AssertionError:
        print('Number of records in merged table does not match with the size of datetime union set of all tables.')
        import ipdb
        ipdb.set_trace()

    df_merged.reset_index(inplace=True)
    df_merged.sort_values(by=['Datetime'], inplace=True)

    try:
        assert(len(df_merged) == len(df_merged.Datetime.unique()))
    except AssertionError:
        print('There are duplicated records with the same datetime')
        import ipdb
        ipdb.set_trace()

    columns = np.sort(df_merged.columns[2:]).tolist()
    return df_merged[['PatientID', 'Datetime'] + columns]

def main():
    output_path = os.path.join(preproc_utils.datapath, '3_merged', version)
    if output_to_disk and not os.path.exists(output_path):
        os.makedirs(output_path)

    pid_list = np.array(pid_chunkfile_index[pid_chunkfile_index.ChunkfileIndex==index_chunk].index)
    output_path = os.path.join(output_path, 'fmat_%d_%d--%d.h5'%(index_chunk, np.min(pid_list), np.max(pid_list)))
    if output_to_disk and os.path.exists(output_path):
        print('Already os.path.exists: %s.'%output_path)
        print('Please delete it manually if you want to reproduce a new one.')
        return -1

    vois = [np.unique(preproc_utils.voi_id_name_mapping(tmp, version=version).index) for tmp in ['monvals', 'dervals', 'observrec', 'labres', 'pharmarec']]
    
    num_pid = len(pid_list)
    df_idx_start = 0
    for i in range(num_pid):
        df = merge_single_patient_feature_mats(pid_list[i])

        if len(df) == 0:
            print('Patient', pid_list[i], 'does not have data from any table.')
            continue

        if output_to_disk:
            df.set_index(np.arange(df_idx_start, df_idx_start+len(df)), inplace=True, drop=True)
            df['PatientID'] = df.PatientID.astype(int)
            df.to_hdf(output_path, 'fmat', append=True, complevel=5, 
                      complib='blosc:lz4', data_columns=['PatientID'], format='table')
            df_idx_start += len(df)

        if (i+1)%50 == 0:
            print('Patient %d / %d'%(i+1, num_pid))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-version')
    parser.add_argument('--index_chunk', type=int, default=None)
    parser.add_argument('--output_to_disk', action='store_true')
    args = parser.parse_args()
    version = args.version
    index_chunk = args.index_chunk
    output_to_disk = args.output_to_disk
    pid_chunkfile_index = preproc_utils.get_chunking_info(version=version)
    main()
