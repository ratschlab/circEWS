#!/usr/bin/env python
import pandas as pd
import numpy as np

import os

import sys
sys.path.append('../utils')
import preproc_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tbl_name')
parser.add_argument('-version')
parser.add_argument('--index_chunk', type=int, default=None)
parser.add_argument('--output_to_disk', action='store_true')
args = parser.parse_args()
tbl_name = args.tbl_name
version = args.version
index_chunk = args.index_chunk
output_to_disk = args.output_to_disk

data_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version)
input_path = os.path.join(data_path, 'datetime_fixed' if tbl_name=='pharmarec' else 'oor_removed', tbl_name)
output_path = os.path.join(data_path, 'duplicates_removed', tbl_name)
if not os.path.exists(output_path) and output_to_disk:
    os.makedirs(output_path)
pid_chunkfile_index = preproc_utils.get_chunking_info(version=version)    
pid_list = np.array( pid_chunkfile_index.index[pid_chunkfile_index.ChunkfileIndex==index_chunk] )
output_path = os.path.join(output_path, '%s_%d_%d--%d.h5'%(tbl_name, index_chunk, np.min(pid_list), np.max(pid_list)))

voi = preproc_utils.voi_id_name_mapping(tbl_name, True, version=version)
if tbl_name == 'pharmarec':
    for n, pid in enumerate(pid_list):    
        filename = [f for f in os.listdir(input_path) if '%s_%d_'%(tbl_name, index_chunk) in f][0]
        df = pd.read_hdf(os.path.join(input_path, filename), where='PatientID=%d'%pid)
        
        cols_of_interest = ['PharmaID', 'InfusionID', 'Datetime', 'Status']
        if len(df.drop_duplicates(cols_of_interest)) != len(df):
            idx = set(df.index.tolist()) - set(df.drop_duplicates(cols_of_interest).index.tolist())
            idx = list(idx)
            for tmp in idx:
                if tmp not in df.index:
                    continue
                df_tmp  = df[np.logical_and(df.PharmaID==df.loc[tmp]['PharmaID'], 
                                            df.InfusionID==df.loc[tmp]['InfusionID'])]
                iloc = np.where(df_tmp.index==tmp)[0][0]
                if df.loc[tmp]['Status'] == 776:
                    index2drop = df_tmp.index[np.logical_and(df_tmp.Status == df.loc[tmp]['Status'], df_tmp.GivenDose==0)]                        
                    df.drop(index2drop, axis=0, inplace=True)
                elif df.loc[tmp]['Status'] == 780:
                    index_dups = df_tmp.index[np.logical_and(df_tmp.Status == df.loc[tmp]['Status'], 
                                                             df_tmp.Datetime==df.loc[tmp]['Datetime'])]
                    df.loc[index_dups[-1], 'GivenDose'] = np.sum(df.loc[index_dups, 'GivenDose'])
                    df.loc[index_dups[-1], 'CumulDose'] = np.sum(df.loc[index_dups, 'CumulDose'])
                    df.drop(index_dups[:-1], axis=0, inplace=True)
                else:
                    index_dups = df_tmp.index[np.logical_and(df_tmp.Status == df.loc[tmp]['Status'], 
                                                             df_tmp.Datetime==df.loc[tmp]['Datetime'])]
                    df.loc[index_dups[-1], 'GivenDose'] = np.mean(df.loc[index_dups, 'GivenDose'])
                    df.loc[index_dups[-1], 'CumulDose'] = np.mean(df.loc[index_dups, 'CumulDose'])
                    df.drop(index_dups[:-1], axis=0, inplace=True)

        cols_of_interest = ['PharmaID', 'InfusionID', 'Datetime']
        if len(df.drop_duplicates(cols_of_interest)) != len(df):           
            idx = set(df.index.tolist()) - set(df.drop_duplicates(cols_of_interest).index.tolist())
            idx = list(idx)
            for tmp in idx:
                if tmp not in df.index:
                    continue
                df_tmp  = df[np.logical_and(df.PharmaID==df.loc[tmp]['PharmaID'], 
                                            df.InfusionID==df.loc[tmp]['InfusionID'])]
                index_dups = df_tmp[df_tmp.Datetime==df.loc[tmp]['Datetime']].sort_values(['Datetime', 'EnterTime']).index
                df_dup_tmp = df_tmp.loc[index_dups]
                iloc = np.where(df_tmp.index==tmp)[0][0]
                if 544 in df_dup_tmp.Status.values:
                    if np.sum(df_dup_tmp[df_dup_tmp.Status==544].GivenDose==0) == np.sum(df_dup_tmp.Status==544):
                        print(df_tmp.loc[index_dups])
                        index2drop = df_dup_tmp.index[df_dup_tmp.Status==544]
                        df.drop(index2drop, axis=0, inplace=True)
                    else:
                        print(df_tmp.loc[index_dups])
                else:
                    print(df_tmp.loc[index_dups])
                    df.loc[index_dups[-1], 'GivenDose'] = np.sum(df.loc[index_dups, 'GivenDose'])
                    df.loc[index_dups[-1], 'CumulDose'] = np.sum(df.loc[index_dups, 'CumulDose'])
                    df.drop(index_dups[:-1], axis=0, inplace=True)
                     
 
        if output_to_disk:
            df.to_hdf(output_path, 'data', append=True, format='table', data_columns=True, complevel=5, complib='blosc:lz4')

        if (n+1)%50 == 0:
            print('%d / %d'%(n+1, len(pid_list)))
    
else:
    glb_std_path = os.path.join(preproc_utils.datapath, 'misc_derived', 'xinrui', version, 'global_mean_std')
    if tbl_name == 'monvals':
        mean_std = []
        for f in os.listdir(glb_std_path):
            if tbl_name not in f:
                continue
            mean_std_info_path = os.path.join(glb_std_path, f)
            mean_std.append(pd.read_csv(mean_std_info_path, sep='\t').set_index('VariableID'))
        mean_std = pd.concat(mean_std, axis=0)
    else:
        mean_std_info_path = os.path.join(glb_std_path, tbl_name+'.tsv')
        mean_std = pd.read_csv(mean_std_info_path, sep='\t').set_index('VariableID')
    
    if tbl_name != 'labres':
        mean_std = mean_std.merge(voi[['VariableUnit']], how='left', left_index=True, right_index=True)

    for n, pid in enumerate(pid_list):    
        filename = [f for f in os.listdir(input_path) if '%s_%d_'%(tbl_name, index_chunk) in f][0]
        df = pd.read_hdf(os.path.join(input_path, filename), where='PatientID=%d'%pid)        
        df = df[df.VariableID.isin(voi.index)].copy()

        df.drop_duplicates(['VariableID', 'Datetime', 'Value'], inplace=True)
        for vid in df.VariableID.unique():
            df_tmp = df[df.VariableID==vid]
            dt_cnt = df_tmp.Datetime.value_counts()
            if dt_cnt.max() > 1:
                dt_dup = dt_cnt.index[dt_cnt > 1]
                for dt in dt_dup:
                    tmp = df_tmp[df_tmp.Datetime==dt]
                    if tbl_name == 'labres':
                        df.drop(tmp.index[tmp.Status.isin([9, 72, 136, -120])], inplace=True)
                        tmp.drop(tmp.index[tmp.Status.isin([9, 72, 136, -120])], inplace=True)

                    if len(tmp) > 1:
                        if tbl_name!='labres' and voi.loc[vid].VariableUnit in ['Categorical', 'Ordinal Score']:
                            df.drop(tmp.index, axis=0, inplace=True) 
                        elif tmp.Value.std() < 0.05 * mean_std.loc[vid].Std:
                            df.loc[tmp.index, 'Value'] = tmp.Value.mean()
                        else:
                            df.drop(tmp.index, axis=0, inplace=True)

        df.drop_duplicates(['Datetime', 'VariableID', 'Value'], inplace=True)
        if output_to_disk:
            df.to_hdf(output_path, 'data', append=True, format='table', data_columns=['PatientID', 'VariableID'], complevel=5, complib='blosc:lz4')

        if (n+1)%50 == 0:
            print('%d / %d'%(n+1, len(pid_list)))

