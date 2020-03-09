#!/usr/bin/env python
import pandas as pd
import numpy as np

import gc
import os

import sys
sys.path.append('../utils')
import preproc_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-version')
args = parser.parse_args()
version = args.version

hdf_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version, 'datetime_fixed')

# VariableID and table information of the static variables
static_info = dict()
static_info.update(dict(generaldata=['PatientID', 'Sex', 'birthYear', 'AdmissionTime', 'PatGroup']))
static_info.update(dict(observrec=dict(Discharge=14000100, 
                                       Emergency=14000140, 
                                       Surgical=14000150)))
static_info.update(dict(dervals=dict(Euroscores=30010001, 
                                     APACHECode=30000100)))

# Map the variableID to its corresponding table
id_name_mapping = {val: key for key, val in static_info['observrec'].items()}
id_name_mapping.update({val: key for key, val in static_info['dervals'].items()})

# Only use patientID after 2008
pid_list = preproc_utils.get_filtered_patient_ids()

# Load the generaldata table where most static variables are
generaldata = pd.read_hdf(os.path.join(preproc_utils.datapath, '1_hdf5_consent', '180704', 'generaldata.h5'))
generaldata['AdmissionTime'] = pd.to_datetime(generaldata['AdmissionTime'])
generaldata = generaldata[generaldata.PatientID.isin(pid_list)].copy()
generaldata['Age'] = generaldata.AdmissionTime.apply(lambda x: x.year) - generaldata.birthYear


pid_chunkfile_index = preproc_utils.get_chunking_info(version=version)
static = []
for tbl in ['observrec', 'dervals']:
    vid_list = [str(val) for _, val in static_info[tbl].items()]
    df = []
    for idx_chunkfile in np.sort(pid_chunkfile_index.ChunkfileIndex.unique()):
        tmp_pid_list = np.array(pid_chunkfile_index.index[pid_chunkfile_index.ChunkfileIndex==idx_chunkfile])
        chunkfile_path = os.path.join(hdf_path, tbl, '%s_%d_%d--%d.h5'%(tbl, idx_chunkfile, np.min(tmp_pid_list), np.max(tmp_pid_list)))
        df.append(pd.read_hdf(chunkfile_path, where='VariableID in (%s)'%(','.join(vid_list))))
    df = pd.concat(df, axis=0)
    df.rename(columns={'DateTime': 'Datetime'}, inplace=True)
    
    # remove invalidated records based on the status
    status_set = df.Status.unique()
    status_binary = ['{0:11b}'.format(s)[::-1] for s in status_set]
    # 1: invalidated; 5: notified but not measured; 
    invalid_status_set = status_set[np.where( [(x[1]=='1') or (x[5]=='1') for x in status_binary])]
    if len(invalid_status_set) > 0:
        df.drop(df.index[df.Status.isin(invalid_status_set)], inplace=True)
        
    if tbl == 'observrec':
        # delete records with values that are not 0 or 1 for emergency variable
        tmp = df[df.VariableID==14000140]
        if len(tmp) > 0:
            df.drop(tmp.index[np.logical_and(tmp.Value!=0, tmp.Value!=1)], axis=0, inplace=True)
        
    static.append(df)

static = pd.concat(static, axis=0)
static['VariableName'] = static.VariableID.apply(lambda x: id_name_mapping[x])

# Pivot the static table to static feature matrix
# Keep all records with different values for now 
static = pd.pivot_table(static, values='Value', index=['PatientID', 'Datetime', 'EnterTime'],
                        columns=['VariableName'])


# Read the clean height information
height_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version, 'oor_removed', 'height')
df_height = []
for f in os.listdir(height_path):
    df_height.append(pd.read_hdf(os.path.join(height_path, f)))
df_height = pd.concat(df_height, axis=0)
df_height = df_height[df_height.PatientID.isin(pid_list)]
df_height.rename(columns={'DateTime': 'Datetime'}, inplace=True)
df_height = pd.pivot_table(df_height, values='Value', index=['PatientID', 'Datetime', 'EnterTime'], columns='VariableID')
df_height.rename(columns={10000450: 'Height'}, inplace=True)
gc.collect()


print(len(df_height.reset_index().PatientID.unique()))

# Add heights to the static feature matrix
static = static.merge(df_height, left_index=True, right_index=True, how='outer')
static.reset_index(inplace=True)


# Add generaldata feature matrix to static feature matrix
static = generaldata.merge(static, how='outer', left_on='PatientID', right_on='PatientID')
del generaldata                      
gc.collect()


# Remove duplicates and only keep the latest records (the latest record can be correct of the previous ones instead of being dynamic)
static = static.sort_values(by=['PatientID', 'Datetime'])
static.drop_duplicates(set(static.columns) - {'EnterTime', 'Datetime'}, inplace=True, keep='last')
gc.collect()


# Forward fill and backward fill the values 
pid_cnt = static.PatientID.value_counts()
for pid in pid_cnt.index[pid_cnt>1]:
    static.loc[static.index[static.PatientID==pid]] = static[static.PatientID==pid].fillna(method='ffill')
    static.loc[static.index[static.PatientID==pid]] = static[static.PatientID==pid].fillna(method='bfill')


static.drop_duplicates(set(static.columns) - {'EnterTime', 'Datetime'}, inplace=True, keep='last')


for col in ['Surgical', 'Emergency', 'Discharge', 'APACHECode', 'Euroscores', 'Height']:
    if col not in static.columns:
        continue
    # After removing imputed duplicates, there still exist multiple values for the patients, and I cannot decide
    # which one to use. It's better to set them `NaN` than picking the wrong one
    pid_cnt = static[['PatientID', col]].drop_duplicates().PatientID.value_counts()
    if np.sum(pid_cnt>1) > 0:
        pid_to_drop = pid_cnt.index[pid_cnt>1]
        static.loc[static.index[static.PatientID.isin(pid_to_drop)], col] = float('NaN')
static.drop_duplicates(set(static.columns) - {'EnterTime', 'Datetime'}, inplace=True)

# Alternative: only keep the latest static records
static.drop(['Datetime', 'EnterTime'], axis=1, inplace=True)

# integrate the APACHE group information from Matthias. 
tmp = open(os.path.join(preproc_utils.datapath, 'misc_derived', 'apachegroup_patients_180130.tsv'), 'r').readlines()
df_apachegroup  = []
for i, line in enumerate(tmp):
    if i > 0:
        df_apachegroup.append([float(x) for x in line.rstrip('\n').split('\t')])
df_apachegroup = pd.DataFrame(df_apachegroup, columns=['PatientID', 'APACHEPatGroup'])
df_apachegroup['PatientID'] = df_apachegroup.PatientID.astype(np.int64)
df_apachegroup['APACHEPatGroup'] = df_apachegroup.APACHEPatGroup.astype(np.int64)
df_apachegroup.set_index('PatientID', inplace=True)

static.set_index('PatientID', inplace=True)

static = static.merge(df_apachegroup, how='left', left_index=True, right_index=True)
static.reset_index(inplace=True)

static.to_hdf(os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version, 'static.h5'), 'data', 
              complevel=5, complib='blosc:lz4', data_columns=True, format='table')
    

