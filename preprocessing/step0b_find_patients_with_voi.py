#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../utils')
import preproc_utils


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-version')
parser.add_argument('-tbl_name')
args = parser.parse_args()
version = args.version
tbl_name = args.tbl_name

voi = preproc_utils.voi_id_name_mapping(tbl_name, version='v6' if version=='v6b' else version)

if tbl_name == 'pharmarec':
    df = pd.read_hdf(os.path.join(preproc_utils.datapath, '1_hdf5_consent', '180704', '%s.h5'%tbl_name), 
                     where='PharmaID in (%s)'%(','.join([str(vid) for vid in voi.index])), columns=['PatientID', 'PharmaID', 'GivenDose'])
    df.drop(df.index[df.GivenDose.isnull()], inplace=True)
    pids = df.PatientID.unique()

elif tbl_name == 'monvals':
    chunksize = 5 * 10**6
    df = pd.read_hdf(os.path.join(preproc_utils.datapath, '1_hdf5_consent', '180704', '%s.h5'%tbl_name), 
                     where='VariableID in (%s)'%(','.join([str(vid) for vid in voi.index])), columns=['PatientID', 'VariableID', 'Value'], chunksize=chunksize)
    pids = []
    for chunk in df:
        chunk.drop(chunk.index[chunk.Value.isnull()], inplace=True)
        pids.extend(chunk.PatientID.unique())
    pids = np.unique(pids)
else:
    df = pd.read_hdf(os.path.join(preproc_utils.datapath, '1_hdf5_consent', '180704', '%s.h5'%tbl_name), 
                     where='VariableID in (%s)'%(','.join([str(vid) for vid in voi.index])), columns=['PatientID', 'VariableID', 'Value'])
    df.drop(df.index[df.Value.isnull()], inplace=True)
    pids = df.PatientID.unique()

df_pid = pd.DataFrame(np.reshape(pids, (-1,1)), columns=['PatientID'])
if not os.path.exists(os.path.join(preproc_utils.datapath, 'misc_derived', 'id_lists', version)):
    os.mkdir(os.path.join(preproc_utils.datapath, 'misc_derived', 'id_lists', version))
df_pid.to_csv(os.path.join(preproc_utils.datapath, 'misc_derived', 'id_lists', version, 'patients_with_voi_in_%s.csv'%tbl_name), index=False)

