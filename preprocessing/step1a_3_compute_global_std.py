#!/usr/bin/env python
import numpy as np
import pandas as pd

import os
import gc

import sys
sys.path.append('../utils')
import preproc_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-tbl_name')
parser.add_argument('-version')
parser.add_argument('--idx_vid', type=int, default=None)
args = parser.parse_args()
tbl_name = args.tbl_name
version = args.version
idx_vid = args.idx_vid

voi = preproc_utils.voi_id_name_mapping(tbl_name, True, version=version)
input_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version, 'oor_removed', tbl_name)
output_path = os.path.join(preproc_utils.datapath, 'misc_derived', 'xinrui', version, 'global_mean_std')
if not os.path.exists(output_path):
    os.makedirs(output_path)

if tbl_name == 'monvals':
    vid = voi.index[idx_vid]
    voi = voi.iloc[idx_vid]
    values = []
    for f in os.listdir(input_path):
        if '.h5' not in f:
            continue
        df = pd.read_hdf(os.path.join(input_path, f), where='VariableID=%d'%vid, columns=['Value'])
        values.extend(df.Value.tolist())
        gc.collect()
    mean_std = [[voi.VariableName, vid, np.mean(values), np.std(values)]]
    mean_std = pd.DataFrame(mean_std, columns=['VariableName', 'VariableID', 'Mean', 'Std'])
    mean_std.to_csv(os.path.join(output_path, tbl_name+'_%d.tsv'%idx_vid), sep='\t', index=False)
else:
    mean_std = []
    for i, vid in enumerate(voi.index):
        values = []
        for f in os.listdir(input_path):
            if '.h5' not in f:
                continue
            df = pd.read_hdf(os.path.join(input_path, f), where='VariableID=%d'%vid, columns=['Value'])
            values.extend(df.Value.tolist())
            gc.collect()
        mean_std.append([vid, np.mean(values), np.std(values)])
        print('%d / %d'%(i+1, len(voi)))
    mean_std = pd.DataFrame(mean_std, columns=['VariableID', 'Mean', 'Std'])
    mean_std = mean_std.merge(voi[['VariableName']], how='left', left_on='VariableID', right_index=True )
    mean_std.to_csv(os.path.join(output_path, tbl_name+'.tsv'), sep='\t', index=False)
