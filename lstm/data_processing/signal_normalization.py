import pandas as pd
import numpy as np

from os import listdir, makedirs
from os.path import join, exists, split

import gc
import h5py
import pickle

import sys
sys.path.append('../../utils')
import preproc_utils


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-index_batch', type=int)
parser.add_argument('--data_split', default='temporal_5')
args = parser.parse_args()
index_batch = args.index_batch
data_split = args.data_split


# index_batch = 8
# data_split = 'temporal_5'


data_version = '180918'

bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'

feature_path = join(bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', data_split, 'all_signals', 'unnormalized')
label_path = join(bern_path, '7_ml_input', data_version, 'reduced', data_split, 'AllLabels_0.0_8.0', 'y')


std_feature_path = feature_path.replace("unnormalized", 'X_std_scaled')
mm_feature_path = feature_path.replace("unnormalized", 'X_mm_scaled')
output_label_path = feature_path.replace("unnormalized", 'Y')


if not exists(std_feature_path):
    makedirs(std_feature_path)

if not exists(mm_feature_path):
    makedirs(mm_feature_path)

if not exists(output_label_path):
    makedirs(output_label_path)
    
for set_ in ['train', 'val', 'test']:
    if not exists(join(std_feature_path, set_)):
        makedirs(join(std_feature_path, set_))

    if not exists(join(mm_feature_path, set_)):
        makedirs(join(mm_feature_path, set_))
        
    if not exists(join(output_label_path, set_)):
        makedirs(join(output_label_path, set_))


with open(join(split(feature_path)[0], 'onehot_encoders.pkl'), 'rb') as f:
    onehot_enc, categorical_vars = pickle.load(f)

with open(join(split(feature_path)[0], 'standard_scaler.pkl'), 'rb') as f:
    s_scaler, numerical_vars = pickle.load(f)
with open(join(split(feature_path)[0], 'minmax_scaler.pkl'), 'rb') as f:
    m_scaler = pickle.load(f)[0]

with h5py.File(join(feature_path, listdir(feature_path)[0]), mode='r') as tmp:
    pids = [key for key in tmp.keys()]
df_tmp = pd.read_hdf(join(feature_path, listdir(feature_path)[0]), pids[0])
variables = np.sort([col for col in df_tmp.columns if ('vm' in col or 'pm' in col) and 'obs' not in col])
binary_vars = [col for col in variables if col not in categorical_vars+numerical_vars] + [col for col in df_tmp.columns if 'obs' in col]


split_path = join(bern_path, 'misc_derived', 'temporal_split_%s.pickle'%data_version)
with open(split_path, 'rb') as f:
    set_pids_ = pickle.load(f)[data_split]
all_batches = np.sort([int(f.split('_')[-1][:-3]) for f in listdir(feature_path) if 'h5' in f])


with h5py.File(join(feature_path, 'batch_%d.h5'%index_batch), 'r') as tmp:
    pids = [key for key in tmp.keys()]

for n, pid in enumerate(pids):
    for s in ['train', 'val', 'test']:
        if int(pid[1:]) in set_pids_[s]:
            set_ = s
            break
            
    df = pd.read_hdf(join(feature_path, 'batch_%d.h5'%index_batch), pid)

    df_label = pd.read_hdf(join(label_path, 'batch_%d.h5'%index_batch), pid[1:])
    assert(np.sum( np.abs(df_label.AbsDatetime - df.AbsDatetime) / np.timedelta64(1, 's')) == 0)
    
    
    df.drop('PatientID', axis=1, inplace=True)
    df.set_index('AbsDatetime', inplace=True)
    
    df_label.set_index('AbsDatetime', inplace=True)
    label_col = 'Label_WorseStateFromZero0.0To8.0Hours'
    df_label = df_label[[label_col]]
    gc.collect()
    
    df_label[label_col] = df_label[label_col].fillna(2)
    
    # RelDatetime
    df['RelDatetime'] = df.RelDatetime / (28 * 24 * 3600)  - 0.5

    # Binary variable normalization
    binary_mat = df[binary_vars].as_matrix()
    binary_mat[binary_mat==0] = -1
    df[binary_vars] = binary_mat
    # Categorical variable encoding
    new_categorical_cols = []
    for var in categorical_vars:
        var_codes = onehot_enc[var].transform(df[var].values.reshape((-1,1)))
        var_codes[var_codes==0] = -1
        for j in range(var_codes.shape[1]):
            df['%s_%d'%(var,j)] = var_codes[:,j]
            new_categorical_cols.append('%s_%d'%(var,j))
    df.drop(categorical_vars, axis=1, inplace=True)
    # Ordinal and numerical variables
    val_mat = df[numerical_vars].as_matrix().copy()


    trans_mat = s_scaler.transform(val_mat)
    for k, col in enumerate(numerical_vars):
        df[col] = trans_mat[:,k]
    df.to_hdf(join(std_feature_path, set_, 'batch_%d.h5'%index_batch), pid, 
              complib='blosc:lz4', complevel=5)
    trans_mat = m_scaler.transform(val_mat)
    for k, col in enumerate(numerical_vars):
        df[col] = trans_mat[:,k]
    df.to_hdf(join(mm_feature_path, set_, 'batch_%d.h5'%index_batch), pid, 
              complib='blosc:lz4', complevel=5)

    df_label.to_hdf(join(output_label_path, set_, 'batch_%d.h5'%index_batch), pid, 
                    complib='blosc:lz4', complevel=5)
    gc.collect()

