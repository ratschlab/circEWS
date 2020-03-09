import pandas as pd
import numpy as np

from os import listdir, makedirs
from os.path import join, exists, split

import gc
import h5py
import pickle

from sklearn.preprocessing import StandardScaler as sk_StandardScaler
from sklearn.preprocessing import MinMaxScaler as sk_MinMaxScaler
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
import sys
sys.path.append('../../utils')
import preproc_utils


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='temporal_5')
args = parser.parse_args()
data_split = args.data_split


# data_split = 'temporal_5'


data_version = '180918'
bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'
input_path = join(bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', data_split, 'all_signals', 'unnormalized')


with h5py.File(join(input_path, listdir(input_path)[0])) as tmp:
    pids = [key for key in tmp.keys()]
df_tmp = pd.read_hdf(join(input_path, listdir(input_path)[0]), pids[0])
variables = [col for col in df_tmp.columns if ('vm' in col or 'pm' in col) and 'obs' not in col]
obs_true = [col for col in df_tmp.columns if 'obs' in col]
print('# columns in unnormalized data dataframe:', len(df_tmp.columns))
print('# columns for true observed values:', len(variables))
print('# columns for imputed status features', len(obs_true))
del df_tmp
gc.collect()


# Find the data type of different variables from the excel reference table
voi = []
for tbl_name in ['monvals', 'dervals', 'observrec', 'pharmarec', 'labres']:
    voi.append(preproc_utils.voi_id_name_mapping(tbl_name, True))
    if tbl_name == 'labres':
        voi[-1]['Type'] = 'Lab'
        voi[-1]['MetaVariableName'] = voi[-1]['VariableName']
voi = pd.concat(voi)

voi = voi[['MetaVariableID', 'MetaVariableName', 'MetaVariableUnit', 'Type']]
voi = voi[voi.MetaVariableUnit.notnull()]

mvid_categorical = voi[voi.MetaVariableUnit.apply(lambda x: 'categorical' in x.lower())].MetaVariableID.unique()
mvid_ordinal = voi[voi.MetaVariableUnit.apply(lambda x: 'count' in x.lower() or 'ordinal' in x.lower())].MetaVariableID.unique()
mvid_binary = voi[voi.MetaVariableUnit.apply(lambda x: '[yes/no]' in x.lower())].MetaVariableID.unique()

def find_variables(vid_list):
    var_list = []
    for mvid in vid_list:
        for col in variables:
            if col == 'pm%d'%mvid or col == 'vm%d'%mvid:
                var_list.append(col)
    var_list.sort()
    return var_list

binary_vars = find_variables(mvid_binary)
categorical_vars = find_variables(mvid_categorical)
ordinal_vars = find_variables(mvid_ordinal)


print('# binary variables:', len(binary_vars))
print('# categorical variables:', len(categorical_vars))
print('# ordinal variables:', len(ordinal_vars))
categorical_vars += ['ep_category']

# Exclude binary and categorical variables from the scaler computation
numerical_vars = list(set(variables) - set(binary_vars) - set(categorical_vars) - set(ordinal_vars))
numerical_vars.sort()
print('# Numerical variables for computing scaler:', len(numerical_vars))


with open(join(bern_path, 'misc_derived', 'temporal_split_%s.pickle'%data_version), 'rb') as f:
    set_pids_ = pickle.load(f)[data_split]


categorical_values_dict = {col: set() for col in categorical_vars}
s_scaler = sk_StandardScaler()
m_scaler = sk_MinMaxScaler(feature_range=(-1, 1))

all_batches = np.sort([int(f.split('_')[-1][:-3]) for f in listdir(input_path) if 'h5' in f])
for index_batch in all_batches:    
    print('*** Batch %d ***'%index_batch)
    with h5py.File(join(input_path, 'batch_%d.h5'%index_batch), 'r') as tmp:
        pids = np.array([key for key in tmp.keys()])
        
    for n, pid in enumerate(pids):
        df = pd.read_hdf(join(input_path, 'batch_%d.h5'%index_batch), pid)[categorical_vars+numerical_vars+ordinal_vars]
        for col in categorical_vars:
            categorical_values_dict[col] |= set(df[col].values)
        
        if int(pid[1:]) not in set_pids_['train']:
            continue
        numerical_mat = df[numerical_vars+ordinal_vars].as_matrix()
        s_scaler.partial_fit(numerical_mat)
        m_scaler.partial_fit(numerical_mat)
        gc.collect()

onehot_enc = {col: sk_OneHotEncoder(sparse=False) for col in categorical_vars}
for key, val in categorical_values_dict.items():
    onehot_enc[key].fit(np.reshape(list(val), (-1,1)))


with open(join(split(input_path)[0], 'standard_scaler.pkl'), 'wb') as f:
    pickle.dump((s_scaler, numerical_vars+ordinal_vars), f)


with open(join(split(input_path)[0], 'minmax_scaler.pkl'), 'wb') as f:
    pickle.dump((m_scaler, numerical_vars+ordinal_vars), f)


with open(join(split(input_path)[0], 'onehot_encoders.pkl'), 'wb') as f:
    pickle.dump((onehot_enc, categorical_vars), f)

