import pandas as pd
import numpy as np
from os.path import join, exists
from os import listdir, makedirs
import gc
import pickle
import h5py

from sklearn.preprocessing import StandardScaler as sk_StandardScaler
from sklearn.preprocessing import MinMaxScaler as sk_MinMaxScaler
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

import sys
sys.path.append('../../utils')
import preproc_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='temporal_5')
parser.add_argument('--data_version', default='180918')
args = parser.parse_args()
data_split = args.data_split
data_version = args.data_version

bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'
data_path = join(bern_path, '7_ml_input', data_version, 'reduced', data_split,'AllLabels_0.0_8.0')
output_path = join(bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', data_split, 'all_features')
if not exists(output_path):
    makedirs(output_path)
    print('Create =>',  output_path)

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


with h5py.File(join(data_path, 'X', 'batch_8.h5'), 'r') as tmp:
    pids = [key for key in tmp.keys()]
df = pd.read_hdf(join(data_path, 'X', 'batch_8.h5'), pids[0], mode='r')

final_cols = [col for col in df.columns if col not in ['AbsDatetime', 'PatientID'] and 'SampleStatus' not in col]

var_cols = [col for col in final_cols if 'vm' in col or 'pm' in col]
non_var_cols = [col for col in final_cols if col not in var_cols]

categorical_vars = np.unique([col.split('_')[2] for col in var_cols if 'mode' in col])
categorical_var_cols = [col for col in var_cols if np.sum([(x+'_' in col or col[-len(x):]==x)  and ('mode' in col or 'plain' in col) for x in categorical_vars])>0]

binary_var_cols = [col for col in var_cols if np.sum(['vm%d'%x in col or 'pm%d'%x in col for x in mvid_binary])>0 and 'plain_' in col]
binary_var_cols += [col for col in non_var_cols if '_cur' in col]

numerical_val_cols = [col for col in var_cols if col not in categorical_var_cols+binary_var_cols] + [col for col in non_var_cols if '_cur' not in col]

categorical_value = {key: set() for key in categorical_var_cols}
s_scaler = sk_StandardScaler()
m_scaler = sk_MinMaxScaler(feature_range=(-1, 1))

cnt_pid_no_data = 0
all_batches = np.sort([int(tmp.split('_')[-1][:-3]) for tmp in listdir(join(data_path, 'X')) if 'repacked' not in tmp])

with open(join(bern_path, 'misc_derived', 'temporal_split_%s.pickle'%data_version), 'rb') as f:
    set_pids_ = pickle.load(f)[data_split]
    
for index_batch in all_batches:
    with h5py.File(join(data_path, 'X', 'batch_%d.h5'%index_batch), 'r') as tmp:
        pids = [key for key in tmp.keys()]

    print('*** batch %d ***'%index_batch)
    for n, pid in enumerate(pids):
        df = pd.read_hdf(join(data_path, 'X', 'batch_%d.h5'%index_batch), pid, mode='r')
        for key in categorical_var_cols:
            categorical_value[key] |= set(df[key].values)
            
        if int(pid) not  in set_pids_['train']:
            continue
            
        if len(df) == 0 or np.sum(df['SampleStatus_WorseStateFromZero0.0To8.0Hours']=='VALID') == 0:
            print('patient %s does not have valid data'%pid)
            cnt_pid_no_data += 1
            continue

        df = df[final_cols]
        df['RelDatetime'] /= 3600
        mat = df[numerical_val_cols].as_matrix()
        s_scaler.partial_fit(mat)
        m_scaler.partial_fit(mat)
        gc.collect()

onehot_enc = {key: sk_OneHotEncoder(sparse=False) for key in categorical_value.keys()}
for key, val in categorical_value.items():
    onehot_enc[key].fit(np.reshape(list(val), ((-1,1))))

with open(join(output_path, 'standard_scaler.pkl'), 'wb') as f:
    pickle.dump((s_scaler, numerical_val_cols), f)

with open(join(output_path, 'minmax_scaler.pkl'), 'wb') as f:
    pickle.dump((m_scaler, numerical_val_cols), f)


with open(join(output_path, 'onehot_encoders.pkl'), 'wb') as f:
    pickle.dump((onehot_enc, categorical_var_cols), f)
    
with open(join(output_path, 'feature_columns.npz'), 'wb') as f:
    np.savez(f, 
             final=final_cols, 
             numerical=numerical_val_cols, 
             binary=binary_var_cols,
             categorical=categorical_var_cols)

