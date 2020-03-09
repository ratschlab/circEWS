import numpy as np
import pandas as pd
from os.path import join, exists
from os import listdir, mkdir, makedirs
import pickle
import h5py
import gc


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-index_batch', type=int)
parser.add_argument('--data_split', default='temporal_5')
parser.add_argument('--data_version', default='180918')

args = parser.parse_args()
index_batch = args.index_batch
data_split = args.data_split
data_version = args.data_version



# index_batch = 8
# data_split = 'temporal_5'
# data_version = '180918'

bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'
data_path = join(bern_path, '7_ml_input', data_version, 'reduced', data_split,'AllLabels_0.0_8.0')
feature_path = join(bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', data_split, 'all_features')
label_path = join(bern_path, '7_ml_input', data_version, 'reduced', data_split, 'AllLabels_0.0_8.0', 'y')
ep_path = join(bern_path, '3a_endpoints', 'v6b','reduced')


std_feature_path = join(feature_path, 'X_std_scaled')
mm_feature_path = join(feature_path, 'X_mm_scaled')
output_label_path = join(feature_path, 'Y')

if not exists(std_feature_path):
    mkdir(std_feature_path)
if not exists(mm_feature_path):
    mkdir(mm_feature_path)
if not exists(output_label_path):
    makedirs(output_label_path)


for set_ in ['train', 'val', 'test']:
    if not exists(join(std_feature_path, set_)):
        mkdir(join(std_feature_path, set_))

    if not exists(join(mm_feature_path, set_)):
        mkdir(join(mm_feature_path, set_))

    if not exists(join(output_label_path, set_)):
        makedirs(join(output_label_path, set_))

with open(join(bern_path, 'misc_derived', 'temporal_split_180918.pickle'), 'rb') as f:
    set_pids_ = pickle.load(f)[data_split]


with open(join(feature_path, 'onehot_encoders.pkl'), 'rb') as f:
    onehot_enc, categorical_cols = pickle.load(f)
with open(join(feature_path, 'standard_scaler.pkl'), 'rb') as f:
    s_scaler, numerical_cols = pickle.load(f)
with open(join(feature_path, 'minmax_scaler.pkl'), 'rb') as f:
    m_scaler = pickle.load(f)[0]


with np.load(join(feature_path, 'feature_columns.npz')) as tmp:
    fcols = tmp['final'].tolist()
    binary_cols = tmp['binary'].tolist()
    


final_numerical_cols = np.array(numerical_cols)[s_scaler.scale_!=1].tolist()


with h5py.File(join(data_path, 'X', 'batch_%d.h5'%index_batch),'r') as tmp:
    pids = [key for key in tmp.keys()]


pids_current_split = np.concatenate([val for key, val in set_pids_.items()])
if np.sum(np.isin([int(pid) for pid in pids], pids_current_split)) == 0:
    print('No patient from temporal %d exist in batch %d'%(data_split, index_batch))
    exit(0)


for n, pid in enumerate(pids):
    pid_num = int(pid)
    
    for s in ['train', 'val', 'test']:
        if pid_num in set_pids_[s]:
            set_ = s
            break
            
    df_l = pd.read_hdf(join(label_path, 'batch_%d.h5'%index_batch), pid)
#     try:
#         df_l = pd.read_hdf(join(label_path, set_, 'batch_%d.h5'%index_batch), 'p'+pid, mode='r')
#     except:
#         print('patient %s does not have label'%pid)
#         continue
    df_f = pd.read_hdf(join(data_path, 'X', 'batch_%d.h5'%index_batch), pid, mode='r')
    df_f.drop('PatientID', axis=1, inplace=True)
    assert(np.sum( np.abs(df_l.AbsDatetime - df_f.AbsDatetime) / np.timedelta64(1, 's')) == 0)

    if len(df_f) == 0:
        print('patient %s does not have valid data'%pid)
        continue
    df_f = df_f[['AbsDatetime']+fcols]
#     df_f.set_index('AbsDatetime', inplace=True)

    df_l.set_index('AbsDatetime', inplace=True)
    label_col = 'Label_WorseStateFromZero0.0To8.0Hours'
    df_l = df_l[[label_col]]
    gc.collect()

    df_l[label_col] = df_l[label_col].fillna(2)
    assert(len(df_f)==len(df_l))

    ep_f = [f for f in listdir(ep_path) if '_%d_'%index_batch in f][0]
    df_ep = pd.read_hdf(join(ep_path, ep_f), where='PatientID=%s'%pid, 
                        columns=['PatientID', 'Datetime', 'endpoint_status'],
                        mode='r').drop('PatientID', axis=1)
    

    dt_ref = df_f.iloc[0].AbsDatetime
    idx_dt_ref = np.where(np.abs(df_ep.Datetime - dt_ref)/np.timedelta64(60, 's')<=2.5)[0][-1]
    df_ep['AbsDatetime'] = dt_ref + (df_ep.Datetime-df_ep.iloc[idx_dt_ref].Datetime)
    df_ep.drop('Datetime', axis=1, inplace=True)
    df_ep.set_index('AbsDatetime', inplace=True)
    df_f.set_index('AbsDatetime', inplace=True)

    df_merge = df_ep.merge(df_f, how='outer', left_index=True, right_index=True)
    df_merge = df_merge.merge(df_l, how='outer', left_index=True, right_index=True)

    if len(df_merge) > len(df_f):
        df_merge = df_merge[df_merge.index<=df_f.index[-1]]

    if np.sum(df_merge.isnull().sum(axis=1) >= len(fcols)) == 1:
        df_merge = df_merge[df_merge.index >= dt_ref]
    elif np.sum(df_merge.isnull().sum(axis=1) >= len(fcols)) > 1:
        print(np.sum(df_merge.isnull().sum(axis=1) >= len(fcols)))
        import ipdb
        ipdb.set_trace()
    
    tmp = df_merge[np.logical_and(df_merge.endpoint_status=='event 1', 
                                  df_merge['Label_WorseStateFromZero0.0To8.0Hours']!=2)]
    try:
        assert(len(tmp) == 0)
    except AssertionError:
        print(tmp[[col for col in tmp.columns if col not in fcols]])

    df_f = df_merge[fcols].copy()
    df_f['RelDatetime'] /= 3600
        
    for col in binary_cols:
        df_f.loc[df_f.index[df_f[col]==0], col] = -1
        
    new_categorical_features = []
    for key in categorical_cols:
        mat_enc = onehot_enc[key].transform(df_f[key].values.reshape((-1,1)))
        mat_enc[mat_enc==0] = -1
        for j in range(mat_enc.shape[1]):
            df_f['%s_%d'%(key, j)] = mat_enc[:,j] 
            new_categorical_features.append('%s_%d'%(key, j))
    df_f.drop(categorical_cols, axis=1, inplace=True)
    
    final_fcols = np.sort(final_numerical_cols+new_categorical_features+binary_cols)

    mat = df_f[numerical_cols].as_matrix().copy()
    trans_mat = s_scaler.transform(mat.copy())
    for j in range(mat.shape[1]):
        df_f[numerical_cols[j]] = trans_mat[:,j]
        
    df_f.to_hdf(join(std_feature_path, set_, 'batch_%d.h5'%index_batch), pid,
              complib='blosc:lz4', complevel=5)

#     with h5py.File(join(std_feature_path, set_, 'batch_%d.h5'%index_batch), 'a') as fstore_standard:
#         fstore_standard.create_dataset(pid, data=df_f[final_fcols].as_matrix(), compression='gzip')

    trans_mat = m_scaler.transform(mat.copy())
    for j in range(mat.shape[1]):
        df_f[numerical_cols[j]] = trans_mat[:,j]

    df_f.to_hdf(join(mm_feature_path, set_, 'batch_%d.h5'%index_batch), pid,
              complib='blosc:lz4', complevel=5)
        
#     with h5py.File(join(mm_feature_path, set_, 'batch_%d.h5'%index_batch), 'a') as fstore_minmax:
#         fstore_minmax.create_dataset(pid, data=df_f[final_fcols].as_matrix(), compression='gzip')
        
    df_l.to_hdf(join(output_label_path, set_, 'batch_%d.h5'%index_batch), pid,
                complib='blosc:lz4', complevel=5)
 
    gc.collect()

np.save(join(feature_path, 'feature_columns.npy'), final_fcols)

