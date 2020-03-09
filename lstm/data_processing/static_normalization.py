import pandas as pd
import numpy as np
from os.path import join, exists
from os import mkdir
import pickle

import sklearn.preprocessing as sk_preprocessing


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='temporal_5')
parser.add_argument('--output_dir', default='all_features')
args = parser.parse_args()
data_split = args.data_split
output_dir = args.output_dir


data_version = '180918'
bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'
static_path = join(bern_path, '5_imputed', 'imputed_'+data_version, 'reduced', data_split)
output_path = join(bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', data_split, output_dir)
split_path = join(bern_path, 'misc_derived', 'temporal_split_180918.pickle')
if not exists(output_path):
    mkdir(output_path)
with open(split_path, 'rb') as f:
    set_pids_ = pickle.load(f)[data_split]


df_static = pd.read_hdf(join(static_path, 'static.h5'))
df_static = df_static[df_static.PatientID.isin(np.concatenate([val for _, val in set_pids_.items()]))]
df_static.set_index('PatientID', inplace=True)
df_static['Sex'] = df_static.Sex.apply(lambda x: 1 if x=='F' else 0)
df_apachegroup = pd.read_hdf('/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/derived_mafacz/bern_apacheII/appache_group.h5', 'table', mode='r')

df_static = df_static.merge(df_apachegroup[['patientid', 'groupid']].set_index('patientid'), how='left', left_index=True, right_index=True)

df_static.drop('APACHEPatGroup', axis=1, inplace=True)
df_static.rename(columns={'groupid': 'APACHEPatGroup'}, inplace=True)

df_static = df_static[df_static.isnull().sum(axis=1) == 0]

cat_cols = []
bin_cols = []
num_cols = []
for col in df_static.columns:
    if len(df_static[col].unique()) == 2:
        bin_cols.append(col)
    elif len(df_static[col].unique()) < 25:
        cat_cols.append(col)
    else:
        num_cols.append(col)


print('# numerical static features:', len(num_cols), num_cols)
print('# categorical static features:', len(cat_cols), cat_cols)
print('# binary static features:', len(bin_cols), bin_cols)


# binarize binary variables
for col in bin_cols:
    lb = sk_preprocessing.LabelBinarizer()
    tmp = lb.fit_transform(df_static[col].values)
    tmp[tmp==0] = -1
    df_static[col] = tmp


# one-hot encode categorical variables
df_static.loc[df_static.index[df_static.PatGroup==-1], 'PatGroup']=1
for col in cat_cols:
    enc = sk_preprocessing.OneHotEncoder(sparse=False)
    code = enc.fit_transform(df_static[col].values.reshape((-1,1)))
    code[code==0] = -1
    for i in range(code.shape[1]):
        tmp = code[:,i].astype(int)
        df_static[col+str(i)] = tmp
df_static.drop(cat_cols, axis=1, inplace=True)


# normalize numerical variables
train_mat = df_static[df_static.index.isin(set_pids_['train'])][num_cols].as_matrix().copy()
all_mat = df_static[num_cols].as_matrix().copy()

scaler = sk_preprocessing.MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_mat)
df_static[num_cols] = scaler.transform(all_mat)
df_static.to_hdf(join(output_path, 'X_mm_scaled', 'static.h5'), 'data',
                 complib='blosc:lz4', complevel=5, format='table')
  

scaler = sk_preprocessing.StandardScaler()
scaler.fit(train_mat)
df_static[num_cols] = scaler.transform(all_mat)
df_static.to_hdf(join(output_path, 'X_std_scaled', 'static.h5'), 'data',
                 complib='blosc:lz4', complevel=5, format='table')

