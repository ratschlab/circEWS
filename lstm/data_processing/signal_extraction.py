import pandas as pd
import numpy as np

from os import listdir, makedirs
from os.path import join, exists

import gc


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-index_batch', type=int)
parser.add_argument('--data_split', default='temporal_5')
args = parser.parse_args()
index_batch = args.index_batch
data_split = args.data_split


data_version = '180918'
bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'

impute_path = join(bern_path, '5_imputed', 'imputed_'+data_version, 'reduced', data_split)
label_path = join(bern_path, '7_ml_input', data_version, 'reduced', data_split, 'AllLabels_0.0_8.0', 'y')
output_path = join(bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', data_split, 'all_signals', 'unnormalized')
ep_path = join(bern_path, '3a_endpoints', 'v6b', 'reduced')
if not exists(output_path):
    makedirs(output_path)


df_tmp = pd.read_hdf(join(impute_path, listdir(impute_path)[0]), start=0, stop=0, mode='r')
info_cols = ['PatientID', 'AbsDatetime', 'RelDatetime']
variables = [col for col in df_tmp.columns if col not in info_cols and 'CUM' not in col and 'TIME_TO' not in col]
variables.sort()
obs_cnt_cumsum = np.array([col for col in df_tmp.columns if 'CUM' in col])
obs_cnt_cumsum = obs_cnt_cumsum[np.argsort([col.split('_')[0] for col in obs_cnt_cumsum])]
obs_time_to = np.array([col for col in df_tmp.columns if 'TIME_TO' in col])
obs_time_to = obs_time_to[np.argsort([col.split('_')[0] for col in obs_time_to])]
non_pharma_vars = [col for col in variables if 'vm' in col]
pharma_vars = [col for col in variables if 'pm' in col]
print('# columns in imputed dataframe:', len(df_tmp.columns))
print('# columns for cumulative cnt of observations:', len(obs_cnt_cumsum))
print('# columns for time to last observations:', len(obs_cnt_cumsum))
print('# columns for true observed values:', len(variables))
print('# non-pharma variables:', len(non_pharma_vars))
print('# pharma variables:', len(pharma_vars))
del df_tmp
gc.collect()


cnt_pid_no_data = 0

columns = np.concatenate((info_cols, variables, obs_cnt_cumsum))
rename_dict = {col: col.split('_')[0]+'_obs' for col in obs_cnt_cumsum}
obs_true = [col.split('_')[0]+'_obs' for col in obs_cnt_cumsum]
ep_files = listdir(ep_path)


df_impute = pd.read_hdf(join(impute_path, 'batch_%d.h5'%index_batch), columns=columns).rename(columns=rename_dict)
f_ep = [f for f in ep_files if '_%d_'%index_batch in f][0]
pids = df_impute.PatientID.unique()
print('# patients in batch %d:'%index_batch, len(pids))


for n, pid in enumerate(pids):

    df_label = pd.read_hdf(join(label_path, 'batch_%d.h5'%index_batch), str(pid))
    if len(df_label) == 0 or np.sum(df_label['SampleStatus_WorseStateFromZero0.0To8.0Hours']=='VALID') == 0:
        print('patient %s does not have valid data'%pid)
        cnt_pid_no_data += 1
        continue    
    del df_label


    # Use real observation values as features 
    df = df_impute[df_impute.PatientID==pid].copy()
    
    # Use features showing whether a variable value is imputed or real, 0 for imputed value and 1 for real value
    for col in obs_true:
        df.loc[df.index[1:], col] = np.diff(df[col])
        df[col] = (df[col].values > 0).astype(int)

    # Use the history of non-interpolated endponit status as a feature 
    df_ep = pd.read_hdf(join(ep_path, f_ep), where='PatientID=%d'%pid, 
                        columns=['PatientID', 'Datetime', 'endpoint_status_nointerp'])
    df_ep['ep_category'] = 0
    idx_category_1 = df_ep.index[df_ep.endpoint_status_nointerp.isin(['event 1', 'event 2', 'event 3'])]
    idx_category_2 = df_ep.index[df_ep.endpoint_status_nointerp.isin(['probably not 1', 'probably not 2', 'probably not 3'])]
    idx_category_3 = df_ep.index[df_ep.endpoint_status_nointerp.isin(['maybe 1', 'maybe 2', 'maybe 3'])]
    df_ep.loc[idx_category_1, 'ep_category'] = 1
    df_ep.loc[idx_category_2, 'ep_category'] = 2
    df_ep.loc[idx_category_3, 'ep_category'] = 3
    
    # Aligning the absolute time of the endpoint dataframe with the imputed dataframe
    dt_ref = df.iloc[0].AbsDatetime
    idx_dt_ref_in_ep = np.where(np.abs(df_ep.Datetime - dt_ref) / np.timedelta64(1,'s') <= 2.5*60)[0][0]
    df_ep['AbsDatetime'] = (df_ep.Datetime - df_ep.iloc[idx_dt_ref_in_ep].Datetime) + dt_ref
    df_ep.drop(['PatientID', 'Datetime', 'endpoint_status_nointerp'], inplace=True, axis=1)

    # Merge the imputed dataframe with the endpoint feature dataframe
    df_ep.set_index('AbsDatetime', inplace=True)    
    df.set_index('AbsDatetime', inplace=True)
    df = df.merge(df_ep, how='left', left_index=True, right_index=True)
    df.reset_index(inplace=True)
    
    # Fill in the nan values for endpoint features with 0
    df['ep_category'] = df.ep_category.fillna(0)

    # Check if there is other nan values in the feature matrix
    assert(df.isnull().sum().sum()==0)
    
    # Output data to file 
    df.to_hdf(join(output_path, 'batch_%d.h5'%index_batch), 'p'+str(pid), 
              complib='blosc:lz4', complevel=5)
    print('patient %s written to disk.'%pid)
    gc.collect()
