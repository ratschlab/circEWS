import pandas as pd
import numpy as np
from os.path import join
from os import listdir
import h5py
import sys
import gc
import pickle


# data_split = 'temporal_1'


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_split')
args = parser.parse_args()
data_split = args.data_split


data_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/lstm_180918/reduced'

topk_features = np.load(join(data_path, 'top500_features.npz'))
all_features = np.load(join(data_path, data_split, 'all_features', 'feature_columns.npy'))

is_topk_dynamic_features = np.vectorize(lambda x: np.sum([(tmp in x if 'mode' in x else x==tmp) for tmp in topk_features['dynamic']]) > 0)            
idx_topk_dynamic_features = np.where(is_topk_dynamic_features(all_features))[0]                                                                       
topk_dynamic_features = all_features[idx_topk_dynamic_features]                                                                                  


pid_info = dict()

for set_ in ['train', 'val', 'test']:
    batch_pid_start_end = []
    mat_store = h5py.File(join(data_path, data_split, 'all_features', 'X_std_scaled', set_+'.h5'), 'w') 
    for f in listdir(join(data_path, data_split, 'all_features', 'X_std_scaled', set_)):
        store = pd.HDFStore(join(data_path, data_split, 'all_features', 'X_std_scaled', set_, f), 'r')
        i_start = 0
        batch = int(f[:-3].split('_')[1])
        mat = []
        print(f)
        for i, key in enumerate(store.keys()):
            if (i+1)%100 == 0:
                print(i+1)
            mat.append(store[key][topk_dynamic_features].as_matrix())
            batch_pid_start_end.append( [batch, int(key[1:]), i_start, i_start+len(mat)] )
            i_start += len(mat)
            gc.collect()
        mat = np.vstack(tuple(mat))
        store.close()
        dset = mat_store.create_dataset('batch_%d'%batch, data=mat)
        gc.collect()
    mat_store.close()
    

    batch_pid_start_end = np.stack(tuple(batch_pid_start_end))
    pid_info.update({set_: batch_pid_start_end})
    gc.collect()


pickle.dump(pid_info, open(join(data_path, data_split, 'all_features', 'batch_pid_start_end.pkl'), 'wb'))

