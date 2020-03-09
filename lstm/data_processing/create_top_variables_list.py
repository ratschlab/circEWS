
# coding: utf-8

import pandas as pd
import numpy as np
from os.path import join
import pickle
import h5py
import csv
import sklearn.preprocessing as sk_preprocessing

data_version = '180918'
bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'

def get_top_features(num_topv):
    SHAPLEY_VALUES_PATH=join(bern_path, '8_predictions', data_version, 'reduced', 'temporal_5', 
                             'WorseStateFromZero_0.0_8.0_normal_model_50_percent_lightgbm_full',
                             'shap_selected_features.tsv')
    sel_col_X = []
    tuple_lst=[]
    with open(SHAPLEY_VALUES_PATH,'r') as fp:
        csv_fp=csv.reader(fp,delimiter='\t')
        next(csv_fp)
        for fname,score in csv_fp:
            tuple_lst.append((fname,float(score)))
            sel_col_X.append(fname)
    filter_cols=list(map(lambda elem: elem[0], sorted(tuple_lst,key=lambda elem: elem[1])))[-num_topv:]
    final_cols=list(filter(lambda col: col in filter_cols, sel_col_X))
    static_topv = [col.split('_')[1] for col in final_cols if 'static' in col]
    final_cols = [col for col in final_cols if 'static' not in col]
    return final_cols, static_topv,

num_topv = 500
fdynamic, fstatic = get_top_features(num_topv)
print(len(fdynamic), len(fstatic))
np.savez(join(ber_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', 'top%d_features.npz'%num_topv), 
         dynamic=fdynamic, static=fstatic)

vdynamic = list(set([[xx for xx in col.split('_') if 'vm' in xx or 'pm' in xx][0] for col in fdynamic if 'vm' in col or 'pm' in col]))
vdynamic.sort()
vstatic = fstatic
num_topv = len(vdynamic) + len(vstatic)
print(len(vdynamic), len(vstatic))
np.savez(join(bern_path, '7_ml_input', 'lstm_%s'%data_version, 'reduced', 'top%d_variables.npz'%num_topv), 
         dynamic=vdynamic, static=vstatic)

