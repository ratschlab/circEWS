'''
Class for generating a features data-frame from an imputed data-frame.
'''

import os
import sys
import os.path
import gc
import psutil
import random
import timeit
import time
import ipdb

import numpy as np
import scipy as sp
import scipy.stats as sp_stats
import pandas as pd

import circews.functions.util.array as mlhc_array

class Features:
    '''
    Transforms the imputed data patient frame into a data-frame that has final features for Machine Learning 
    in the columns. Combines different strategies like Fixed history features, multi-scale features, features summarizing the 
    entire history of the patient, and summarizing the current status using so-called instability history features.
    '''

    def __init__(self, dim_reduced_data=None, impute_grid_unit=None, dataset=None):
        self.dim_reduced_data=dim_reduced_data
        self.impute_grid_unit=int(impute_grid_unit/60.0)
        self.impute_grid_unit_secs=impute_grid_unit
        
        self.minimum_history_hours=0.5

        # High-resolution variable horizons (in minutes)
        self.horizon_hi_0_mins=30.0
        self.horizon_hi_1_mins=60.0
        self.horizon_hi_2_mins=240.0
        self.horizon_hi_3_mins=720.0

        # Medium-resolution variable horizons (in minutes)
        self.horizon_med_0_mins=720.0
        self.horizon_med_1_mins=1440.0
        self.horizon_med_2_mins=2160.0
        self.horizon_med_3_mins=2880.0

        # Low-resolution variable horizons (in minutes)
        self.horizon_low_0_mins=960.0
        self.horizon_low_1_mins=1920.0
        self.horizon_low_2_mins=2880.0
        self.horizon_low_3_mins=4320.0

        self.dataset=dataset
        
    def set_varencoding_dict(self, varencoding_dict):
        ''' Set the variable encoding dictionary as context information'''
        self.var_encoding_dict=varencoding_dict

    def set_pharma_dict(self, pharma_dict):
        ''' Set pharma dict as context information'''
        self.pharma_dict=pharma_dict

    def set_varparameters_dict(self, varparameters_dict):
        ''' Set the variable parameters dictionary as context information, which allows
            to look-up the scale of each variable for multi-scale history features'''
        self.varparameters_dict=varparameters_dict

    def transform(self, df_pat, df_label_pat, pid=None):
        ''' 
        Transforms the imputed data frame to a data frame with final features
        '''
        sample_idx=0
        start_ts=self.minimum_history_hours*3600.0-self.impute_grid_unit_secs
        samples_per_hour=int(3600.0/self.impute_grid_unit_secs)
        all_cols=list(filter(lambda col: "IMPUTE_STATUS" not in col, df_pat.columns.values.tolist()))
        label_cols=list(filter(lambda col: col not in ["AbsDatetime", "RelDatetime", "PatientID"], df_label_pat.columns.values.tolist()))
        rel_dt_col=df_pat["RelDatetime"]
        abs_dt_col=df_pat["AbsDatetime"]
        pid_col=df_pat["PatientID"]

        SYMBOLIC_MAX=43200

        if self.dim_reduced_data:
            cont_pharma_cols=list(filter(lambda item: item[0:2]=='pm' and "_IMPUTED_" not in item \
                                         and self.var_encoding_dict[item] in ["continuous","ordinal"], all_cols))
            binary_pharma_cols=list(filter(lambda item: item[0:2]=="pm" and "_IMPUTED_" not in item \
                                           and self.var_encoding_dict[item]=="binary", all_cols))
            cont_var_cols=list(filter(lambda item: item[0:2]=="vm" and "_IMPUTED_" not in item \
                                      and self.var_encoding_dict[item] in ["continuous","ordinal"], all_cols))
            cat_var_cols=list(filter(lambda item: item[0:2]=='vm' and "_IMPUTED_" not in item \
                                     and self.var_encoding_dict[item]=="categorical", all_cols))
            binary_var_cols=list(filter(lambda item: item[0:2]=='vm' and "_IMPUTED_" not in item \
                                        and self.var_encoding_dict[item]=="binary", all_cols))
        else:
            assert(False) # CODE PATH NOT USED

        cat_freq_hi_cols=[]
        cat_freq_med_cols=[]
        cat_freq_low_cols=[]

        cont_freq_hi_cols=[]
        cont_freq_med_cols=[]
        cont_freq_low_cols=[]

        binary_pharma_freq_hi_cols=[]
        binary_pharma_freq_med_cols=[]
        binary_pharma_freq_low_cols=[]

        cont_pharma_freq_hi_cols=[]
        cont_pharma_freq_med_cols=[]
        cont_pharma_freq_low_cols=[]

        for col in cont_var_cols:
            med_interval_mins=self.varparameters_dict[col]/60.0
            if med_interval_mins<=15.0:
                cont_freq_hi_cols.append(col)
            elif med_interval_mins>15.0 and med_interval_mins<=480.0:
                cont_freq_med_cols.append(col)
            else:
                cont_freq_low_cols.append(col)

        for col in cat_var_cols:
            med_interval_mins=self.varparameters_dict[col]/60.0
            if med_interval_mins<=15.0:
                cat_freq_hi_cols.append(col)
            elif med_interval_mins>15.0 and med_interval_mins<=480.0:
                cat_freq_med_cols.append(col)
            else:
                cat_freq_low_cols.append(col)           

        for col in binary_pharma_cols:
            act_period_mins=self.pharma_dict[col]
            if act_period_mins<=15.0:
                binary_pharma_freq_hi_cols.append(col)
            elif act_period_mins>15.0 and act_period_mins<=480.0:
                binary_pharma_freq_med_cols.append(col)
            else:
                binary_pharma_freq_low_cols.append(col)

        for col in cont_pharma_cols:
            act_period_mins=self.pharma_dict[col]
            if act_period_mins<=15.0:
                cont_pharma_freq_hi_cols.append(col)
            elif act_period_mins>15.0 and act_period_mins<=480.0:
                cont_pharma_freq_med_cols.append(col)
            else:
                cont_pharma_freq_low_cols.append(col)

        if df_pat.shape[0]==0 or df_label_pat.shape[0]==0:
            print("WARNING: Patient without information in labels or imputed data",flush=True)
            return (None,None)

        assert(df_pat.shape[0]==df_label_pat.shape[0])
        ts_col=np.array(df_pat["RelDatetime"])
        label_cols_dict={}

        for lcol in label_cols:
            label_cols_dict[lcol]=np.array(df_label_pat[lcol])

        status_cols_dict={}

        for lcol in label_cols:
            status_cols_dict[lcol]=[]

        interesting_cols=cont_pharma_cols+binary_pharma_cols+cont_var_cols+cat_var_cols+binary_var_cols
        nan_cols=[]

        # Detect NAN columns which should not be used for feature generation
        for col in interesting_cols:
            arr_col=np.array(df_pat[col])
            if np.sum(np.isnan(arr_col))>0:
                nan_cols.append(col)

        # Actually remove from the column type lists
        for col in nan_cols:
            if col in cont_pharma_cols:
                cont_pharma_cols.remove(col)
            elif col in binary_pharma_cols:
                binary_pharma_cols.remove(col)
            elif col in cont_var_cols:
                cont_var_cols.remove(col)
            elif col in cat_var_cols:
                cat_var_cols.remove(col)
            elif col in binary_var_cols:
                binary_var_cols.remove(col)

        n_feat_cols=20*len(cont_pharma_cols) + len(cont_pharma_cols) + \
                    5*len(binary_pharma_cols) + \
                    5*len(binary_var_cols) + \
                    20*len(cont_var_cols) + len(cont_var_cols) + \
                    5*len(cat_var_cols) + 84 + 14

        X=np.zeros((ts_col.size,n_feat_cols))
        sample_idx=0

        for idx,ts in np.ndenumerate(ts_col):
            row_idx=idx[0]

            if ts<start_ts:
                for lcol in label_cols:
                    status_cols_dict[lcol].append("INVALID_INITIAL")
            else:
                for lcol in label_cols:
                    if np.isnan(label_cols_dict[lcol][idx]):
                        status_cols_dict[lcol].append("INVALID_LABEL_NAN")
                    else:
                        status_cols_dict[lcol].append("VALID")

            # 1) Process continuous pharma variables
            df_cont_pharma_high_0=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_0_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_hi_cols])
            summary_cont_pharma_high_0=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_high_0)
            df_cont_pharma_high_1=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_1_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_hi_cols])
            summary_cont_pharma_high_1=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_high_1)
            df_cont_pharma_high_2=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_2_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_hi_cols])
            summary_cont_pharma_high_2=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_high_2)
            df_cont_pharma_high_3=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_3_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_hi_cols])
            summary_cont_pharma_high_3=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_high_3)
            df_cont_pharma_high_entire=np.array(df_pat.iloc[0:row_idx][cont_pharma_freq_hi_cols])
            summary_cont_pharma_high_entire=np.mean(df_cont_pharma_high_entire,axis=0)

            df_cont_pharma_med_0=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_med_cols])
            summary_cont_pharma_med_0=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_med_0)
            df_cont_pharma_med_1=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_med_cols])
            summary_cont_pharma_med_1=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_med_1)
            df_cont_pharma_med_2=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_med_cols])
            summary_cont_pharma_med_2=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_med_2)
            df_cont_pharma_med_3=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_med_cols])
            summary_cont_pharma_med_3=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_med_3)
            df_cont_pharma_med_entire=np.array(df_pat.iloc[0:row_idx][cont_pharma_freq_med_cols])
            summary_cont_pharma_med_entire=np.mean(df_cont_pharma_med_entire,axis=0)

            df_cont_pharma_low_0=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_0_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_low_cols])
            summary_cont_pharma_low_0=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_low_0)
            df_cont_pharma_low_1=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_1_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_low_cols])
            summary_cont_pharma_low_1=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_low_1)
            df_cont_pharma_low_2=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_2_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_low_cols])
            summary_cont_pharma_low_2=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_low_2)
            df_cont_pharma_low_3=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_3_mins/self.impute_grid_unit)):row_idx][cont_pharma_freq_low_cols])
            summary_cont_pharma_low_3=mlhc_array.time_window_5point_summary_non_robust(df_cont_pharma_low_3)
            df_cont_pharma_low_entire=np.array(df_pat.iloc[0:row_idx][cont_pharma_freq_low_cols])
            summary_cont_pharma_low_entire=np.mean(df_cont_pharma_low_entire,axis=0)

            # 2) Process binary pharma variables
            df_binary_hi_0=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_0_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_hi_cols])
            summary_binary_pharma_high_0=np.mean(df_binary_hi_0,axis=0)
            df_binary_hi_1=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_1_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_hi_cols])
            summary_binary_pharma_high_1=np.mean(df_binary_hi_1,axis=0)
            df_binary_hi_2=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_2_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_hi_cols])
            summary_binary_pharma_high_2=np.mean(df_binary_hi_2,axis=0)
            df_binary_hi_3=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_hi_3_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_hi_cols])
            summary_binary_pharma_high_3=np.mean(df_binary_hi_3,axis=0)
            df_binary_hi_entire=np.array(df_pat.iloc[0:row_idx][binary_pharma_freq_hi_cols])
            summary_binary_pharma_high_entire=np.mean(df_binary_hi_entire,axis=0)

            df_binary_med_0=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_med_cols])
            summary_binary_pharma_med_0=np.mean(df_binary_med_0,axis=0)
            df_binary_med_1=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_med_cols])
            summary_binary_pharma_med_1=np.mean(df_binary_med_1,axis=0)
            df_binary_med_2=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_med_cols])
            summary_binary_pharma_med_2=np.mean(df_binary_med_2,axis=0)
            df_binary_med_3=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_med_cols])
            summary_binary_pharma_med_3=np.mean(df_binary_med_3,axis=0)
            df_binary_med_entire=np.array(df_pat.iloc[0:row_idx][binary_pharma_freq_med_cols])
            summary_binary_pharma_med_entire=np.mean(df_binary_med_entire,axis=0)

            df_binary_low_0=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_0_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_low_cols])
            summary_binary_pharma_low_0=np.mean(df_binary_low_0,axis=0)
            df_binary_low_1=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_1_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_low_cols])
            summary_binary_pharma_low_1=np.mean(df_binary_low_1,axis=0)
            df_binary_low_2=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_2_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_low_cols])
            summary_binary_pharma_low_2=np.mean(df_binary_low_2,axis=0)
            df_binary_low_3=np.array(df_pat.iloc[max(0, row_idx-int(self.horizon_low_3_mins/self.impute_grid_unit)):row_idx][binary_pharma_freq_low_cols])
            summary_binary_pharma_low_3=np.mean(df_binary_low_3,axis=0)
            df_binary_low_entire=np.array(df_pat.iloc[0:row_idx][binary_pharma_freq_low_cols])
            summary_binary_pharma_low_entire=np.mean(df_binary_low_entire,axis=0)

            # 3) Binary non-pharma variables
            df_binary_lv_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_0_mins/self.impute_grid_unit)):row_idx][binary_var_cols])
            summary_binary_lv_0=np.mean(df_binary_lv_0,axis=0)
            df_binary_lv_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_1_mins/self.impute_grid_unit)):row_idx][binary_var_cols])
            summary_binary_lv_1=np.mean(df_binary_lv_1,axis=0)
            df_binary_lv_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_2_mins/self.impute_grid_unit)):row_idx][binary_var_cols])
            summary_binary_lv_2=np.mean(df_binary_lv_2,axis=0)
            df_binary_lv_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_3_mins/self.impute_grid_unit)):row_idx][binary_var_cols])
            summary_binary_lv_3=np.mean(df_binary_lv_3,axis=0)
            df_binary_lv_entire=np.array(df_pat.iloc[0:row_idx][binary_var_cols])
            summary_binary_lv_entire=np.mean(df_binary_lv_entire,axis=0)

            # 4) Process continuous lab/vital sign variables
            df_cont_high_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_0_mins/self.impute_grid_unit)):row_idx][cont_freq_hi_cols])
            cont_summary_high_0=mlhc_array.time_window_5point_summary(df_cont_high_0)
            df_cont_high_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_1_mins/self.impute_grid_unit)):row_idx][cont_freq_hi_cols])
            cont_summary_high_1=mlhc_array.time_window_5point_summary(df_cont_high_1)
            df_cont_high_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_2_mins/self.impute_grid_unit)):row_idx][cont_freq_hi_cols])
            cont_summary_high_2=mlhc_array.time_window_5point_summary(df_cont_high_2)
            df_cont_high_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_3_mins/self.impute_grid_unit)):row_idx][cont_freq_hi_cols])
            cont_summary_high_3=mlhc_array.time_window_5point_summary(df_cont_high_3)
            df_cont_high_entire=np.array(df_pat.iloc[0:row_idx][cont_freq_hi_cols])
            cont_summary_high_entire=np.median(df_cont_high_entire,axis=0)

            df_cont_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx][cont_freq_med_cols])
            cont_summary_med_0=mlhc_array.time_window_5point_summary(df_cont_med_0)
            df_cont_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx][cont_freq_med_cols])
            cont_summary_med_1=mlhc_array.time_window_5point_summary(df_cont_med_1)
            df_cont_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx][cont_freq_med_cols])
            cont_summary_med_2=mlhc_array.time_window_5point_summary(df_cont_med_2)
            df_cont_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx][cont_freq_med_cols])
            cont_summary_med_3=mlhc_array.time_window_5point_summary(df_cont_med_3)
            df_cont_med_entire=np.array(df_pat.iloc[0:row_idx][cont_freq_med_cols])
            cont_summary_med_entire=np.median(df_cont_med_entire,axis=0)

            df_cont_low_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_0_mins/self.impute_grid_unit)):row_idx][cont_freq_low_cols])
            cont_summary_low_0=mlhc_array.time_window_5point_summary(df_cont_low_0)
            df_cont_low_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_1_mins/self.impute_grid_unit)):row_idx][cont_freq_low_cols])
            cont_summary_low_1=mlhc_array.time_window_5point_summary(df_cont_low_1)
            df_cont_low_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_2_mins/self.impute_grid_unit)):row_idx][cont_freq_low_cols])
            cont_summary_low_2=mlhc_array.time_window_5point_summary(df_cont_low_2)
            df_cont_low_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_3_mins/self.impute_grid_unit)):row_idx][cont_freq_low_cols])
            cont_summary_low_3=mlhc_array.time_window_5point_summary(df_cont_low_3)
            df_cont_low_entire=np.array(df_pat.iloc[0:row_idx][cont_freq_low_cols])
            cont_summary_low_entire=np.median(df_cont_low_entire,axis=0)

            # 3) Process categorical lab/vital sign variables
            df_cat_high_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_0_mins/self.impute_grid_unit)):row_idx][cat_freq_hi_cols])
            df_cat_high_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_1_mins/self.impute_grid_unit)):row_idx][cat_freq_hi_cols])
            df_cat_high_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_2_mins/self.impute_grid_unit)):row_idx][cat_freq_hi_cols])
            df_cat_high_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_hi_3_mins/self.impute_grid_unit)):row_idx][cat_freq_hi_cols])
            df_cat_high_entire=np.array(df_pat.iloc[0:row_idx][cat_freq_hi_cols])

            if df_cat_high_0.size==0:
                cat_summary_high_0=np.zeros(df_cat_high_0.shape[1]).flatten()
                cat_summary_high_1=np.zeros(df_cat_high_1.shape[1]).flatten()
                cat_summary_high_2=np.zeros(df_cat_high_2.shape[1]).flatten()
                cat_summary_high_3=np.zeros(df_cat_high_3.shape[1]).flatten()
                cat_summary_high_entire=np.zeros(df_cat_high_entire.shape[1]).flatten()
            else:
                cat_summary_high_0=sp_stats.mode(df_cat_high_0,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_high_1=sp_stats.mode(df_cat_high_1,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_high_2=sp_stats.mode(df_cat_high_2,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_high_3=sp_stats.mode(df_cat_high_3,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_high_entire=sp_stats.mode(df_cat_high_entire,axis=0,nan_policy="omit")[0].flatten()

            df_cat_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx][cat_freq_med_cols])
            df_cat_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx][cat_freq_med_cols])
            df_cat_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx][cat_freq_med_cols])
            df_cat_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx][cat_freq_med_cols])
            df_cat_med_entire=np.array(df_pat.iloc[0:row_idx][cat_freq_med_cols])

            if df_cat_med_0.size==0:
                cat_summary_med_0=np.zeros(df_cat_med_0.shape[1]).flatten()
                cat_summary_med_1=np.zeros(df_cat_med_1.shape[1]).flatten()
                cat_summary_med_2=np.zeros(df_cat_med_2.shape[1]).flatten()
                cat_summary_med_3=np.zeros(df_cat_med_3.shape[1]).flatten()
                cat_summary_med_entire=np.zeros(df_cat_med_entire.shape[1]).flatten()
            else:
                cat_summary_med_0=sp_stats.mode(df_cat_med_0,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_med_1=sp_stats.mode(df_cat_med_1,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_med_2=sp_stats.mode(df_cat_med_2,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_med_3=sp_stats.mode(df_cat_med_3,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_med_entire=sp_stats.mode(df_cat_med_entire,axis=0, nan_policy="omit")[0].flatten()

            df_cat_low_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_0_mins/self.impute_grid_unit)):row_idx][cat_freq_low_cols])
            df_cat_low_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_1_mins/self.impute_grid_unit)):row_idx][cat_freq_low_cols])
            df_cat_low_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_2_mins/self.impute_grid_unit)):row_idx][cat_freq_low_cols])
            df_cat_low_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_low_3_mins/self.impute_grid_unit)):row_idx][cat_freq_low_cols])
            df_cat_low_entire=np.array(df_pat.iloc[0:row_idx][cat_freq_low_cols])

            if df_cat_low_0.size==0:
                cat_summary_low_0=np.zeros(df_cat_low_0.shape[1]).flatten()
                cat_summary_low_1=np.zeros(df_cat_low_1.shape[1]).flatten()
                cat_summary_low_2=np.zeros(df_cat_low_2.shape[1]).flatten()
                cat_summary_low_3=np.zeros(df_cat_low_3.shape[1]).flatten()
                cat_summary_low_entire=np.zeros(df_cat_low_entire.shape[1]).flatten()
            else:
                cat_summary_low_0=sp_stats.mode(df_cat_low_0,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_low_1=sp_stats.mode(df_cat_low_1,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_low_2=sp_stats.mode(df_cat_low_2,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_low_3=sp_stats.mode(df_cat_low_3,axis=0, nan_policy="omit")[0].flatten()
                cat_summary_low_entire=sp_stats.mode(df_cat_low_entire,axis=0,nan_policy="omit")[0].flatten()


            # Build some special pseudo-variables which summarize the entire stay up to now (endpoint components)
            if self.dim_reduced_data:
                
                # Mean arterial pressure
                map_all=np.array(df_pat.iloc[0:row_idx]["vm5"])
                map_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx]["vm5"])
                map_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx]["vm5"])
                map_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx]["vm5"])
                map_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx]["vm5"])
                map_occs=np.sum(map_all<=65)

                map_feat_all=np.sum(map_all<=65)/map_all.size
                map_feat_med_0=np.sum(map_med_0<=65)/map_med_0.size
                map_feat_med_1=np.sum(map_med_1<=65)/map_med_1.size
                map_feat_med_2=np.sum(map_med_2<=65)/map_med_2.size
                map_feat_med_3=np.sum(map_med_3<=65)/map_med_3.size

                map_to=SYMBOLIC_MAX if map_occs==0 else (map_all.size-np.nonzero(map_all<=65)[0][0])*5

                # Lactate
                alac_all=np.array(df_pat.iloc[0:row_idx]["vm136"])
                alac_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx]["vm136"])
                alac_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx]["vm136"])
                alac_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx]["vm136"])
                alac_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx]["vm136"])

                vlac_key="vm146" if self.dataset=="bern" else "vm136" # Closest analogue VM136, still feature should be probably ignored..
                vlac_all=np.array(df_pat.iloc[0:row_idx][vlac_key])
                vlac_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx][vlac_key])
                vlac_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx][vlac_key])
                vlac_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx][vlac_key])
                vlac_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx][vlac_key])

                lac_occs=np.sum((alac_all>=2) | (vlac_all>=2))

                lac_feat_all=np.sum((alac_all>=2) | (vlac_all>=2))/alac_all.size
                lac_feat_med_0=np.sum((alac_med_0>=2) | (vlac_med_0>=2))/alac_med_0.size
                lac_feat_med_1=np.sum((alac_med_1>=2) | (vlac_med_1>=2))/alac_med_1.size
                lac_feat_med_2=np.sum((alac_med_2>=2) | (vlac_med_2>=2))/alac_med_2.size
                lac_feat_med_3=np.sum((alac_med_3>=2) | (vlac_med_3>=2))/alac_med_3.size

                lac_to=SYMBOLIC_MAX if lac_occs==0 else (alac_all.size-np.nonzero((alac_all>=2) | (vlac_all>=2))[0][0])*5

                # Dobutamine
                dop_all=np.array(df_pat.iloc[0:row_idx]["pm41"])
                dop_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx]["pm41"])
                dop_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx]["pm41"])
                dop_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx]["pm41"])
                dop_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx]["pm41"])  

                dop_occs=np.sum(dop_all>0)

                dop_feat_all=np.sum(dop_all>0)/dop_all.size
                dop_feat_med_0=np.sum(dop_med_0>0)/dop_med_0.size
                dop_feat_med_1=np.sum(dop_med_1>0)/dop_med_1.size
                dop_feat_med_2=np.sum(dop_med_2>0)/dop_med_2.size
                dop_feat_med_3=np.sum(dop_med_3>0)/dop_med_3.size

                dop_to=SYMBOLIC_MAX if dop_occs==0 else (dop_all.size-np.nonzero(dop_all>0)[0][0])*5

                # Milrinone
                mil_all=np.array(df_pat.iloc[0:row_idx]["pm42"])
                mil_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx]["pm42"])
                mil_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx]["pm42"])
                mil_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx]["pm42"])
                mil_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx]["pm42"])

                mil_occs=np.sum(mil_all>0)

                mil_feat_all=np.sum(mil_all>0)/mil_all.size
                mil_feat_med_0=np.sum(mil_med_0>0)/mil_med_0.size
                mil_feat_med_1=np.sum(mil_med_1>0)/mil_med_1.size
                mil_feat_med_2=np.sum(mil_med_2>0)/mil_med_2.size
                mil_feat_med_3=np.sum(mil_med_3>0)/mil_med_3.size
 
                mil_to=SYMBOLIC_MAX if mil_occs==0 else (mil_all.size-np.nonzero(mil_all>0)[0][0])*5

                # Levesemendin
                lev_key="pm43" if self.dataset=="bern" else "pm41" # Closest analogue PM41, still feature should be probably ignored...
                lev_all=np.array(df_pat.iloc[0:row_idx][lev_key])
                lev_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx][lev_key])
                lev_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx][lev_key])
                lev_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx][lev_key])
                lev_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx][lev_key])

                lev_occs=np.sum(lev_all>0)

                lev_feat_all=np.sum(lev_all>0)/lev_all.size
                lev_feat_med_0=np.sum(lev_med_0>0)/lev_med_0.size
                lev_feat_med_1=np.sum(lev_med_1>0)/lev_med_1.size
                lev_feat_med_2=np.sum(lev_med_2>0)/lev_med_2.size
                lev_feat_med_3=np.sum(lev_med_3>0)/lev_med_3.size

                lev_to=SYMBOLIC_MAX if lev_occs==0 else (lev_all.size-np.nonzero(lev_all>0)[0][0])*5

                # Theophyllin
                theo_key="pm44" if self.dataset=="bern" else "pm41" # Closest analogue PM41, still feature should be probably ignored..
                theo_all=np.array(df_pat.iloc[0:row_idx][theo_key])
                theo_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx][theo_key])
                theo_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx][theo_key])
                theo_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx][theo_key])
                theo_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx][theo_key])

                theo_occs=np.sum(theo_all>0)

                theo_feat_all=np.sum(theo_all>0)/theo_all.size
                theo_feat_med_0=np.sum(theo_med_0>0)/theo_med_0.size
                theo_feat_med_1=np.sum(theo_med_1>0)/theo_med_1.size
                theo_feat_med_2=np.sum(theo_med_2>0)/theo_med_2.size
                theo_feat_med_3=np.sum(theo_med_3>0)/theo_med_3.size

                theo_to=SYMBOLIC_MAX if theo_occs==0 else (theo_all.size-np.nonzero(theo_all>0)[0][0])*5
                
                # Event 1
                event1_occs=np.sum( ((vlac_all>=2) | (alac_all>=2)) & ((map_all<=65) | (dop_all>0) | (mil_all>0) | (lev_all>0) | (theo_all>0)) )

                event1_feat_all=  np.sum( ((vlac_all>=2)   | (alac_all>=2))   & ((map_all<=65)   | (dop_all>0)   | (mil_all>0)   | (lev_all>0)   | (theo_all>0))   )/dop_all.size
                event1_feat_med_0=np.sum( ((vlac_med_0>=2) | (alac_med_0>=2)) & ((map_med_0<=65) | (dop_med_0>0) | (mil_med_0>0) | (lev_med_0>0) | (theo_med_0>0)) )/dop_med_0.size
                event1_feat_med_1=np.sum( ((vlac_med_1>=2) | (alac_med_1>=2)) & ((map_med_1<=65) | (dop_med_1>0) | (mil_med_1>0) | (lev_med_1>0) | (theo_med_1>0)) )/dop_med_1.size
                event1_feat_med_2=np.sum( ((vlac_med_2>=2) | (alac_med_2>=2)) & ((map_med_2<=65) | (dop_med_2>0) | (mil_med_2>0) | (lev_med_2>0) | (theo_med_2>0)) )/dop_med_2.size
                event1_feat_med_3=np.sum( ((vlac_med_3>=2) | (alac_med_3>=2)) & ((map_med_3<=65) | (dop_med_3>0) | (mil_med_3>0) | (lev_med_3>0) | (theo_med_3>0)) )/dop_med_3.size

                event1_to=SYMBOLIC_MAX if event1_occs==0 else (dop_all.size-np.nonzero( ((vlac_all>=2) | (alac_all>=2)) & ((map_all<=65) | (dop_all>0) | (mil_all>0) | (lev_all>0) | (theo_all>0)) )[0][0])*5

                # Norephenephrine
                noreph_all=np.array(df_pat.iloc[0:row_idx]["pm39"])
                noreph_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx]["pm39"])
                noreph_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx]["pm39"])
                noreph_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx]["pm39"])
                noreph_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx]["pm39"])

                noreph_stage1_occs=np.sum((noreph_all>0) & (noreph_all<0.1))
                noreph_stage2_occs=np.sum(noreph_all>=0.1)

                noreph_feat_all=np.sum((noreph_all>0) & (noreph_all<0.1))/noreph_all.size
                noreph_feat_med_0=np.sum((noreph_med_0>0) & (noreph_med_0<0.1))/noreph_med_0.size
                noreph_feat_med_1=np.sum((noreph_med_1>0) & (noreph_med_1<0.1))/noreph_med_1.size
                noreph_feat_med_2=np.sum((noreph_med_2>0) & (noreph_med_2<0.1))/noreph_med_2.size
                noreph_feat_med_3=np.sum((noreph_med_3>0) & (noreph_med_3<0.1))/noreph_med_3.size

                noreph_feat2_all=np.sum(noreph_all>=0.1)/noreph_all.size
                noreph_feat2_med_0=np.sum(noreph_med_0>=0.1)/noreph_med_0.size
                noreph_feat2_med_1=np.sum(noreph_med_1>=0.1)/noreph_med_1.size
                noreph_feat2_med_2=np.sum(noreph_med_2>=0.1)/noreph_med_2.size
                noreph_feat2_med_3=np.sum(noreph_med_3>=0.1)/noreph_med_3.size

                noreph_stage1_to=SYMBOLIC_MAX if noreph_stage1_occs==0 else (noreph_all.size-np.nonzero((noreph_all>0) & (noreph_all<0.1))[0][0])*5
                noreph_stage2_to=SYMBOLIC_MAX if noreph_stage2_occs==0 else (noreph_all.size-np.nonzero(noreph_all>=0.1)[0][0])*5

                # Epinephrine
                epineph_all=np.array(df_pat.iloc[0:row_idx]["pm40"])
                epineph_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx]["pm40"])
                epineph_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx]["pm40"])
                epineph_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx]["pm40"])
                epineph_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx]["pm40"])

                epineph_stage1_occs=np.sum((epineph_all>0) & (epineph_all<0.1))
                epineph_stage2_occs=np.sum(epineph_all>=0.1)

                epineph_feat_all=np.sum((epineph_all>0) & (epineph_all<0.1))/epineph_all.size
                epineph_feat_med_0=np.sum((epineph_med_0>0) & (epineph_med_0<0.1))/epineph_med_0.size
                epineph_feat_med_1=np.sum((epineph_med_1>0) & (epineph_med_1<0.1))/epineph_med_1.size
                epineph_feat_med_2=np.sum((epineph_med_2>0) & (epineph_med_2<0.1))/epineph_med_2.size
                epineph_feat_med_3=np.sum((epineph_med_3>0) & (epineph_med_3<0.1))/epineph_med_3.size

                epineph_feat2_all=np.sum(epineph_all>=0.1)/epineph_all.size
                epineph_feat2_med_0=np.sum(epineph_med_0>=0.1)/epineph_med_0.size
                epineph_feat2_med_1=np.sum(epineph_med_1>=0.1)/epineph_med_1.size
                epineph_feat2_med_2=np.sum(epineph_med_2>=0.1)/epineph_med_2.size
                epineph_feat2_med_3=np.sum(epineph_med_3>=0.1)/epineph_med_3.size

                epineph_stage1_to=SYMBOLIC_MAX if epineph_stage1_occs==0 else (epineph_all.size-np.nonzero((epineph_all>0) & (epineph_all<0.1))[0][0])*5
                epineph_stage2_to=SYMBOLIC_MAX if epineph_stage2_occs==0 else (epineph_all.size-np.nonzero(epineph_all>=0.1)[0][0])*5

                # Event 2
                event2_occs=np.sum( ((alac_all>=2) | (vlac_all>=2)) & ((noreph_all>0) & (noreph_all<0.1) | (epineph_all>0) & (epineph_all<0.1)) )

                event2_feat_all=np.sum(  ((alac_all>=2) | (vlac_all>=2)) & ((noreph_all>0) & (noreph_all<0.1) | (epineph_all>0) & (epineph_all<0.1)) )/noreph_all.size
                event2_feat_med_0=np.sum( ((alac_med_0>=2) | (vlac_med_0>=2)) & ((noreph_med_0>0) & (noreph_med_0<0.1) | (epineph_med_0>0) & (epineph_med_0<0.1)) )/noreph_med_0.size
                event2_feat_med_1=np.sum( ((alac_med_1>=2) | (vlac_med_1>=2)) & ((noreph_med_1>0) & (noreph_med_1<0.1) | (epineph_med_1>0) & (epineph_med_1<0.1)) )/noreph_med_1.size
                event2_feat_med_2=np.sum( ((alac_med_2>=2) | (vlac_med_2>=2)) & ((noreph_med_2>0) & (noreph_med_2<0.1) | (epineph_med_2>0) & (epineph_med_2<0.1)) )/noreph_med_2.size
                event2_feat_med_3=np.sum( ((alac_med_3>=2) | (vlac_med_3>=2)) & ((noreph_med_3>0) & (noreph_med_3<0.1) | (epineph_med_3>0) & (epineph_med_3<0.1)) )/noreph_med_3.size

                event2_to=SYMBOLIC_MAX if event2_occs==0 else (dop_all.size-np.nonzero(  ((alac_all>=2) | (vlac_all>=2)) & ((noreph_all>0) & (noreph_all<0.1) | (epineph_all>0) & (epineph_all<0.1)) )[0][0])*5

                # Vasopressin
                vaso_all=np.array(df_pat.iloc[0:row_idx]["pm45"])
                vaso_med_0=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_0_mins/self.impute_grid_unit)):row_idx]["pm45"])
                vaso_med_1=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_1_mins/self.impute_grid_unit)):row_idx]["pm45"])
                vaso_med_2=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_2_mins/self.impute_grid_unit)):row_idx]["pm45"])
                vaso_med_3=np.array(df_pat.iloc[max(0,row_idx-int(self.horizon_med_3_mins/self.impute_grid_unit)):row_idx]["pm45"])

                vaso_occs=np.sum(vaso_all>0)

                vaso_feat_all=np.sum(vaso_all>0)/vaso_all.size
                vaso_feat_med_0=np.sum(vaso_med_0>0)/vaso_med_0.size
                vaso_feat_med_1=np.sum(vaso_med_1>0)/vaso_med_1.size
                vaso_feat_med_2=np.sum(vaso_med_2>0)/vaso_med_2.size
                vaso_feat_med_3=np.sum(vaso_med_3>0)/vaso_med_3.size

                vaso_to=SYMBOLIC_MAX if vaso_occs==0 else (vaso_all.size-np.nonzero(vaso_all>0)[0][0])*5

                # Event 3
                event3_occs=np.sum( ((alac_all>=2) | (vlac_all>=2)) & ((noreph_all>=0.1) | (epineph_all>=0.1) | (vaso_all>0)) )

                event3_feat_all=np.sum( ((alac_all>=2) | (vlac_all>=2)) & ((noreph_all>=0.1) | (epineph_all>=0.1) | (vaso_all>0)) )/noreph_all.size
                event3_feat_med_0=np.sum( ((alac_med_0>=2) | (vlac_med_0>=2)) & ((noreph_med_0>=0.1) | (epineph_med_0>=0.1) | (vaso_med_0>0)) )/noreph_med_0.size
                event3_feat_med_1=np.sum( ((alac_med_1>=2) | (vlac_med_1>=2)) & ((noreph_med_1>=0.1) | (epineph_med_1>=0.1) | (vaso_med_1>0)) )/noreph_med_1.size
                event3_feat_med_2=np.sum( ((alac_med_2>=2) | (vlac_med_2>=2)) & ((noreph_med_2>=0.1) | (epineph_med_2>=0.1) | (vaso_med_2>0)) )/noreph_med_2.size
                event3_feat_med_3=np.sum( ((alac_med_3>=2) | (vlac_med_3>=2)) & ((noreph_med_3>=0.1) | (epineph_med_3>=0.1) | (vaso_med_3>0)) )/noreph_med_3.size

                event3_to=SYMBOLIC_MAX if event3_occs==0 else (dop_all.size-np.nonzero( ((alac_all>=2) | (vlac_all>=2)) & ((noreph_all>=0.1) | (epineph_all>=0.1) | (vaso_all>0)) )[0][0])*5

                all_stay_arr=np.array([map_feat_all, map_feat_med_0, map_feat_med_1, map_feat_med_2, map_feat_med_3, map_to,
                                       lac_feat_all, lac_feat_med_0, lac_feat_med_1, lac_feat_med_2, lac_feat_med_3, lac_to, 
                                       dop_feat_all, dop_feat_med_0, dop_feat_med_1, dop_feat_med_2, dop_feat_med_3, dop_to,
                                       mil_feat_all, mil_feat_med_0, mil_feat_med_1, mil_feat_med_2, mil_feat_med_3, mil_to,
                                       lev_feat_all, lev_feat_med_0, lev_feat_med_1, lev_feat_med_2, lev_feat_med_3, lev_to,
                                       theo_feat_all, theo_feat_med_0, theo_feat_med_1, theo_feat_med_2, theo_feat_med_3, theo_to,
                                       event1_feat_all, event1_feat_med_0, event1_feat_med_1, event1_feat_med_2, event1_feat_med_3, event1_to,
                                       noreph_feat_all, noreph_feat_med_0, noreph_feat_med_1, noreph_feat_med_2, noreph_feat_med_3, noreph_stage1_to,
                                       noreph_feat2_all, noreph_feat2_med_0, noreph_feat2_med_1, noreph_feat2_med_2, noreph_feat2_med_3, noreph_stage2_to, 
                                       epineph_feat_all, epineph_feat_med_0, epineph_feat_med_1, epineph_feat_med_2, epineph_feat_med_3, epineph_stage1_to,
                                       epineph_feat2_all, epineph_feat2_med_0, epineph_feat2_med_1, epineph_feat2_med_2, epineph_feat2_med_3, epineph_stage2_to,
                                       event2_feat_all, event2_feat_med_0, event2_feat_med_1, event2_feat_med_2, event2_feat_med_3, event2_to, 
                                       vaso_feat_all, vaso_feat_med_0, vaso_feat_med_1, vaso_feat_med_2, vaso_feat_med_3, vaso_to,
                                       event3_feat_all, event3_feat_med_0, event3_feat_med_1, event3_feat_med_2, event3_feat_med_3, event3_to])
                
            else:
                assert(False) # CODE PATH NOT USED

            if self.dim_reduced_data:
                map_cur=np.array(df_pat.iloc[row_idx]["vm5"])
                map_feat_cur=np.sum(map_cur<=65)
                alac_cur=np.array(df_pat.iloc[row_idx]["vm136"])
                vlac_cur=np.array(df_pat.iloc[row_idx][vlac_key])
                lac_feat_cur=np.sum((alac_cur>=2) | (vlac_cur>=2))
                dop_cur=np.array(df_pat.iloc[row_idx]["pm41"])
                dop_feat_cur=np.sum(dop_cur>0)
                mil_cur=np.array(df_pat.iloc[row_idx]["pm42"])
                mil_feat_cur=np.sum(mil_cur>0)
                lev_cur=np.array(df_pat.iloc[row_idx][lev_key])
                lev_feat_cur=np.sum(lev_cur>0)
                theo_cur=np.array(df_pat.iloc[row_idx][theo_key])
                theo_feat_cur=np.sum(theo_cur>0)
                event1_feat_cur=np.sum( ((alac_cur>=2) | (vlac_cur>=2)) & ((map_cur<=65) | (dop_cur>0) | (mil_cur>0) | (lev_cur>0) | (theo_cur>0)) )
                noreph_cur=np.array(df_pat.iloc[row_idx]["pm39"])
                noreph_feat_cur=np.sum((noreph_cur>0) & (noreph_cur<0.1))
                noreph_feat2_cur=np.sum(noreph_cur>=0.1)
                epineph_cur=np.array(df_pat.iloc[row_idx]["pm40"])
                epineph_feat_cur=np.sum((epineph_cur>0) & (epineph_cur<0.1))
                epineph_feat2_cur=np.sum(epineph_cur>=0.1)
                event2_feat_cur=np.sum( ((alac_cur>=2) | (vlac_cur>=2)) & ((noreph_cur>0) & (noreph_cur<0.1) | (epineph_cur>0) & (epineph_cur<0.1)) )
                vaso_cur=np.array(df_pat.iloc[row_idx]["pm45"])
                vaso_feat_cur=np.sum(vaso_cur>0)
                event3_feat_cur=np.sum( ((alac_cur>=2) | (vlac_cur>=2)) & ((noreph_cur>=0.1) | (epineph_cur>=0.1) | (vaso_cur>0)) )

                current_stay_arr=np.array([map_feat_cur, lac_feat_cur, dop_feat_cur, mil_feat_cur, lev_feat_cur, theo_feat_cur, 
                                           event1_feat_cur, noreph_feat_cur, noreph_feat2_cur, 
                                           epineph_feat_cur, epineph_feat2_cur, event2_feat_cur,
                                           vaso_feat_cur, event3_feat_cur])

            else:
                assert(False) # CODE PATH NOT USED


            if self.dim_reduced_data:
                averaged_sample=np.concatenate([summary_cont_pharma_high_0, summary_cont_pharma_high_1, summary_cont_pharma_high_2, summary_cont_pharma_high_3, summary_cont_pharma_high_entire,
                                                summary_cont_pharma_med_0, summary_cont_pharma_med_1, summary_cont_pharma_med_2, summary_cont_pharma_med_3, summary_cont_pharma_med_entire,
                                                summary_cont_pharma_low_0, summary_cont_pharma_low_1, summary_cont_pharma_low_2, summary_cont_pharma_low_3, summary_cont_pharma_low_entire, 
                                                summary_binary_pharma_high_0, summary_binary_pharma_high_1, summary_binary_pharma_high_2, summary_binary_pharma_high_3, summary_binary_pharma_high_entire,
                                                summary_binary_pharma_med_0, summary_binary_pharma_med_1, summary_binary_pharma_med_2, summary_binary_pharma_med_3, summary_binary_pharma_med_entire,
                                                summary_binary_pharma_low_0, summary_binary_pharma_low_1, summary_binary_pharma_low_2, summary_binary_pharma_low_3, summary_binary_pharma_low_entire, 
                                                summary_binary_lv_0, summary_binary_lv_1, summary_binary_lv_2, summary_binary_lv_3, summary_binary_lv_entire, 
                                                cont_summary_high_0, cont_summary_high_1, cont_summary_high_2, cont_summary_high_3, cont_summary_high_entire, 
                                                cont_summary_med_0, cont_summary_med_1, cont_summary_med_2, cont_summary_med_3, cont_summary_med_entire, 
                                                cont_summary_low_0, cont_summary_low_1, cont_summary_low_2, cont_summary_low_3, cont_summary_low_entire,
                                                cat_summary_high_0, cat_summary_high_1, cat_summary_high_2, cat_summary_high_3, cat_summary_high_entire, 
                                                cat_summary_med_0, cat_summary_med_1, cat_summary_med_2, cat_summary_med_3, cat_summary_med_entire, 
                                                cat_summary_low_0, cat_summary_low_1, cat_summary_low_2, cat_summary_low_3, cat_summary_low_entire, 
                                                all_stay_arr, current_stay_arr]) 


                if self.dataset=="bern":
                    assert(len(averaged_sample)==3259)
                
            else: # UNUSED CODE PATH
                assert(False)

            if row_idx==0:
                averaged_sample[np.isnan(averaged_sample)]=0.0
            else:
                assert(np.isfinite(averaged_sample).all())

            X[sample_idx,:]=averaged_sample
            sample_idx+=1

        X_df_dict={}
        X_df_dict["RelDatetime"]=rel_dt_col
        X_df_dict["AbsDatetime"]=abs_dt_col
        X_df_dict["PatientID"]=pid_col

        for col in interesting_cols:
            if col not in nan_cols:
                X_df_dict["plain_{}".format(col)]=df_pat[col]

        for col in interesting_cols:
            if col not in nan_cols and "pm" not in col:
                im_array=np.array(df_pat["{}_IMPUTED_STATUS_TIME_TO".format(col)])
                im_array[im_array==-1.0]=SYMBOLIC_MAX
                X_df_dict["{}_time_to_last_ms".format(col)]=im_array
                X_df_dict["{}_measure_density".format(col)]=np.divide(np.array(df_pat["{}_IMPUTED_STATUS_CUM_COUNT".format(col)]),
                                                                      5*np.arange(1,rel_dt_col.size+1))

        for lcol in label_cols:
            X_df_dict["SampleStatus_{}".format(lcol)]=status_cols_dict[lcol]

        X_df_col_headers=[]

        # 1) Continuous pharma
        for block,block_cols in [("high",cont_pharma_freq_hi_cols), ("med",cont_pharma_freq_med_cols), ("low",cont_pharma_freq_low_cols)]:
            for level in [0,1,2,3]:
                for feat_idx, feat_name in enumerate(["mean", "std", "min", "max", "trend"]):
                    for inner_idx,col_name in enumerate(block_cols):
                        X_df_col_headers.append("{}_{}_{}_{}".format(block, level, col_name,feat_name))
            for level in ["entire"]:
                for feat_idx, feat_name in enumerate(["mean"]):
                    for inner_idx,col_name in enumerate(block_cols):
                        X_df_col_headers.append("{}_{}_{}_{}".format(block, level, col_name,feat_name))

        # 1.1) Binary pharma
        for block,block_cols in [("high",binary_pharma_freq_hi_cols), ("med",binary_pharma_freq_med_cols), ("low",binary_pharma_freq_low_cols)]:
            for level in [0,1,2,3,"entire"]:
                for feat_idx, feat_name in enumerate(["mean"]):
                    for inner_idx,col_name in enumerate(block_cols):
                        X_df_col_headers.append("{}_{}_{}_{}".format(block, level, col_name,feat_name))

        # 1.2) Binary lab/vital 
        for level in [0,1,2,3,"entire"]:
            for feat_idx, feat_name in enumerate(["mean"]):
                for inner_idx,col_name in enumerate(binary_var_cols):
                    X_df_col_headers.append("{}_{}".format(col_name,feat_name))

        # 2) Continuous lab/vital
        for block,block_cols in [("high",cont_freq_hi_cols), ("med",cont_freq_med_cols), ("low",cont_freq_low_cols)]:
            for level in [0,1,2,3]:
                for feat_idx, feat_name in enumerate(["median", "iqr", "min", "max", "trend"]):
                    for inner_idx,col_name in enumerate(block_cols):
                        X_df_col_headers.append("{}_{}_{}_{}".format(block, level, col_name,feat_name))
            for level in ["entire"]:
                for feat_idx, feat_name in enumerate(["median"]):
                    for inner_idx,col_name in enumerate(block_cols):
                        X_df_col_headers.append("{}_{}_{}_{}".format(block, level, col_name,feat_name))

        # 3) Categorical lab/vital
        for block,block_cols in [("high",cat_freq_hi_cols), ("med",cat_freq_med_cols), ("low",cat_freq_low_cols)]:
            for level in [0,1,2,3,"entire"]:
                for feat_idx, feat_name in enumerate(["mode"]):
                    for inner_idx,col_name in enumerate(block_cols):
                        X_df_col_headers.append("{}_{}_{}_{}".format(block, level, col_name,feat_name))


        X_df_col_headers.extend(["map_event_ratio_entire", "map_event_ratio_0", "map_event_ratio_1", "map_event_ratio_2", "map_event_ratio_3", "map_event_time",
                                 "lac_event_ratio_entire", "lac_event_ratio_0", "lac_event_ratio_1", "lac_event_ratio_2", "lac_event_ratio_3", "lac_event_time",
                                 "dop_event_ratio_entire", "dop_event_ratio_0", "dop_event_ratio_1", "dop_event_ratio_2", "dop_event_ratio_3","dop_event_time",
                                 "mil_event_ratio_entire", "mil_event_ratio_0", "mil_event_ratio_1", "mil_event_ratio_2", "mil_event_ratio_3","mil_event_time", 
                                 "lev_event_ratio_entire", "lev_event_ratio_0", "lev_event_ratio_1", "lev_event_ratio_2", "lev_event_ratio_3", "lev_event_time", 
                                 "theo_event_ratio_entire", "theo_event_ratio_0", "theo_event_ratio_1", "theo_event_ratio_2", "theo_event_ratio_3", "theo_event_time",
                                 "event1_ratio_entire", "event1_ratio_0", "event1_ratio_1", "event1_ratio_2", "event1_ratio_3", "event1_time", 
                                 "noreph_l1_ratio_entire","noreph_l1_ratio_0", "noreph_l1_ratio_1", "noreph_l1_ratio_2", "noreph_l1_ratio_3", "noreph_l1_time",
                                 "noreph_l2_ratio_entire", "noreph_l2_ratio_0", "noreph_l2_ratio_1", "noreph_l2_ratio_2", "noreph_l2_ratio_3","noreph_l2_time", 
                                 "epineph_l1_ratio_entire", "epineph_l1_ratio_0", "epineph_l1_ratio_1", "epineph_l1_ratio_2", "epineph_l1_ratio_3","epineph_l1_time", 
                                 "epineph_l2_ratio_entire", "epineph_l2_ratio_0", "epineph_l2_ratio_1", "epineph_l2_ratio_2", "epineph_l2_ratio_3", "epineph_l2_time", 
                                 "event2_ratio_entire","event2_ratio_0", "event2_ratio_1", "event2_ratio_2", "event2_ratio_3",  "event2_time",
                                 "vaso_event_ratio_entire","vaso_event_ratio_0", "vaso_event_ratio_1", "vaso_event_ratio_2", "vaso_event_ratio_3",  "vaso_event_time", 
                                 "event3_ratio","event3_ratio_0", "event3_ratio_1", "event3_ratio_2", "event3_ratio_3",  "event3_time"])

        X_df_col_headers.extend(["map_event_cur", "lac_event_cur", "dop_event_cur", "mil_event_cur", "lev_event_cur", "theo_event_cur",
                                 "event1_cur", "noreph_l1_cur", "noreph_l2_cur", "epineph_l1_cur", "epineph_l2_cur", "event2_cur", "vaso_event_cur", 
                                 "event3_cur"])
        
        assert(len(X_df_col_headers)==X.shape[1])

        for idx in range(len(X_df_col_headers)):
            X_df_dict[X_df_col_headers[idx]]=X[:,idx]

        gc.collect()
        X_df=pd.DataFrame(X_df_dict)

        y_df_dict={}
        y_df_dict["RelDatetime"]=rel_dt_col
        y_df_dict["AbsDatetime"]=abs_dt_col
        y_df_dict["PatientID"]=pid_col

        for lcol in label_cols:
            y_df_dict["SampleStatus_{}".format(lcol)]=status_cols_dict[lcol]
            y_df_dict["Label_{}".format(lcol)]=label_cols_dict[lcol]

        y_df=pd.DataFrame(y_df_dict)

        return (X_df, y_df)

        
    
