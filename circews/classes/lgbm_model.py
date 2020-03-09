
import random
import os
import ipdb
import gc
import sys
import math
import timeit
import warnings

import scipy as sp
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
import sklearn.feature_selection as skfselect
import sklearn.preprocessing as skpproc
import sklearn.impute as skimpute
import sklearn.compose as skcompose
import sklearn.linear_model as sklm
import sklearn.neural_network as sknn
import shap
import numpy.random as nprand
import lightgbm
import xgboost
import catboost

import mlhc_data_manager.util.array as mlhc_array

def custom_auprc_metric(y_true,y_pred):
    return ("custom_auprc", skmetrics.average_precision_score(y_true,y_pred), True)

class LightGbmModel:

    def __init__(self, task_key=None, time_key=None, train_subsample=False,
                 n_estimators=None, num_leaves=None, learning_rate=None, colsample_bytree=None, rowsample=None,
                 var_desc=None, no_datetime=False, select_features_forward=False, select_features_backward=False,
                 univariate_test=False, use_xgboost=False, use_catboost=False, dataset=None,
                 only_decision_tree=False, serial_mode=False,
                 decision_tree_mode=False, logreg_mode=False, mlp_mode=False):
        self.std_scale_data=False
        self.filter_low_variance=False
        self.std_eps=1e-5
        self.task_key=task_key
        self.time_key=time_key
        self.train_subsample=train_subsample
        self.var_desc=var_desc
        self.use_xgboost=use_xgboost
        self.use_catboost=use_catboost
        self.dataset=dataset
        self.only_decision_tree=only_decision_tree
        self.serial_mode=serial_mode

        self.check_order=None
        self.sel_cols_X=None

        self.decision_tree_mode=decision_tree_mode
        self.logreg_mode=logreg_mode
        self.mlp_mode=mlp_mode
        
        self.fit_first=True

        self.no_datetime=no_datetime

        self.select_features_forward=select_features_forward
        self.select_features_backward=select_features_backward
        self.univariate_test=univariate_test

        # LGBM hyperparameters
        self.n_estimators=n_estimators
        self.num_leaves=num_leaves
        self.learning_rate=learning_rate
        self.colsample_bytree=colsample_bytree
        self.rowsample=rowsample

        # Training data
        self.collect_X=[]
        self.collect_y=[]

        # Validation data
        self.collect_X_val=[]
        self.collect_y_val=[]

        if self.use_xgboost and not self.serial_mode:
            self.ml_model=xgboost.XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, 
                                                max_depth=int(math.log2(self.num_leaves)),silent=True,
                                                subsample=0.5,colsample_bytree=self.colsample_bytree, random_state=2018,
                                                tree_method="hist",disable_default_eval_metric=1,scale_pos_weight=25.31)
        elif self.use_catboost and not self.serial_mode:
            self.ml_model=catboost.CatBoostClassifier(eval_metric="AUC",iterations=self.n_estimators,learning_rate=self.learning_rate,
                                                      random_state=2018, bootstrap_type="Bernoulli", subsample=0.5, use_best_model=True,
                                                      max_depth=int(math.log2(self.num_leaves)), one_hot_max_size=100,
                                                      colsample_bylevel=self.colsample_bytree, early_stopping_rounds=20, max_bin=255, 
                                                      logging_level="Silent",allow_writing_files=False, scale_pos_weight=25.31,
                                                      od_type="Iter")
        elif self.only_decision_tree and not self.serial_mode:
            self.ml_model=lightgbm.LGBMClassifier(n_estimators=1, n_jobs=1, random_state=2018,
                                                  num_leaves=32, learning_rate=self.learning_rate, 
                                                  colsample_bytree=1, verbose=-1,silent=True,is_unbalance=True,
                                                  subsample_freq=0, subsample=1.0, max_depth=int(math.log2(32)),
                                                  subsample_for_bin=1000000,min_child_samples=1000, max_cat_to_onehot=100, cat_smooth=0.0,
                                                  cat_l2=0.0)
        elif not self.serial_mode:
            self.ml_model=lightgbm.LGBMClassifier(n_estimators=self.n_estimators,n_jobs=1,random_state=2018,
                                                  num_leaves=self.num_leaves, learning_rate=self.learning_rate,
                                                  colsample_bytree=self.colsample_bytree,verbose=-1,silent=True,is_unbalance=True,
                                                  subsample_freq=1, subsample=self.rowsample, max_depth=int(math.log2(self.num_leaves)),
                                                  subsample_for_bin=1000000,min_child_samples=1000,max_cat_to_onehot=100,cat_smooth=0.0,cat_l2=0.0)

        self.count_pos_idx=0
        self.count_neg_idx=0

        self.verbose=True

        self.static_cols_without_encode=["Age","Height","Emergency"]
        self.static_cols_one_hot_encode=["Surgical","APACHEPatGroup"]
        self.static_cols_one_hot_encode_str=["Sex"]

        self.static_cols_without_encode_final=None
        self.static_cols_one_hot_encode_final=None
        self.static_cols_one_hot_encode_str_final=None

        self.unique_values={}

        # Unique values of static variables
        self.unique_values["PatGroup"]=[113,116,5,115,114,117,-1,118]
        self.unique_values["APACHECode"]=[5,6,3,0,2,10,11,8,7,4]
        self.unique_values["Discharge"]=[2,4]
        self.unique_values["Euroscores"]=[17,16,18,19,20,15,21,22,14,24,23]
        self.unique_values["Surgical"]=[3,0,1]
        self.unique_values["Sex"]=['M','F','U']
        self.str_to_int_dict={"M": 0, "F": 1, "U": 2}

        self.X_col_names=None
        self.X_cat_cols=None

        self.pos_to_negative_upsample_factor=3
        self.neg_only_sample_factor=0.1

    def set_std_mean(self, mean_arr):
        self.mean_arr=mean_arr


    def set_std_std(self, std_arr):
        self.std_arr=std_arr

        if self.filter_low_variance:
            self.std_arr=self.std_arr[self.std_arr>self.std_eps]
            

    def set_std_class_weights(self, class_weights):
        self.class_weights=class_weights

    def add_train_patient(self, X_df, y_df, df_static=None):

        add_col_info=False

        if self.X_col_names is None:
            self.X_col_names=[]
            self.X_cat_cols=[]
            add_col_info=True

        if self.static_cols_without_encode_final is None:
            self.static_cols_without_encode_final = list(filter(lambda col2: col2 in self.static_cols_without_encode,
                                                        df_static.columns.values.tolist()))
            self.static_cols_one_hot_encode_final = list(filter(lambda col2: col2 in self.static_cols_one_hot_encode,
                                                        df_static.columns.values.tolist()))
            self.static_cols_one_hot_encode_str_final = list(filter(lambda col2: col2 in self.static_cols_one_hot_encode_str,
                                                        df_static.columns.values.tolist()))

        cols_X=sorted(X_df.columns.values.tolist())

        if self.sel_cols_X is None and self.no_datetime:
            self.sel_cols_X=list(filter(lambda col: "Patient" not in col and "AbsDatetime" not in col and "RelDatetime" not in col and "Status" not in col, cols_X))
        elif self.sel_cols_X is None:
            self.sel_cols_X=list(filter(lambda col: "Patient" not in col and "AbsDatetime" not in col and "Status" not in col, cols_X))

        X_df=X_df[self.sel_cols_X]
        X_mat=np.array(X_df)

        if self.std_scale_data:
            X_mat=(X_mat-self.mean_arr)/self.std_arr
        
        y_vect=np.array(y_df)

        if self.train_subsample:

            pos_idx=np.where(y_vect==1.0)[0]
            neg_idx=np.where(y_vect==0.0)[0]

            # No positive labels in this patient
            if len(neg_idx)==0 and len(pos_idx)==0:
                return
            elif len(pos_idx)==0:
                full_idx=nprand.choice(neg_idx,int(self.neg_only_sample_factor*len(neg_idx)),replace=False)
                self.count_neg_idx+=int(self.neg_only_sample_factor*len(neg_idx))
            elif len(neg_idx)==0:
                full_idx=pos_idx
                self.count_pos_idx+=len(pos_idx)
            else:
                full_idx=np.concatenate([pos_idx,nprand.choice(neg_idx,min(self.pos_to_negative_upsample_factor*len(pos_idx),
                                                                           len(neg_idx)),replace=False)])
                self.count_pos_idx+=len(pos_idx)
                self.count_neg_idx+=min(self.pos_to_negative_upsample_factor*len(pos_idx),len(neg_idx))

            random.shuffle(full_idx)
            X_mat=X_mat[full_idx]
            y_vect=y_vect[full_idx]

        X_full_collect=[X_mat]

        if add_col_info:
            self.X_col_names.extend(self.sel_cols_X)
            for col in sorted(X_df.columns.values.tolist()):
                if "_mode" in col or col in ["plain_vm19", "plain_vm60", "plain_vm66"]:
                    self.X_cat_cols.append(col)

        for col in self.static_cols_without_encode_final:
            static_val=float(df_static[col].iloc[0])
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

            if add_col_info:
                self.X_col_names.append("static_{}".format(col))

        for col in self.static_cols_one_hot_encode_final:
            static_val=int(df_static[col].iloc[0])
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

            if add_col_info:
                self.X_col_names.append("static_{}".format(col))
                self.X_cat_cols.append("static_{}".format(col))

        for col in self.static_cols_one_hot_encode_str_final:
            static_val=self.str_to_int_dict[str(df_static[col].iloc[0])]
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

            if add_col_info:
                self.X_col_names.append("static_{}".format(col))
                self.X_cat_cols.append("static_{}".format(col))

        X_full=np.concatenate(X_full_collect,axis=1)
        assert(X_full.shape[1]==len(self.X_col_names))
        
        self.collect_X.append(X_full.astype(np.float32))
        self.collect_y.append(y_vect.astype(np.float32))

    def add_val_patient(self, X_df, y_df, df_static=None):
        X_df=X_df[self.sel_cols_X]
        X_mat=np.array(X_df)

        if self.std_scale_data:
            X_mat=(X_mat-self.mean_arr)/self.std_arr
        
        y_vect=np.array(y_df)
        X_full_collect=[X_mat]

        for col in self.static_cols_without_encode_final:
            static_val=float(df_static[col].iloc[0])
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

        for col in self.static_cols_one_hot_encode_final:
            static_val=int(df_static[col].iloc[0])
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

        for col in self.static_cols_one_hot_encode_str_final:
            static_val=self.str_to_int_dict[str(df_static[col].iloc[0])]
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

        X_full=np.concatenate(X_full_collect,axis=1)
        assert(X_full.shape[1]==len(self.X_col_names))
        self.collect_X_val.append(X_full.astype(np.float32))
        self.collect_y_val.append(y_vect.astype(np.float32))


    def derived_feature_set(self, probe_set):
        collected_features=[]

        for probe_val in probe_set:
            for idx,col_name in enumerate(self.X_col_names):

                # Static column match
                if "static_" in col_name and str(probe_val) in col_name:
                    collected_features.append(idx)
                    
                # Exact match case..
                elif col_name==probe_val:
                    collected_features.append(idx)

                # General column match
                elif "m{}_".format(probe_val) in col_name or \
                   "plain_pm{}".format(probe_val)==col_name or \
                   "plain_vm{}".format(probe_val)==col_name or \
                   probe_val==5 and "map_" in col_name or \
                   probe_val==41 and "dop_" in col_name or \
                   probe_val==42 and "mil_" in col_name or \
                   probe_val==43 and "lev_" in col_name or \
                   probe_val==44 and "theo_" in col_name or \
                   probe_val==39 and "noreph_" in col_name or \
                   probe_val==40 and "epineph_" in col_name or \
                   probe_val==45 and "vaso_" in col_name:
                    collected_features.append(idx)

                elif "lac_" in col_name and 136 in probe_set and 146 in probe_set:
                    collected_features.append(idx)

                # Event 1
                elif "event1_" in col_name and 41 in probe_set and 136 in probe_set and 146 in probe_set and \
                   42 in probe_set and 43 in probe_set and 44 in probe_set and 5 in probe_set:
                    collected_features.append(idx)

                # Event 2
                elif "event2_" in col_name and 39 in probe_set and \
                   40 in probe_set and 136 in probe_set and 146 in probe_set:
                    collected_features.append(idx)

                # Event 3
                elif "event3_" in col_name and 39 in probe_set and \
                   40 in probe_set and 45 in probe_set and 136 in probe_set and 146 in probe_set:
                    collected_features.append(idx)

        collected_features=list(set(collected_features))
        return collected_features

    def extract_raw_model(self):
        ''' Extract the raw ML model'''
        return self.ml_model

    def reset_hyperparameters(self,n_estimators=None, num_leaves=None, learning_rate=None,
                              colsample_bytree=None, rowsample_bytree=None,
                              alpha=None, hidden_layer_size=None):

        if self.only_decision_tree or self.decision_tree_mode:
            self.ml_model=lightgbm.LGBMClassifier(n_estimators=1, n_jobs=1, random_state=2018,
                                                  num_leaves=num_leaves, learning_rate=learning_rate, 
                                                  colsample_bytree=1, verbose=-1,silent=True,is_unbalance=True,
                                                  subsample_freq=0, subsample=1.0, max_depth=int(math.log2(num_leaves)),
                                                  subsample_for_bin=1000000,min_child_samples=1000, max_cat_to_onehot=100, cat_smooth=0.0,cat_l2=0.0)
        elif self.logreg_mode:
            self.ml_model=sklm.SGDClassifier(loss="log", penalty="l2", alpha=alpha, max_iter=1000, tol=1e-3, random_state=2018,
                                             learning_rate="optimal", class_weight="balanced")
        elif self.mlp_mode:
            self.ml_model=sknn.MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation="relu", solver="adam",
                                             alpha=alpha, batch_size=128, learning_rate="constant", learning_rate_init=learning_rate,
                                             max_iter=1000, random_state=2018)
        else:
            self.ml_model=lightgbm.LGBMClassifier(n_estimators=n_estimators,n_jobs=1,random_state=2018,
                                                  num_leaves=num_leaves, learning_rate=learning_rate,
                                                  colsample_bytree=colsample_bytree,verbose=-1,silent=True,is_unbalance=True,
                                                  subsample_freq=1, subsample=rowsample_bytree, max_depth=int(math.log2(num_leaves)),
                                                  subsample_for_bin=1000000,min_child_samples=1000,max_cat_to_onehot=100,cat_smooth=0.0,cat_l2=0.0)

    def fit(self):

        if self.fit_first:
            self.full_X=np.concatenate(self.collect_X,axis=0).astype(np.float32)
            self.full_y=np.concatenate(self.collect_y).astype(np.float32)

            self.collect_X=[]
            self.collect_y=[]
            gc.collect()

            self.full_X_val=np.concatenate(self.collect_X_val,axis=0).astype(np.float32)
            self.full_y_val=np.concatenate(self.collect_y_val).astype(np.float32)

            self.collect_X_val=[]
            self.collect_y_val=[]
            gc.collect()
            self.fit_first=False

            # In case we are MLP or LogReg transform the matrices
            if self.logreg_mode or self.mlp_mode:
                self.imputer_1=skimpute.SimpleImputer(strategy="mean")
                self.imputer_2=skimpute.SimpleImputer(strategy="mean")
                self.imputer_3=skimpute.SimpleImputer(strategy="mean")                                
                self.imputer_cat_1=skimpute.SimpleImputer(strategy="most_frequent")
                self.imputer_cat_2=skimpute.SimpleImputer(strategy="most_frequent")                
                self.scaler_1=skpproc.StandardScaler()
                self.scaler_2=skpproc.StandardScaler()
                self.scaler_3=skpproc.StandardScaler()                
                self.cat_encoder_1=skpproc.OneHotEncoder(categories="auto",sparse=False,handle_unknown="ignore")
                self.cat_encoder_2=skpproc.OneHotEncoder(categories="auto",sparse=False,handle_unknown="ignore")
                
                # Gather indices of the categorical columns
                cat_col_idxs=[]
                print("Number of cat cols: {}".format(len(self.X_cat_cols)))
                for cat_colname in self.X_cat_cols:
                    cat_col_idxs.append(self.X_col_names.index(cat_colname))

                assert(self.full_X.shape[1]==500)
                sorted_cat_cols=list(sorted(cat_col_idxs))

                if len(sorted_cat_cols)==0:
                    self.composed_imputer=self.imputer_1
                    self.composed_scaler_encoder=self.scaler_1
                
                elif sorted_cat_cols[1]-sorted_cat_cols[0]==1:
                    lidx=sorted_cat_cols[0]
                    ridx=sorted_cat_cols[1]                    
                    self.composed_imputer=skcompose.ColumnTransformer([("cont_impute_1",self.imputer_1,np.arange(lidx)),
                                                                       ("cat_impute_1",self.imputer_cat_1,[lidx,ridx]),
                                                                       ("cont_impute_2", self.imputer_2,np.arange(ridx+1,self.full_X.shape[1]))],
                                                                      sparse_threshold=0)
                    self.composed_scaler_encoder=skcompose.ColumnTransformer([("cont_scale_1",self.scaler_1,np.arange(lidx)),
                                                                              ("cat_encode_1",self.cat_encoder_1,[lidx,ridx]),
                                                                              ("cont_scale_3",self.scaler_2,np.arange(ridx+1,self.full_X.shape[1]))],
                                                                             sparse_threshold=0)

                else:
                    lidx=sorted_cat_cols[0]
                    ridx=sorted_cat_cols[1]
                    self.composed_imputer=skcompose.ColumnTransformer([("cont_impute_1",self.imputer_1,np.arange(lidx)),
                                                                       ("cat_impute_1",self.imputer_cat_1,[lidx]),
                                                                       ("cont_impute_2", self.imputer_2,np.arange(lidx+1,ridx)),
                                                                       ("cat_impute_2", self.imputer_cat_2,[ridx]),
                                                                       ("cont_impute_3", self.imputer_3,np.arange(ridx+1,self.full_X.shape[1]))],
                                                                      sparse_threshold=0)

                    self.composed_scaler_encoder=skcompose.ColumnTransformer([("cont_scale_1",self.scaler_1,np.arange(lidx)),
                                                                              ("cat_encode_1",self.cat_encoder_1,[lidx]),
                                                                              ("cont_scale_2",self.scaler_2,np.arange(lidx+1,ridx)),
                                                                              ("cat_encode_2",self.cat_encoder_2,[ridx]),
                                                                              ("cont_scale_3",self.scaler_3,np.arange(ridx+1,self.full_X.shape[1]))],
                                                                             sparse_threshold=0)

                self.full_X[~np.isfinite(self.full_X)]=np.nan
                self.full_X_val[~np.isfinite(self.full_X_val)]=np.nan
                self.full_X=self.composed_imputer.fit_transform(self.full_X)
                self.full_X=self.composed_scaler_encoder.fit_transform(self.full_X)
                self.full_X_val=self.composed_imputer.transform(self.full_X_val)
                self.full_X_val=self.composed_scaler_encoder.transform(self.full_X_val)

        if self.verbose:
            print("Training matrix dimension: {}x{}".format(self.full_X.shape[0], self.full_X.shape[1]),flush=True)
            print("Validation matrix dimension: {}x{}".format(self.full_X_val.shape[0], self.full_X_val.shape[1]), flush=True)
            
        if self.univariate_test:
            Fstat,_=skfselect.f_classif(self.full_X, self.full_y)
            sort_idx=list(np.argsort(Fstat))[::-1]

            with open("./features_F_scores.tsv",'w') as fp:
                print("feat_name\tFscore",file=fp)
                for jdx in sort_idx:
                    print("{}\t{}".format(self.X_col_names[jdx], Fstat[jdx]),file=fp)

            sys.exit(0)

        if self.select_features_forward:
            n_vars_to_select=21
            selected_vars=[]
            search_vars=[136,146,5,"RelDatetime","Age",1,41,42,43,44,13,28,172,174,176,4,62,3,20,87,23]
            assert(len(search_vars)==n_vars_to_select)
            
            while len(selected_vars)<n_vars_to_select:
                best_score_round=-np.inf
                best_vid_round=None

                for idx,vid in enumerate(search_vars):
                    probe_set=selected_vars+[vid]
                    selected_idxs=self.derived_feature_set(probe_set)
                    der_X=self.full_X[:,selected_idxs]
                    der_X_val=self.full_X_val[:,selected_idxs]
                    derived_names=[self.X_col_names[jdx] for jdx in selected_idxs]
                    derived_cat_cols=list(set(derived_names).intersection(set(self.X_cat_cols)))
                    try:
                        self.ml_model.fit(der_X,self.full_y, feature_name=derived_names,eval_set=[(der_X_val, self.full_y_val)],early_stopping_rounds=20,verbose=False,
                                          categorical_feature=derived_cat_cols,eval_metric=custom_auprc_metric)
                    except:
                        print("Degenerate variable set: Skipping...")
                        continue

                    metrics=self.get_validation_scores(red_idxs=selected_idxs)
                    current_auprc=metrics["auprc"]

                    if current_auprc>best_score_round:
                        best_vid_round=vid
                        best_score_round=current_auprc

                selected_vars.append(best_vid_round)
                search_vars.remove(best_vid_round)
                print("Feature selection round {}/30 DONE".format(len(selected_vars)))
                print("Added variable {}, New score AUPRC={:.3f}".format(best_vid_round,best_score_round))

            print("Feature selection finalized...")
            sys.exit(0)

        if self.select_features_backward:
            search_vars=[136,146,60,5,41,42,43,44,39,40,45,66,12,152,20,72,15,64,65,160,1,168,135,61,14,"PatGroup",
                         "Age", "Height", "Surgical","RelDatetime"]
            selected_vars=search_vars.copy()
            
            while len(selected_vars)>0:
                best_score_round=-np.inf
                best_vid_round=None

                for idx,vid in enumerate(selected_vars):
                    probe_set=selected_vars.copy()
                    probe_set.remove(vid)
                    selected_idxs=self.derived_feature_set(probe_set)
                    der_X=self.full_X[:,selected_idxs]
                    der_X_val=self.full_X_val[:,selected_idxs]
                    derived_names=[self.X_col_names[jdx] for jdx in selected_idxs]
                    derived_cat_cols=list(set(derived_names).intersection(set(self.X_cat_cols)))
                    try:
                        self.ml_model.fit(der_X,self.full_y, feature_name=derived_names,eval_set=[(der_X_val, self.full_y_val)],early_stopping_rounds=10,verbose=False,
                                          categorical_feature=derived_cat_cols,eval_metric="auc")
                    except:
                        print("Degenerate variable set: Skipping...")
                        continue                    
                    
                    metrics=self.get_validation_scores(red_idxs=selected_idxs)
                    current_auprc=metrics["auprc"]

                    if current_auprc>best_score_round:
                        best_vid_round=vid
                        best_score_round=current_auprc

                selected_vars.remove(best_vid_round)
                print("Feature selection round {}/30 DONE".format(len(search_vars)-len(selected_vars)))
                print("Removed variable {}, New score AUPRC={:.3f}".format(best_vid_round,best_score_round))
                
            print("Feature selection finalized...")
            sys.exit(0)

        if not self.use_xgboost and not self.use_catboost and not self.decision_tree_mode and not self.logreg_mode and not self.mlp_mode:
            self.ml_model.set_params(**{"metric": 'None'})

        cat_idxs=[]
        for cidx,feat_name in enumerate(self.X_col_names):
            if feat_name in self.X_cat_cols:
                cat_idxs.append(cidx)

        if self.use_xgboost:
            self.ml_model.fit(self.full_X,self.full_y, eval_set=[(self.full_X_val, self.full_y_val)],eval_metric="logloss",early_stopping_rounds=50,verbose=False)
        elif self.use_catboost:
            catboost_X=pd.DataFrame(self.full_X,columns=self.X_col_names)
            catboost_Xval=pd.DataFrame(self.full_X_val,columns=self.X_col_names)
            for cat_col in self.X_cat_cols:
                catboost_X[cat_col]=catboost_X[cat_col].astype(str)
                catboost_Xval[cat_col]=catboost_Xval[cat_col].astype(str)
            self.ml_model.fit(catboost_X,self.full_y, eval_set=[(catboost_Xval, self.full_y_val)], cat_features=cat_idxs, silent=True,early_stopping_rounds=50)
        else:

            if self.decision_tree_mode:
                self.ml_model.fit(self.full_X,self.full_y, feature_name=self.X_col_names, categorical_feature=self.X_cat_cols,
                                  eval_set=[(self.full_X_val, self.full_y_val)],eval_metric=custom_auprc_metric,verbose=False)
            elif self.logreg_mode:
                self.ml_model.fit(self.full_X, self.full_y)
            elif self.mlp_mode:
                self.ml_model.fit(self.full_X, self.full_y)
            else:
                self.ml_model.fit(self.full_X,self.full_y, feature_name=self.X_col_names, categorical_feature=self.X_cat_cols,
                                  eval_set=[(self.full_X_val, self.full_y_val)],eval_metric=custom_auprc_metric,early_stopping_rounds=50,verbose=False)
            
    def plot_tree(self):
        ''' Plots a tree to a new MPL figure'''
        lightgbm.plot_tree(self.ml_model, tree_index=0, show_info=["split_gain"])

    def get_evaluation_trace(self):
        if self.use_xgboost:
            return self.ml_model.evals_result()["validation_0"]["logloss"]
        else:
            return self.ml_model.evals_result_["valid_0"]["custom_auprc"]

    def get_validation_scores(self,red_idxs=None):
        if red_idxs is not None:
            der_X_val=self.full_X_val[:,red_idxs]
        else:
            der_X_val=self.full_X_val

        if self.use_catboost:
            der_frame=pd.DataFrame(der_X_val,columns=self.X_col_names)
            for cat_col in self.X_cat_cols:
                der_frame[cat_col]=der_frame[cat_col].astype(str)
            pred_vect=self.ml_model.predict_proba(der_frame)[:,1]
        else:
            pred_vect=self.ml_model.predict_proba(der_X_val)[:,1]

        auroc_score=skmetrics.roc_auc_score(self.full_y_val,pred_vect)
        auprc_score=skmetrics.average_precision_score(self.full_y_val,pred_vect)
        return {"auroc": auroc_score, "auprc": auprc_score}

    def get_train_scores(self):

        if self.use_catboost:
            der_frame=pd.DataFrame(self.full_X,columns=self.X_col_names)
            for cat_col in self.X_cat_cols:
                der_frame[cat_col]=der_frame[cat_col].astype(str)
            pred_vect=self.ml_model.predict_proba(der_frame)[:,1]
        else:
            pred_vect=self.ml_model.predict_proba(self.full_X)[:,1]

        auroc_score=skmetrics.roc_auc_score(self.full_y,pred_vect)
        auprc_score=skmetrics.average_precision_score(self.full_y, pred_vect)
        return {"auroc": auroc_score, "auprc": auprc_score}

    def col_names(self):
        return self.X_col_names

    def predict(self, X_df, y_df=None, pid=None, df_static=None):
        abs_dt=X_df["AbsDatetime"]
        rel_dt=X_df["RelDatetime"]
        pred_all_vect=mlhc_array.empty_nan(abs_dt.size)
        X_input=X_df[X_df["SampleStatus_{}".format(self.task_key)]=="VALID"]
        X_input=X_input[self.sel_cols_X]
        X_mat=np.array(X_input)
        
        # No valid samples to use, hence we can not predict anything
        if X_mat.shape[0]==0:
            return None

        if self.filter_low_variance:
            X_mat=X_mat[:,self.std_arr>=self.std_eps]

        if self.std_scale_data:
            X_mat=(X_mat-train_mean_red)/train_std_red
        
        X_full_collect=[X_mat]

        for col in self.static_cols_without_encode_final:
            static_val=float(df_static[col].iloc[0])
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

        for col in self.static_cols_one_hot_encode_final:
            static_val=int(df_static[col].iloc[0])
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

        for col in self.static_cols_one_hot_encode_str_final:
            static_val=self.str_to_int_dict[str(df_static[col].iloc[0])]
            col_arr=mlhc_array.value_empty((X_mat.shape[0],1),static_val)
            X_full_collect.append(col_arr)

        X_full=np.concatenate(X_full_collect,axis=1)

        if self.logreg_mode or self.mlp_mode:
            X_full[~np.isfinite(X_full)]=np.nan
            X_full=self.composed_imputer.transform(X_full)
            X_full=self.composed_scaler_encoder.transform(X_full)
        
        pred_vect=self.ml_model.predict_proba(X_full)[:,1]

        df_out_dict={}        

        if not self.logreg_mode and not self.mlp_mode:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer=shap.TreeExplainer(self.ml_model)
                shap_values=explainer.shap_values(X_full)[1]

            assert(shap_values.shape[1]==len(self.X_col_names))
            
            for cidx,cname in enumerate(self.X_col_names):
                temp_vect=mlhc_array.empty_nan(abs_dt.size)
                temp_vect[X_df["SampleStatus_{}".format(self.task_key)]=="VALID"]=shap_values[:,cidx]
                df_out_dict["RawShap_{}".format(cname)]=temp_vect

        pred_all_vect[X_df["SampleStatus_{}".format(self.task_key)]=="VALID"]=pred_vect
        pid_vect=mlhc_array.value_empty(abs_dt.size, pid)
        cols_y=sorted(y_df.columns.values.tolist())
        y_vect=np.array(y_df["Label_{}".format(self.task_key)])
        y_time_col=np.array(y_df["Label_{}".format(self.time_key)])

        df_out_dict["PatientID"]=pid_vect
        df_out_dict["PredScore"]=pred_all_vect
        df_out_dict["TrueLabel"]=y_vect
        df_out_dict["AbsDatetime"]=abs_dt
        df_out_dict["RelDatetime"]=rel_dt
        df_out_dict["TimeToEvent"]=y_time_col
        df_out=pd.DataFrame(df_out_dict)
        return df_out



    def feature_importances(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer=shap.TreeExplainer(self.ml_model)
            pos_idxs=np.where(self.full_y_val==1.0)[0]
            neg_idxs=np.where(self.full_y_val==0.0)[0]
            red_neg_idxs=nprand.choice(neg_idxs,len(pos_idxs),replace=False)
            all_idxs=np.concatenate([pos_idxs,red_neg_idxs])
            red_X_val=self.full_X_val[all_idxs,:]
            shap_values=explainer.shap_values(red_X_val)[1]

        global_vals=np.mean(np.absolute(shap_values),axis=0)
        return global_vals
