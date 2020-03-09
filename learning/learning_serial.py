''' Machine learning script supporting different ML models'''

import os
import os.path
import sys
import random
import timeit
import csv
import warnings
import argparse
import ipdb
import gc
import glob
import itertools
import json
import pickle
import psutil
import time
import gc

import numpy as np
import pandas as pd
import numpy.random as nprand
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import sklearn.dummy as skdummy

import circews.functions.static as bern_static
import circews.functions.features as bern_features
import circews.functions.util.memory as mlhc_memory
import circews.functions.util.io as mlhc_io

import circews.classes.lgbm_model as bern_lgbm

def search_patient(pid, batch):
    ''' Searches for a patient by going through the pipeline changes 1-by-1'''
    endpoint_path="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3a_endpoints/180704"
    merged_path="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3_merged/180704"
    imputed_path="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_180827"
    features_path="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/180827"
    print("Searching for PID: {}".format(pid))

    ep_files=glob.glob(os.path.join(endpoint_path,"reduced","reduced_endpoints_{}_*.h5".format(batch)))
    assert(len(ep_files)==1)
    
    df=pd.read_hdf(ep_files[0],mode='r',where="PatientID={}".format(pid))

    if df.shape[0]>0:
        print("ENDPOINT STAGE: FINE")

def learning_train_to_test(configs):
    ''' Online fitting, hyperparameter optimization against the validation data, and predicting on testing data,
        saving the results back to disk'''
    random.seed(2018)
    nprand.seed(2018)

    time_after_begin=timeit.default_timer()

    if configs["profile_report"]:
        cum_time_load_df=0.0
        cum_time_load_shapelet=0.0
        cum_time_load_labels=0.0
        cum_time_merge=0.0
        cum_time_static=0.0
        cum_time_add_data=0.0

    hp_grid=configs["GBM_HP_GRID"]

    full_grid=list(itertools.product(hp_grid["n_estimators"], hp_grid["num_leaves"], hp_grid["learning_rate"],
                                     hp_grid["colsample_bytree"], hp_grid["rowsample_bytree"]))

    if configs["decision_tree_mode"]:
        hp_grid=configs["TREE_GRID"]
        full_grid=list(itertools.product(hp_grid["n_estimators"], hp_grid["num_leaves"], hp_grid["learning_rate"]))
    elif configs["logreg_mode"]:
        hp_grid=configs["LR_GRID"]
        full_grid=list(itertools.product(hp_grid["alpha"]))
    elif configs["mlp_mode"]:
        hp_grid=configs["MLP_GRID"]
        full_grid=list(itertools.product(hp_grid["hidden_layer_size"], hp_grid["learning_rate"],hp_grid["alpha"]))

    model_type=configs["ml_model"]
    left_hours=configs["lhours"]
    right_hours=configs["rhours"]
    split_key=configs["split_key"]
    mimic_split_key=configs["mimic_split_key"]
    constant_bern_split_key="held_out"
    label_key=configs["label_key"]
    task_key=configs["task_key"]
    val_type=configs["val_type"]
    reduced_data_str=configs["data_mode"]

    bern_imputed_base_dir=configs["bern_imputed_dir"]
    bern_ml_input_base_dir=configs["bern_ml_input_dir"]
    mimic_imputed_base_dir=configs["mimic_imputed_dir"]
    mimic_ml_input_base_dir=configs["mimic_ml_input_dir"]

    special_test_set_imputed_dir=configs["special_test_set_imputed_dir"]
    special_test_set_ml_input_dir=configs["special_test_set_ml_input_dir"]

    if configs["dataset"]=="mimic":
        mimic_imputed_base_dir=configs["mimic_imputed_dir"]
        mimic_ml_input_base_dir=configs["mimic_ml_input_dir"]

    assert(reduced_data_str in ["reduced","non_reduced"])
    
    if reduced_data_str=="reduced":
        dim_reduced_data=True
    else:
        dim_reduced_data=False

    column_desc=configs["column_set"]

    print("Fitting model with label: {}, interval [{},{}]".format(label_key,left_hours,right_hours),flush=True)
    
    if configs["xinrui_subsample"]:
        subsample_key="xinrui"
    else:
        subsample_key="full"

    if dim_reduced_data:
        problem_dir=os.path.join(bern_ml_input_base_dir,"reduced",split_key,"{}_{}_{}".format(label_key, left_hours, right_hours))
        impute_dir=os.path.join(bern_imputed_base_dir, "reduced",split_key)

        if not configs["special_test_set"]=="NONE":
            print("TESTING IN A DIFFERENT SPLIT THAN TRAINING!", flush=True)                        
            special_test_problem_dir=os.path.join(bern_ml_input_base_dir,"reduced",configs["special_test_set"], "{}_{}_{}".format(label_key, left_hours, right_hours))
            special_test_impute_dir=os.path.join(bern_imputed_base_dir,"reduced", configs["special_test_set"])

        if configs["special_test_set_imputed_dir"] is not None:
            print("TESTING IN A DIFFERENT DATA THAN TRAINING!", flush=True)
            special_test_problem_dir=os.path.join(special_test_set_ml_input_dir,"reduced",split_key, "{}_{}_{}".format(label_key, left_hours, right_hours))
            special_test_impute_dir=os.path.join(special_test_set_imputed_dir,"reduced",split_key)

        if configs["dataset"] in ["mimic","mimic_only"]:
            mimic_problem_dir=os.path.join(mimic_ml_input_base_dir,"reduced",constant_bern_split_key,"{}_{}_{}".format(label_key, left_hours, right_hours))
            mimic_impute_dir=os.path.join(mimic_imputed_base_dir, "reduced",constant_bern_split_key)
            
        output_dir=os.path.join(configs["output_dir"],"reduced",split_key,"{}_{}_{}_{}_{}_{}".format(task_key, left_hours, right_hours, column_desc, model_type, subsample_key))
    else: 
        assert(False) # THIS BRANCH IS DEPRECATED AND NOT IN USE...

    train_subsample = configs["negative_subsampling"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_dir=os.path.join(problem_dir,"X")
    y_dir=os.path.join(problem_dir,"y")
    X_meta_dir=os.path.join(X_dir,"metadata")
    y_meta_dir=os.path.join(y_dir,"metadata")

    if configs["dataset"] in ["mimic","mimic_only"]:
        mimic_X_dir=os.path.join(mimic_problem_dir,"X")
        mimic_y_dir=os.path.join(mimic_problem_dir,"y")
        mimic_X_meta_dir=os.path.join(mimic_X_dir,"metadata")
        mimic_y_meta_dir=os.path.join(mimic_y_dir,"metadata")

    if not configs["special_test_set"]=="NONE" or configs["special_test_set_imputed_dir"] is not None:
        X_dir_special_test=os.path.join(special_test_problem_dir, "X")
        y_dir_special_test=os.path.join(special_test_problem_dir,"y")

    bern_batch_map=mlhc_io.load_pickle(configs["bern_pid_batch_map_path"])["pid_to_chunk"]
    mimic_batch_map=mlhc_io.load_pickle(configs["mimic_pid_batch_map_path"])["pid_to_chunk"]

    if configs["dataset"]=="mimic_only":
        data_split=mlhc_io.load_pickle(configs["mimic_data_split_path"])[mimic_split_key]
    elif not configs["special_development_split"]=="NONE":
        data_split=mlhc_io.load_pickle(configs["bern_temporal_data_split_path"])[configs["special_development_split"]]
    else:
        data_split=mlhc_io.load_pickle(configs["bern_temporal_data_split_path"])[split_key]

    if not configs["special_test_set"]=="NONE":
        assert(configs["dataset"]=="bern")
        special_test_split=mlhc_io.load_pickle(configs["bern_temporal_data_split_path"])[configs["special_test_set"]]

    train=data_split["train"]
    val=data_split["val"]

    if not configs["special_year"]==-1:
        year_len_dict={}
        df_patient_full=pd.read_hdf(configs["bern_general_data_table_path"],mode='r')
        min_num_pids=10000000
        for year in range(2008,2016):
            pids_year=list(filter(lambda pid: int(str(bern_static.lookup_admission_time(pid,df_patient_full))[:4])==year, train))
            min_num_pids=min(len(pids_year),min_num_pids)
            year_len_dict[year]=len(pids_year)
            
        pids_year=list(filter(lambda pid: int(str(bern_static.lookup_admission_time(pid,df_patient_full))[:4])==configs["special_year"], train))
        chosen_pids=random.sample(pids_year,min_num_pids)
        train=chosen_pids[:int(configs["special_year_train_ratio"]*len(chosen_pids))]
        val=chosen_pids[int(configs["special_year_train_ratio"]*len(chosen_pids)):]

        print("TEMP_GEN MODE: Number of train/val PIDs sampled: {}/{}".format(len(train),len(val)),flush=True)

    if configs["dataset"]=="bern" and not configs["special_test_set"]=="NONE":
        test=special_test_split["test"]

    elif configs["dataset"] in ["bern","mimic_only"]:
        test=data_split["test"]
    
    elif configs["dataset"]=="mimic":
        test=list(map(int, mlhc_io.read_list_from_file(configs["mimic_all_pid_list_path"])))

    df_shapelet_path=os.path.join(configs["shapelets_path"],"MHUESER_HDF_FIXED","Shapelet_features_{}".format(configs["split_key"])) 

    if model_type=="lightgbm":
        no_datetime=(configs["column_set"]=="without_datetime")
        ml_model=bern_lgbm.LightGbmModel(task_key="{}{}To{}Hours".format(task_key, left_hours, right_hours), train_subsample=train_subsample,
                                         time_key="TimeToWorseState{}To{}Hours".format(left_hours, right_hours),
                                         var_desc=configs["column_set"],no_datetime=no_datetime, select_features_forward=configs["select_features_forward"],
                                         select_features_backward=configs["select_features_backward"],
                                         univariate_test=configs["univariate_test"], use_xgboost=configs["use_xgboost"],
                                         use_catboost=configs["use_catboost"], dataset=configs["dataset"],
                                         only_decision_tree=configs["decision_tree_baseline"],serial_mode=True,
                                         decision_tree_mode=configs["decision_tree_mode"], logreg_mode=configs["logreg_mode"],
                                         mlp_mode=configs["mlp_mode"])
    else:
        print("ERROR: Invalid model type specified...",flush=True)
        sys.exit(1)

    if model_type=="logreg_legacy":
        train_mean=np.load(os.path.join(X_meta_dir,"train_mean.npy"))
        train_std=np.load(os.path.join(X_meta_dir,"train_std.npy"))
        class_weight_vect=np.load(os.path.join(y_meta_dir,"class_weight_vect_balanced.npy"))

    process_info = psutil.Process(os.getpid())
    random.shuffle(train)
    print("Training {}...".format(model_type),flush=True)

    if configs["10percent_sample"]:
        train=train[:int(0.1*len(train))]
        val=val[:int(0.1*len(val))]
    elif configs["10percent_sample_train"]:
        train=train[:int(0.1*len(train))]
    elif configs["5percent_sample_train"]:
        train=train[:int(0.05*len(train))]
    elif configs["1percent_sample_train"]:
        train=train[:int(0.01*len(train))]
    elif configs["0.1percent_sample_train"]:
        train=train[:int(0.001*len(train))]
    elif configs["20percent_sample_train"]:
        train=train[:int(0.2*len(train))]
    elif configs["50percent_sample_train"]:
        train=train[:int(0.5*len(train))]
    elif configs["1percent_sample_val"]:
        val=val[:int(0.01*len(val))]        
    elif configs["1percent_sample"]:
        train=train[:int(0.01*len(train))]
        val=val[:int(0.01*len(val))]
    elif configs["verysmall_sample"]:
        train=train[:int(0.001*len(train))]
        val=val[:int(0.01*len(val))]

    skip_no_batch_file=0
    skip_no_patdf=0
    skip_no_goodsegment=0
    skip_no_labdf=0
    skip_no_colissue=0
    skip_no_staticdf=0
    n_skipped_patients=0

    if configs["dataset"]=="mimic_only":
        full_static_df=pd.read_hdf(os.path.join(mimic_impute_dir, "static.h5"), mode='r')
    else:
        full_static_df=pd.read_hdf(os.path.join(impute_dir, "static.h5"), mode='r')

    final_cols=None

    for idx,train_patient in enumerate(train):
        
        if column_desc=="mews_score":
            break

        if (idx+1)%100==0:
            print("Train Patient {}/{}: {}, SKIPPED: {}".format(idx+1,len(train),train_patient,n_skipped_patients),flush=True)
            print("SKIP, nbf:{}, npd: {}, ngs: {}, nld: {}, nci: {}".format(skip_no_batch_file, skip_no_patdf, skip_no_goodsegment, skip_no_labdf, skip_no_colissue),
                  flush=True)
            mlhc_memory.print_memory_diags()

            if configs["profile_report"]:
                print("FEAT: {:.3f}, SHAPELET: {:.3f}, LABELS: {:.3f}, MERGE: {:.3f}, STATIC: {:.3f}, ADD_DATA: {:.3f}".format(cum_time_load_df, cum_time_load_shapelet, cum_time_load_labels, 
                                                                                                                               cum_time_merge, cum_time_static, cum_time_add_data),flush=True)
        if configs["dataset"]=="mimic_only":
            batch_pat=mimic_batch_map[train_patient]
        else:
            batch_pat=bern_batch_map[train_patient]

        if configs["xinrui_subsample"] and batch_pat not in [33,34,35,36,37]:
            continue

        if configs["dataset"]=="mimic_only":
            df_path=os.path.join(mimic_X_dir,"batch_{}.h5".format(batch_pat))
            df_label_path=os.path.join(mimic_y_dir,"batch_{}.h5".format(batch_pat))
            df_shapelet_path=os.path.join(configs["shapelets_path"],"Shapelet_features_{}_MIMIC.h5".format(configs["split_key"]))
        else:
            df_path=os.path.join(X_dir,"batch_{}.h5".format(batch_pat))
            df_label_path=os.path.join(y_dir,"batch_{}.h5".format(batch_pat))
        
        if not os.path.exists(df_path) or not os.path.exists(df_label_path):
            n_skipped_patients+=1
            skip_no_batch_file+=1
            continue

        if configs["profile_report"]:
            t_begin=timeit.default_timer()

        try:
            pat_df=pd.read_hdf(df_path,"/{}".format(train_patient),mode='r')
            pat_df=pat_df[pat_df["SampleStatus_{}{}To{}Hours".format(task_key, left_hours,right_hours)]=="VALID"]
        except KeyError:
            n_skipped_patients+=1
            skip_no_patdf+=1
            continue

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_load_df+=t_end-t_begin

        if pat_df.shape[0]==0:
            n_skipped_patients+=1
            skip_no_patdf+=1
            continue

        if configs["profile_report"]:
            t_begin=timeit.default_timer()
        
        if configs["add_shapelets"]:
            shapelet_df=pd.read_hdf(os.path.join(df_shapelet_path,"batch_{}.h5".format(batch_pat)), '/p{}'.format(train_patient), mode='r')
            shapelet_df["AbsDatetime"]=pd.to_datetime(shapelet_df["AbsDatetime"])
            special_cols=["AbsDatetime","PatientID"]
            sel_shapelet_cols=bern_features.yield_final_cols("only_shapelets",configs=configs)
            shapelet_cols=list(filter(lambda col: col in sel_shapelet_cols, sorted(shapelet_df.columns.values.tolist())))
            shapelet_df=shapelet_df[special_cols+shapelet_cols]

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_load_shapelet+=t_end-t_begin

        if configs["add_shapelets"]:
            if shapelet_df.shape[0]==0:
                skip_no_patdf+=1
                continue
        
        if configs["profile_report"]:
            t_begin=timeit.default_timer()

        try:
            pat_label_df=pd.read_hdf(df_label_path,"/{}".format(train_patient),mode='r')
            pat_label_df=pat_label_df[pat_label_df["SampleStatus_{}{}To{}Hours".format(task_key, left_hours, right_hours)]=="VALID"]
        except KeyError:
            n_skipped_patients+=1
            skip_no_labdf+=1
            continue

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_load_labels+=t_end-t_begin

        if configs["profile_report"]:
            t_begin=timeit.default_timer()
            
        if configs["add_shapelets"]:
            df_merged=pd.merge(pat_df,shapelet_df,on=["AbsDatetime","PatientID"])
            pat_df=df_merged
            pat_label_df_orig_cols=sorted(pat_label_df.columns.values.tolist())
            pat_label_df=pd.merge(pat_label_df,shapelet_df,on=["AbsDatetime","PatientID"])
            pat_label_df=pat_label_df[pat_label_df_orig_cols]

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_merge+=t_end-t_begin

        if configs["profile_report"]:
            t_begin=timeit.default_timer()

        static_df=full_static_df[full_static_df["PatientID"]==train_patient]

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_static+=t_end-t_begin
        
        if static_df.shape[0]<1:
            n_skipped_patients+=1
            skip_no_staticdf+=1
            continue

        cols_X=sorted(pat_df.columns.values.tolist())
        sel_cols_X=list(filter(lambda col: "Patient" not in col, cols_X))
        X_df=pat_df[sel_cols_X]

        if final_cols is None:
            final_cols=bern_features.yield_final_cols(column_desc, sel_cols_X, configs)
            final_static_cols=list(filter(lambda col: "static_" in col, final_cols))
            final_static_cols=list(map(lambda col: "_".join(col.split("_")[1:]), final_static_cols))
            final_cols=list(filter(lambda col: "static_" not in col, final_cols))

        X_df=X_df[final_cols]
        cols_y=sorted(pat_label_df.columns.tolist())
        y_vect=pat_label_df["Label_{}{}To{}Hours".format(task_key, left_hours, right_hours)]
        assert(y_vect.shape[0]==X_df.shape[0])
        
        static_df=static_df[final_static_cols]

        if configs["profile_report"]:
            t_begin=timeit.default_timer()

        ml_model.add_train_patient(X_df,y_vect,df_static=static_df)

        if configs["profile_report"]:
            t_end=timeit.default_timer()
            cum_time_add_data+=t_end-t_begin

        gc.collect()

    skip_no_batch_file=0
    skip_no_patdf=0
    skip_no_goodsegment=0
    skip_no_labdf=0
    skip_no_colissue=0
    skip_no_staticdf=0
    n_skipped_patients=0

    time_after_train_load=timeit.default_timer()
    print("Seconds after training set patients load: {:.3f}".format(time_after_train_load-time_after_begin))

    for idx,val_patient in enumerate(val):
        
        if column_desc=="mews_score":
            break

        if (idx+1)%100==0:
            print("Val Patient {}/{}: {}, SKIPPED: {}".format(idx+1,len(val),val_patient,n_skipped_patients),flush=True)
            print("SKIP, nbf:{}, npd: {}, ngs: {}, nld: {}, nci: {}".format(skip_no_batch_file, skip_no_patdf, skip_no_goodsegment, skip_no_labdf, skip_no_colissue),
                  flush=True)
            mlhc_memory.print_memory_diags()

        if configs["dataset"]=="mimic_only":
            batch_pat=mimic_batch_map[val_patient]
        else:
            batch_pat=bern_batch_map[val_patient]

        if configs["xinrui_subsample"] and batch_pat not in [37]:
            continue

        if configs["dataset"]=="mimic_only":
            df_path=os.path.join(mimic_X_dir,"batch_{}.h5".format(batch_pat))
            df_label_path=os.path.join(mimic_y_dir,"batch_{}.h5".format(batch_pat))
            df_shapelet_path=os.path.join(configs["shapelets_path"],"Shapelet_features_{}_MIMIC.h5".format(configs["split_key"]))
        else:
            df_path=os.path.join(X_dir,"batch_{}.h5".format(batch_pat))
            df_label_path=os.path.join(y_dir,"batch_{}.h5".format(batch_pat))
        
        if not os.path.exists(df_path) or not os.path.exists(df_label_path):
            n_skipped_patients+=1
            skip_no_batch_file+=1
            continue
            
        try:
            pat_df=pd.read_hdf(df_path,"/{}".format(val_patient),mode='r')
            pat_df=pat_df[pat_df["SampleStatus_{}{}To{}Hours".format(task_key, left_hours,right_hours)]=="VALID"]
        except KeyError:
            n_skipped_patients+=1
            skip_no_patdf+=1
            continue

        if configs["add_shapelets"]:
            shapelet_df=pd.read_hdf(os.path.join(df_shapelet_path,"batch_{}.h5".format(batch_pat)), '/p{}'.format(val_patient), mode='r')
            shapelet_df["AbsDatetime"]=pd.to_datetime(shapelet_df["AbsDatetime"])
            special_cols=["AbsDatetime","PatientID"]
            sel_shapelet_cols=bern_features.yield_final_cols("only_shapelets",configs=configs)            
            shapelet_cols=list(filter(lambda col: col in sel_shapelet_cols, sorted(shapelet_df.columns.values.tolist())))
            shapelet_df=shapelet_df[special_cols+shapelet_cols]

            if shapelet_df.shape[0]==0:
                skip_no_patdf+=1
                continue

        if pat_df.shape[0]==0:
            skip_no_patdf+=1
            continue

        if configs["add_shapelets"]:
            df_merged=pd.merge(pat_df,shapelet_df,on=["AbsDatetime","PatientID"])
            pat_df=df_merged
        
        try:
            pat_label_df=pd.read_hdf(df_label_path,"/{}".format(val_patient),mode='r')
            pat_label_df=pat_label_df[pat_label_df["SampleStatus_{}{}To{}Hours".format(task_key, left_hours, right_hours)]=="VALID"]
        except KeyError:
            n_skipped_patients+=1
            skip_no_labdf+=1
            continue
        
        if configs["add_shapelets"]:
            pat_label_df_orig_cols=sorted(pat_label_df.columns.values.tolist())
            pat_label_df=pd.merge(pat_label_df,shapelet_df,on=["AbsDatetime","PatientID"])
            pat_label_df=pat_label_df[pat_label_df_orig_cols]
            
        if configs["dataset"]=="mimic_only":
            static_df=pd.read_hdf(os.path.join(mimic_impute_dir, "static.h5"), mode='r')            
        else:
            static_df=pd.read_hdf(os.path.join(impute_dir, "static.h5"), mode='r')

        static_df=static_df[static_df["PatientID"]==val_patient]
        
        if static_df.shape[0]<1:
            n_skipped_patients+=1
            skip_no_staticdf+=1
            continue

        cols_X=sorted(pat_df.columns.tolist())
        sel_cols_X=list(filter(lambda col: "Patient" not in col, cols_X))
        X_df=pat_df[sel_cols_X]

        if dim_reduced_data:
            X_df=X_df[final_cols]

        static_df=static_df[final_static_cols]

        cols_y=sorted(pat_label_df.columns.tolist())
        y_vect=pat_label_df["Label_{}{}To{}Hours".format(task_key, left_hours, right_hours)]
        assert(y_vect.shape[0]==X_df.shape[0])
        ml_model.add_val_patient(X_df, y_vect,df_static=static_df)
        gc.collect()

    if configs["systrace_mode"]:
        return

    hp_metric_dict={}
    grid_point_cnt=1

    time_after_val_load=timeit.default_timer()
    print("Seconds after validation set patients load: {:.3f}".format(time_after_val_load-time_after_begin))
        
    for grid_point in full_grid:

        if configs["decision_tree_mode"]:
            tree_nestimators,tree_numleaves,tree_learningrate=grid_point
            hp_dict={}
            hp_dict["n_est"]=tree_nestimators
            hp_dict["num_leaves"]=tree_numleaves
            hp_dict["learning_rate"]=tree_learningrate
            print("Exploring GRID point ({},{},{})".format(tree_nestimators, tree_numleaves, tree_learningrate),flush=True)
        elif configs["logreg_mode"]:
            lr_alpha=grid_point[0]
            hp_dict={}
            hp_dict["alpha"]=lr_alpha
            print("Exploring GRID point ({})".format(lr_alpha), flush=True)
        elif configs["mlp_mode"]:
            mlp_hiddenlayersize,mlp_learningrate,mlp_alpha=grid_point
            hp_dict={}
            hp_dict["hiddenlayersize"]=mlp_hiddenlayersize
            hp_dict["learningrate"]=mlp_learningrate
            hp_dict["alpha"]=mlp_alpha
            print("Exploring GRID point ({},{},{})".format(mlp_hiddenlayersize,mlp_learningrate, mlp_alpha), flush=True)
        else:
            lgbm_nestimators, lgbm_numleaves, lgbm_learningrate, lgbm_colsamplebytree, lgbm_rowsamplebytree=grid_point
            hp_dict={}
            hp_dict["n_est"]=lgbm_nestimators
            hp_dict["num_leaves"]=lgbm_numleaves
            hp_dict["learning_rate"]=lgbm_learningrate
            hp_dict["colsample_bytree"]=lgbm_colsamplebytree
            hp_dict["rowsample_bytree"]=lgbm_rowsamplebytree
            print("Exploring GRID point ({},{},{},{},{})".format(lgbm_nestimators, lgbm_numleaves, lgbm_learningrate, 
                                                                 lgbm_colsamplebytree, lgbm_rowsamplebytree),flush=True) 
        

        print("HP setting {}/{}".format(grid_point_cnt,len(full_grid)))

        grid_point_cnt+=1

        if model_type=="dummy":
            ml_model=skdummy.DummyClassifier()

        if not column_desc=="mews_score":

            if model_type=="dummy":
                full_y=np.concatenate(collect_y)
                ml_model.fit(X=None,y=full_y)
            elif model_type in ["xgboost","lightgbm","lightgbm_meta","logreg"]:

                if configs["decision_tree_mode"]:
                    ml_model.reset_hyperparameters(n_estimators=tree_nestimators, num_leaves=tree_numleaves, learning_rate=tree_learningrate)
                elif configs["logreg_mode"]:
                    ml_model.reset_hyperparameters(alpha=lr_alpha)
                elif configs["mlp_mode"]:
                    ml_model.reset_hyperparameters(hidden_layer_size=mlp_hiddenlayersize, learning_rate=mlp_learningrate,alpha=mlp_alpha)
                else:
                    ml_model.reset_hyperparameters(n_estimators=lgbm_nestimators, num_leaves=lgbm_numleaves, learning_rate=lgbm_learningrate,
                                                   colsample_bytree=lgbm_colsamplebytree, rowsample_bytree=lgbm_rowsamplebytree)
                t_begin_fit=timeit.default_timer()
                ml_model.fit()
                t_end_fit=timeit.default_timer()
                print("Fitting time: {:.3f} seconds".format(t_end_fit-t_begin_fit))
                gc.collect()

        train_score_dict=ml_model.get_train_scores()
        val_score_dict=ml_model.get_validation_scores()

        if not configs["use_catboost"] and not configs["decision_tree_mode"] and not configs["logreg_mode"] and not configs["mlp_mode"]:
            eval_score_dict=ml_model.get_evaluation_trace()

        if not configs["debug_mode"]:

            if configs["decision_tree_mode"]:
                config_string="n-est_{}_num-leaves_{}_learning-rate_{}".format(tree_nestimators,
                                                                               tree_numleaves,
                                                                               tree_learningrate)
                train_fname_tsv="trainscore_{}.tsv".format(config_string)                
                val_fname_tsv="valscore_{}.tsv".format(config_string)
                dump_fname="model_{}.pickle".format(config_string)                                
            elif configs["logreg_mode"]:
                config_string="alpha_{}".format(lr_alpha)
                train_fname_tsv="trainscore_{}.tsv".format(config_string)                
                val_fname_tsv="valscore_{}.tsv".format(config_string)
                dump_fname="model_{}.pickle".format(config_string)                                
            elif configs["mlp_mode"]:
                config_string="hlayersize_{}_learning-rate_{}_alpha_{}".format(mlp_hiddenlayersize,
                                                                               mlp_learningrate,
                                                                               mlp_alpha)
                train_fname_tsv="trainscore_{}.tsv".format(config_string)                
                val_fname_tsv="valscore_{}.tsv".format(config_string)
                dump_fname="model_{}.pickle".format(config_string)                
            else:
                config_string="n-est_{}_num-leaves_{}_learning-rate_{}_colsample-bytree_{}_rowsample_{}".format(lgbm_nestimators,
                                                                                                                lgbm_numleaves,
                                                                                                                lgbm_learningrate,
                                                                                                                lgbm_colsamplebytree,
                                                                                                                lgbm_rowsamplebytree)
                train_fname_tsv="trainscore_{}.tsv".format(config_string)
                val_fname_tsv="valscore_{}.tsv".format(config_string)
                eval_fname_tsv="trace_{}.tsv".format(config_string)
                dump_fname="model_{}.pickle".format(config_string)

            with open(os.path.join(output_dir,train_fname_tsv),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["auroc", "auprc"])
                csv_fp.writerow([train_score_dict["auroc"], train_score_dict["auprc"]])

            with open(os.path.join(output_dir,val_fname_tsv),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["auroc", "auprc"])
                csv_fp.writerow([val_score_dict["auroc"], val_score_dict["auprc"]])

            with open(os.path.join(output_dir,dump_fname), 'wb') as fp:
                pickle.dump(ml_model.extract_raw_model(), fp)

            hp_metric_dict[json.dumps(hp_dict)]=float(val_score_dict["auprc"])

            if not configs["decision_tree_mode"] and not configs["logreg_mode"] and not configs["mlp_mode"]:
                with open(os.path.join(output_dir,eval_fname_tsv),'w') as fp:
                    csv_fp=csv.writer(fp,delimiter='\t')
                    csv_fp.writerow(["epoch", "val_auprc"])
                    for idx,val_auprc in enumerate(eval_score_dict):
                        csv_fp.writerow([str(idx+1), str(val_auprc)])

        if configs["decision_tree_baseline"]:
            break
                    
    best_hps=json.loads(max(hp_metric_dict, key=hp_metric_dict.get))

    if configs["decision_tree_mode"]:
        nestimators_star=best_hps["n_est"]
        numleaves_star=best_hps["num_leaves"]
        learningrate_star=best_hps["learning_rate"]
    elif configs["logreg_mode"]:
        alpha_star=best_hps["alpha"]
    elif configs["mlp_mode"]:
        hiddenlayersize_star=best_hps["hiddenlayersize"]
        learningrate_star=best_hps["learningrate"]
        alpha_star=best_hps["alpha"]
    else:
        nestimators_star=best_hps["n_est"]
        numleaves_star=best_hps["num_leaves"]
        learningrate_star=best_hps["learning_rate"]
        colsample_star=best_hps["colsample_bytree"]
        rowsample_star=best_hps["rowsample_bytree"]

    if configs["decision_tree_mode"]:
        print("Best HPs: ({},{},{})".format(nestimators_star, numleaves_star, learningrate_star))
        ml_model.reset_hyperparameters(n_estimators=nestimators_star, num_leaves=numleaves_star, learning_rate=learningrate_star)
    elif configs["logreg_mode"]:
        print("Best HPs: ({})".format(alpha_star))
        ml_model.reset_hyperparameters(alpha=alpha_star)
    elif configs["mlp_mode"]:
        print("Best HPs: ({},{},{})".format(hiddenlayersize_star,learningrate_star,alpha_star))
        ml_model.reset_hyperparameters(hidden_layer_size=hiddenlayersize_star, learning_rate=learningrate_star, alpha=alpha_star)
    else:
        print("Best HPs: ({},{},{},{},{})".format(nestimators_star, numleaves_star, learningrate_star, colsample_star, rowsample_star))
        ml_model.reset_hyperparameters(n_estimators=nestimators_star, num_leaves=numleaves_star, learning_rate=learningrate_star,
                                       colsample_bytree=colsample_star, rowsample_bytree=rowsample_star)
        
    ml_model.fit()

    if not configs["debug_mode"] and configs["plot_tree"] and not configs["decision_tree_mode"] and not configs["logreg_mode"] and not configs["mlp_mode"]:
        ml_model.plot_tree()
        plt.savefig(os.path.join(output_dir, "tree_diagram.pdf"),dpi=2000)

    if not configs["debug_mode"]:

        if not configs["decision_tree_mode"] and not configs["logreg_mode"] and not configs["mlp_mode"]:
            fimps_fname_tsv="best_model_shapley_values.tsv"
            feat_imp_vect=ml_model.feature_importances()
            col_names=ml_model.col_names()
            assert(len(col_names)==feat_imp_vect.size)

            with open(os.path.join(output_dir,fimps_fname_tsv),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["feature", "importance_score"])
                for idx in range(len(col_names)):
                    csv_fp.writerow([col_names[idx], "{}".format(feat_imp_vect[idx])])

        dump_fname="best_model.pickle"

        with open(os.path.join(output_dir, dump_fname),'wb') as fp:
            pickle.dump(ml_model.extract_raw_model(), fp)

    print("Testing...",flush=True)
    random.shuffle(test)
    n_skipped_patients=0

    test_skip_status_no_feat_labels=0
    test_skip_status_no_shapelets=0
    test_skip_status_no_static=0
    test_skip_status_pred_error=0

    time_after_full_fitting=timeit.default_timer()
    print("Seconds after full fitting: {:.3f}".format(time_after_full_fitting-time_after_begin))

    for idx,val_patient in enumerate(test):

        if (idx+1)%100==0:
            print("Patient {}/{}: {}, SKIPPED: {}".format(idx+1,len(test),val_patient,n_skipped_patients),flush=True)
            print("Skip reasons: FEAT/LABELS: {}, SHAPELETS: {}, STATIC: {}, PREDS: {}".format(test_skip_status_no_feat_labels, test_skip_status_no_shapelets,
                                                                                               test_skip_status_no_static, test_skip_status_pred_error))
            mlhc_memory.print_memory_diags()

        if configs["xinrui_subsample"] and batch_pat not in [41]:
            continue

        if configs["dataset"]=="bern" and (not configs["special_test_set"]=="NONE" or configs["special_test_set_imputed_dir"] is not None):
            batch_pat=bern_batch_map[val_patient]
            df_path=os.path.join(X_dir_special_test,"batch_{}.h5".format(batch_pat))
            df_label_path=os.path.join(y_dir_special_test,"batch_{}.h5".format(batch_pat))
        elif configs["dataset"]=="bern":
            batch_pat=bern_batch_map[val_patient]
            df_path=os.path.join(X_dir,"batch_{}.h5".format(batch_pat))
            df_label_path=os.path.join(y_dir,"batch_{}.h5".format(batch_pat))
        elif configs["dataset"] in ["mimic","mimic_only"]:
            batch_pat=mimic_batch_map[val_patient]
            df_path=os.path.join(mimic_X_dir,"batch_{}.h5".format(batch_pat))
            df_label_path=os.path.join(mimic_y_dir,"batch_{}.h5".format(batch_pat))
            df_shapelet_path=os.path.join(configs["shapelets_path"],"Shapelet_features_{}_MIMIC.h5".format(configs["split_key"]))            
        
        if not os.path.exists(df_path) or not os.path.exists(df_label_path):
            n_skipped_patients+=1
            test_skip_status_no_feat_labels+=1
            continue
        
        try:
            pat_df=pd.read_hdf(df_path,"/{}".format(val_patient),mode='r')
        except KeyError:
            n_skipped_patients+=1
            test_skip_status_no_feat_labels+=1
            continue

        if configs["add_shapelets"]:
            shapelet_df=pd.read_hdf(os.path.join(df_shapelet_path,"batch_{}.h5".format(batch_pat)), '/p{}'.format(val_patient), mode='r')
            shapelet_df["AbsDatetime"]=pd.to_datetime(shapelet_df["AbsDatetime"])
            special_cols=["AbsDatetime","PatientID"]
            sel_shapelet_cols=bern_features.yield_final_cols("only_shapelets",configs=configs)
            shapelet_cols=list(filter(lambda col: col in sel_shapelet_cols, sorted(shapelet_df.columns.values.tolist())))
            shapelet_df=shapelet_df[special_cols+shapelet_cols]

            if shapelet_df.shape[0]==0:
                n_skipped_patients+=1
                test_skip_status_no_shapelets+=1
                continue

            df_merged=pd.merge(pat_df,shapelet_df,on=["AbsDatetime","PatientID"])
            pat_df=df_merged

        try:
            pat_label_df=pd.read_hdf(df_label_path,"/{}".format(val_patient),mode='r')
        except KeyError:
            n_skipped_patients+=1
            test_skip_status_no_feat_labels+=1
            continue

        if configs["add_shapelets"]:
            pat_label_df_orig_cols=sorted(pat_label_df.columns.values.tolist())
            pat_label_df=pd.merge(pat_label_df,shapelet_df,on=["AbsDatetime","PatientID"])
            pat_label_df=pat_label_df[pat_label_df_orig_cols]

        if configs["dataset"]=="bern" and (not configs["special_test_set"]=="NONE" or configs["special_test_set_imputed_dir"] is not None):
            static_df=pd.read_hdf(os.path.join(special_test_impute_dir, "static.h5"), mode='r')
        elif configs["dataset"]=="bern":
            static_df=pd.read_hdf(os.path.join(impute_dir, "static.h5"), mode='r')
        elif configs["dataset"] in ["mimic","mimic_only"]:
            static_df=pd.read_hdf(os.path.join(mimic_impute_dir,"static.h5"), mode='r')
            
        static_df=static_df[static_df["PatientID"]==val_patient]
        
        if static_df.shape[0]<1:
            n_skipped_patients+=1
            test_skip_status_no_static+=1
            continue

        if pat_label_df.shape[0]==0 or pat_df.shape[0]==0:
            n_skipped_patients+=1
            test_skip_status_no_feat_labels+=1
            continue

        static_df=static_df[final_static_cols]

        if configs["dataset"]=="mimic":
            sel_cols_X_mimic=list(filter(lambda col: col in pat_df.columns.values.tolist(), sel_cols_X))
            X_df=pat_df[sel_cols_X_mimic]
        else:
            X_df=pat_df[sel_cols_X]

        if column_desc=="mews_score":
            pred_vect=X_mat.flatten()
        else:

            if model_type=="lightgbm_meta":
                df_pred_state0, df_pred_state1, df_pred_state2 = ml_model.predict(X_df,pat_label_df,pid=val_patient, df_static=static_df)

                for idx in range(3):
                    if not os.path.exists(os.path.join(output_dir,"from_state_{}".format(idx))):
                        os.mkdir(os.path.join(output_dir,"from_state_{}".format(idx)))
                
                if df_pred_state0 is not None:
                    df_pred_state0.to_hdf(os.path.join(output_dir,"from_state_0","batch_{}.h5".format(batch_pat)), "/p{}".format(val_patient),
                                          complevel=configs["hdf_comp_level"], complib=configs["hdf_comp_alg"])
                
                if df_pred_state1 is not None:
                    df_pred_state1.to_hdf(os.path.join(output_dir,"from_state_1", "batch_{}.h5".format(batch_pat)), "/p{}".format(val_patient),
                                          complevel=configs["hdf_comp_level"], complib=configs["hdf_comp_alg"])

                if df_pred_state2 is not None:
                    df_pred_state2.to_hdf(os.path.join(output_dir,"from_state_2", "batch_{}.h5".format(batch_pat)), "/p{}".format(val_patient),
                                          complevel=configs["hdf_comp_level"], complib=configs["hdf_comp_alg"])


            else:
                assert(X_df.shape[0]==pat_label_df.shape[0])
                df_pred=ml_model.predict(X_df, pat_label_df ,pid=val_patient, df_static=static_df)

                if df_pred is None:
                    n_skipped_patients+=1
                    test_skip_status_pred_error+=1
                    continue

                assert(df_pred.shape[0]==X_df.shape[0])

                if not configs["debug_mode"]:
                    df_pred.to_hdf(os.path.join(output_dir,"batch_{}.h5".format(batch_pat)), "/p{}".format(val_patient),
                                   complevel=configs["hdf_comp_level"], complib=configs["hdf_comp_alg"])

        gc.collect()

    print("Number of skipped test patients: {}".format(n_skipped_patients),flush=True)

    time_after_all=timeit.default_timer()
    print("Seconds after entire execution: {:.3f}".format(time_after_all-time_after_begin))

def parse_cmd_args():
    # Input paths
    BERN_ML_INPUT_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/180918" 
    MIMIC_ML_INPUT_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/ml_input/181023"
    BERN_IMPUTED_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_180918"
    MIMIC_IMPUTED_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023"
    BERN_TEMPORAL_DATA_SPLIT_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/temporal_split_180918.pickle"
    MIMIC_DATA_SPLIT_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/split_181015.pickle" 
    BERN_PID_BATCH_MAP_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle"
    MIMIC_PID_BATCH_MAP_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/id_lists/chunks_181003.pickle"
    MIMIC_ALL_PID_LIST_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/pids_with_endpoint_data.csv.181003"
    SHAPELETS_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/9_s3m/feature_matrices"
    VARENCODING_DICT_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_varencoding_map_v6.pickle"
    FSCORES_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/8_predictions/180830/reduced/temporal_5/WorseStateFromZero_0.0_8.0_normal_model_lightgbm_full/features_F_scores.tsv"
    SHAPLEY_VALUES_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/8_predictions/181108/reduced"
    BERN_GENERAL_DATA_TABLE_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/1_hdf5_consent/180704/generaldata.h5"

    SPECIAL_TEST_SET_IMPUTED_DIR=None
    SPECIAL_TEST_SET_ML_INPUT_DIR=None

    # Output paths
    OUTPUT_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/8_predictions/181109"
    LOG_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log"

    parser=argparse.ArgumentParser()

    # Argument
    parser.add_argument("--ml_model", default="lightgbm", help="Which model should produce the predictions?")
    parser.add_argument("--lhours", default=0.0, type=float, help="Left boundary of the future horizon to use")
    parser.add_argument("--rhours", default=8.0, type=float, help="Right boundary of the future horizon to use")
    parser.add_argument("--split_key", default="held_out", help="Which data split should be evaluated?")
    parser.add_argument("--mimic_split_key", default="held_out", help="Which data split of MIMIC should be evaluated?")
    parser.add_argument("--label_key", default="AllLabels", help="Which label function should be evaluated?")
    parser.add_argument("--task_key", default="WorseStateFromZero", help="Which prediction task should be fit?")
    parser.add_argument("--val_type", default="test", help="On which data-set should we output the predictions?")
    parser.add_argument("--data_mode", default="reduced", help="On which data-set type should we predict?")
    parser.add_argument("--column_set", default="normal_model_shapelets_50_percent", help="Which feature columns should be selected from the model?")
    parser.add_argument("--hdf_comp_alg", default="blosc:lz4", help="Which HDF compression algorithm should be used?")
    parser.add_argument("--hdf_comp_level", default=5, type=int, help="HDF compression level in output data-sets")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debugging mode that does not write to file-system")
    parser.add_argument("--random_state", default=2018, help="Random state to seed experiment")
    parser.add_argument("--run_mode", default="INTERACTIVE", help="Is the job executed interactive or as batch job?")
    parser.add_argument("--add_shapelets", default=False, action="store_true", help="Should shapelets be added to the current feature set?")
    parser.add_argument("--plot_tree", default=True, action="store_true", help="Should a tree be plotted to the file-system?")
    parser.add_argument("--use_xgboost", default=False, action="store_true", help="Should the Xgboost library be used?")
    parser.add_argument("--use_catboost", default=False, action="store_true", help="Should the Catboost library be used?")
    parser.add_argument("--dataset", default="bern", help="For which data-set should we perform testing?")
    parser.add_argument("--special_test_set", default="NONE", help="Split which should be used for the special-test set, if this is NONE, the actual split will be used")
    parser.add_argument("--decision_tree_baseline", default=False, action="store_true", help="If this is activated we are going to train a very simple decision tree")

    # Special run configurations
    parser.add_argument("--select_features_forward", default=False, action="store_true", help="Should be perform embedded forward feature selection?")
    parser.add_argument("--select_features_backward", default=False, action="store_true", help="Should be perform embedded recursive backward feature selection?")
    parser.add_argument("--univariate_test", default=False, action="store_true", help="Perform a univariate statistical test on the features")

    # Subsampling related
    parser.add_argument("--xinrui_subsample", default=False, action="store_true", help="Use only the Xinrui sub-sample for train/validation to make comparable")
    parser.add_argument("--negative_subsampling", default=False, action="store_true", help="Should we use negative sub-sampling?")
    parser.add_argument("--10percent_sample", default=False, action="store_true", help="Should a 10 % of the training/val patients be used?")
    parser.add_argument("--0.1percent_sample_train", default=False, action="store_true", help="Should a 0.1 % of the training patients be used?")            
    parser.add_argument("--1percent_sample_train", default=False, action="store_true", help="Should a 1 % of the training patients be used?")        
    parser.add_argument("--5percent_sample_train", default=False, action="store_true", help="Should a 5 % of the training patients be used?")    
    parser.add_argument("--10percent_sample_train", default=False, action="store_true", help="Should a 10 % of the training patients be used?")
    parser.add_argument("--20percent_sample_train", default=False, action="store_true", help="Should a 20 % of the training patients be used?")
    parser.add_argument("--50percent_sample_train", default=False, action="store_true", help="Should a 50 % of the training patients be used?")
    parser.add_argument("--1percent_sample_val", default=False, action="store_true", help="Use a 1 % sub-sample of the validations set")
    parser.add_argument("--1percent_sample", default=False, action="store_true", help="Should a 1 % of the training patients be used?")
    parser.add_argument("--verysmall_sample", default=False, action="store_true", help="Use a small sample for debug purposes")

    # Temporal generalization experiment related
    parser.add_argument("--special_year", type=int, default=-1, help="Should a special year be sub-sampled from the training data?")
    parser.add_argument("--special_year_sample", type=float, default=0.5, help="Random sub-sampling ratio from the data for special year mode, CURRENTLY NOT USED")
    parser.add_argument("--special_year_train_ratio", type=float, default=0.75, help="Training set ratio of PIDs to use from the selected year")

    parser.add_argument("--profile_report", default=False, action="store_true", help="Should a profiling report be produced?")
    parser.add_argument("--systrace_mode", default=False, action="store_true", help="Stop after loading train/val data for system call tracing")

    # Logistic regression arguments
    parser.add_argument("--logreg_alpha", type=float, default=0.1, help="Regularization parameter for logistic regression")

    # Special modes
    parser.add_argument("--decision_tree_mode", default=False, action="store_true", help="Train a simple decision tree")
    parser.add_argument("--logreg_mode", default=False, action="store_true", help="Train a simple LR classifier")
    parser.add_argument("--mlp_mode", default=False, action="store_true", help="Train a simple MLP classifier")
    parser.add_argument("--special_development_split", default="NONE", help="Provide a non-default if a special random split should be loaded")

    # Paths
    parser.add_argument("--bern_ml_input_dir", default=BERN_ML_INPUT_DIR, help="ML input directory to use for the Bern data-set")
    parser.add_argument("--bern_imputed_dir", default=BERN_IMPUTED_DIR, help="Imputation directory to use for the MIMIC data-set")
    parser.add_argument("--mimic_ml_input_dir", default=MIMIC_ML_INPUT_DIR, help="ML input directory to use for the Bern data-set")
    parser.add_argument("--mimic_imputed_dir", default=MIMIC_IMPUTED_DIR, help="Imputation directory to use for the MIMIC data-set")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory to store predictions")
    parser.add_argument("--bern_pid_batch_map_path", default=BERN_PID_BATCH_MAP_BINARY, help="Path of batch map for the Bern data-set")
    parser.add_argument("--mimic_pid_batch_map_path", default=MIMIC_PID_BATCH_MAP_BINARY, help="Path of batch map for the MIMIC data-set")
    parser.add_argument("--mimic_all_pid_list_path", default=MIMIC_ALL_PID_LIST_PATH, help="Path for the PIDs of interest in the MIMIC data-set")
    parser.add_argument("--bern_temporal_data_split_path", default=BERN_TEMPORAL_DATA_SPLIT_BINARY, help="Temporal split dir for the Bern data-set")
    parser.add_argument("--mimic_data_split_path", default=MIMIC_DATA_SPLIT_BINARY, help="Split dir for the MIMIC data-set")
    parser.add_argument("--log_dir", default=LOG_DIR, help="Logging base directory")
    parser.add_argument("--shapelets_path", default=SHAPELETS_PATH, help="Feature matrix file containing the shapelets")
    parser.add_argument("--varencoding_dict_path", default=VARENCODING_DICT_PATH, help="Variable encoding dictionary")
    parser.add_argument("--fscores_path", default=FSCORES_PATH, help="F scores computed on the features")
    parser.add_argument("--shapley_values_path", default=SHAPLEY_VALUES_PATH, help="Absolute shapley values computed on the features")
    parser.add_argument("--bern_general_data_table_path", default=BERN_GENERAL_DATA_TABLE_PATH, help="Path to the general data table for Bern PIDs")

    parser.add_argument("--special_test_set_imputed_dir", default=SPECIAL_TEST_SET_IMPUTED_DIR, help="If a special test set is used, which imputed data?")
    parser.add_argument("--special_test_set_ml_input_dir", default=SPECIAL_TEST_SET_ML_INPUT_DIR, help="If a special test set is used, which ML input data?")

    args=parser.parse_args()
    configs=vars(args)

    configs["GBM_HP_GRID"]={"n_estimators": [5000], "num_leaves": [8,16,32,64,128], "learning_rate": [0.05], 
                        "colsample_bytree": [0.33,0.66], "rowsample_bytree": [0.33,0.66]}

    # Special grid with 1 point for forward feature selection
    if configs["select_features_forward"]:
        configs["GBM_HP_GRID"]={"n_estimators": [5000], "num_leaves": [64], "learning_rate": [0.05], 
                                "colsample_bytree": [0.75], "rowsample_bytree": [0.75]}

    configs["TREE_GRID"]={"n_estimators": [1], "num_leaves": [8,16,32,64,128], "learning_rate": [0.05]}

    configs["LR_GRID"]= {"alpha": [1.0,0.1,0.01,0.001,0.0001,0.00001]}

    configs["MLP_GRID"]= {"hidden_layer_size": [10,20,50,100], "learning_rate": [0.001], "alpha": [0.01,0.001,0.0001,0.00001]}
    
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()

    run_mode=configs["run_mode"]
    split_key=configs["split_key"]
    dim_reduced_str=configs["data_mode"]
    task_key=configs["task_key"]
    lhours=configs["lhours"]
    rhours=configs["rhours"]
    model_type=configs["ml_model"]

    if run_mode=="CLUSTER":

        if model_type=="lightgbm":
            sys.stdout=open(os.path.join(configs["log_dir"],"MLFIT_{}_{}_{}_{}_{}_{}_{}.stdout".format(split_key,dim_reduced_str, task_key,lhours,rhours, model_type,configs["column_set"])),'w')
            sys.stderr=open(os.path.join(configs["log_dir"],"MLFIT_{}_{}_{}_{}_{}_{}_{}_.stderr".format(split_key,dim_reduced_str, task_key,lhours,rhours, model_type, configs["column_set"])),'w')
        elif model_type=="logreg":
            sys.stdout=open(os.path.join(configs["log_dir"],"MLFIT_{}_{}_{}_{}_{}_{}_{}_alpha{}.stdout".format(split_key,dim_reduced_str,task_key,lhours,rhours, model_type,
                                                                                                               configs["column_set"],configs["logreg_alpha"])),'w')
            sys.stderr=open(os.path.join(configs["log_dir"],"MLFIT_{}_{}_{}_{}_{}_{}_{}_alpha{}.stderr".format(split_key,dim_reduced_str, task_key,lhours,rhours,model_type, 
                                                                                                               configs["column_set"],configs["logreg_alpha"])),'w')

    learning_train_to_test(configs)

