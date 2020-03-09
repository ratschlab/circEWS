''' Imputation of a single batch of patients'''

import gc
import timeit
import os.path
import sys
import os
import os.path
import datetime
import random
import gc
import psutil
import multiprocessing as mp
import concurrent.futures as conc_futures
import time
import csv
import pickle
import glob
import argparse

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import circews.classes.imputer as bern_tf_impute
import circews.classes.imputer_ffill as bern_tf_impute_ffill
import circews.classes.imputer_none as bern_tf_impute_none

import circews.functions.util.io as mlhc_io

def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(),dtype=np.float64) >=0).all()

def impute_one_batch(configs):
    ''' Batch wrapper that loops through the patients of one the 50 batches'''
    batch_idx=configs["batch_idx"]
    split_key=configs["split_key"]
    data_split=mlhc_io.load_pickle(configs["temporal_data_split_binary"])[split_key]

    if configs["dataset"]=="bern":
        all_pids=data_split["train"]+data_split["val"]+data_split["test"]
    elif configs["dataset"]=="mimic":
        all_pids=list(map(int, mlhc_io.read_list_from_file(configs["mimic_all_pid_list_path"])))

    if configs["dataset"]=="bern":
        batch_map=mlhc_io.load_pickle(configs["bern_pid_batch_map_binary"])["chunk_to_pids"]
        merged_reduced_base_path=configs["bern_reduced_merged_path"]
        merged_base_path=configs["bern_merged_path"]
        output_reduced_base_path=configs["bern_imputed_reduced_dir"]
        output_base_path=configs["bern_imputed_dir"]
    else:
        batch_map=mlhc_io.load_pickle(configs["mimic_pid_batch_map_binary"])["chunk_to_pids"]
        merged_reduced_base_path=configs["mimic_reduced_merged_path"]
        merged_base_path=configs["mimic_merged_path"]
        output_reduced_base_path=configs["mimic_imputed_reduced_dir"]
        output_base_path=configs["mimic_imputed_dir"]

    pid_list=batch_map[batch_idx]
    selected_pids=list(set(pid_list).intersection(set(all_pids)))
    dim_reduced_data=(configs["data_mode"]=="reduced")
    n_skipped_patients=0
    no_patient_output=0
    first_write=True

    if configs["imputation_mode"]=="complex":
        print("Using complex imputation mode...")
        tf_model=bern_tf_impute.Timegridder(dim_reduced_data,max_grid_length_days=configs["max_grid_length_days"],use_adaptive_impute=True, grid_period=configs["grid_period"])
    elif configs["imputation_mode"]=="forward_filling":
        print("Using forward filling imputation mode...")
        tf_model=bern_tf_impute_ffill.TimegridderForwardFill(dim_reduced_data,max_grid_length_days=configs["max_grid_length_days"],
                                                             use_adaptive_impute=True, grid_period=configs["grid_period"])
    elif configs["imputation_mode"]=="no_impute":
        print("Using no imputation mode")
        tf_model=bern_tf_impute_none.TimegridderNoImpute(dim_reduced_data,max_grid_length_days=configs["max_grid_length_days"],
                                                         use_adaptive_impute=True, grid_period=configs["grid_period"])
    else:
        print("ERROR: Invalid imputation mode specified...")
        sys.exit(1)

    if configs["dataset"]=="bern":
        df_static=pd.read_hdf(configs["bern_static_info_path"],mode='r')
    elif configs["dataset"]=="mimic":
        df_static=pd.read_hdf(configs["mimic_static_info_path"],mode='r')

    tf_model.set_static_table(df_static)
    typical_weight_dict=np.load(configs["typical_weight_dict_path"],allow_pickle=True).item()
    tf_model.set_typical_weight_dict(typical_weight_dict)
    median_bmi_dict=np.load(configs["median_bmi_dict_path"],allow_pickle=True).item()
    tf_model.set_median_bmi_dict(median_bmi_dict)
    
    if dim_reduced_data:
        varencoding_map=mlhc_io.load_pickle(configs["meta_varencoding_map_path"])
    else:
        varencoding_map=mlhc_io.load_pickle(configs["varencoding_map_path"])

    tf_model.set_var_encoding_map(varencoding_map)

    if dim_reduced_data:
        impute_params_dir=configs["imputation_param_dict_reduced"]
    else:
        impute_params_dir=configs["imputation_param_dict"]
    
    global_impute_dict=mlhc_io.load_pickle(os.path.join(impute_params_dir,"global_vals_from_data_{}.pickle".format(split_key)))
    interval_median_dict=mlhc_io.load_pickle(os.path.join(impute_params_dir,"interval_median_{}.pickle".format(split_key)))
    interval_iqr_dict=mlhc_io.load_pickle(os.path.join(impute_params_dir,"interval_iqr_{}.pickle".format(split_key)))
    tf_model.set_imputation_params(global_impute_dict, interval_median_dict, interval_iqr_dict)

    if dim_reduced_data:
        normal_dict=mlhc_io.load_pickle(configs["meta_normalval_map_path"])
    else:
        normal_dict=mlhc_io.load_pickle(configs["normalval_map_path"])

    tf_model.set_normal_vals(normal_dict)
    first_write=True

    print("Number of patient IDs: {}".format(len(selected_pids),flush=True))
    
    for pidx,pid in enumerate(selected_pids):

        if dim_reduced_data:
            cand_files=glob.glob(os.path.join(merged_reduced_base_path,"reduced_fmat_{}_*.h5".format(batch_idx)))
        else:
            cand_files=glob.glob(os.path.join(merged_base_path,"fmat_{}_*.h5".format(batch_idx)))

        assert(len(cand_files)==1)
        source_fpath=cand_files[0]
        patient_df=pd.read_hdf(source_fpath,where="PatientID={}".format(pid))
        
        if patient_df.shape[0]==0:
            n_skipped_patients+=1

        if not is_df_sorted(patient_df,"Datetime"):
            patient_df=patient_df.sort_values(by="Datetime", kind="mergesort")
            
        imputed_df=tf_model.transform(patient_df,pid=pid)

        if imputed_df is None:
            no_patient_output+=1
            continue

        if first_write:
            append_mode=False
            open_mode='w'
        else:
            append_mode=True
            open_mode='a'

        if dim_reduced_data:
            output_dir=os.path.join(output_reduced_base_path,split_key)
        else:
            output_dir=os.path.join(output_base_path,split_key)

        if not configs["debug_mode"]:
            imputed_df.to_hdf(os.path.join(output_dir,"batch_{}.h5".format(batch_idx)),
                              configs["imputed_dset_id"],complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"],
                              format="table", append=append_mode, mode=open_mode,data_columns=["PatientID"])

        gc.collect()        
        first_write=False

        if (pidx+1)%100==0:
            print("Thread {}: {:.2f} %".format(batch_idx,(pidx+1)/len(selected_pids)*100),flush=True)
            print("Number of skipped patients: {}".format(n_skipped_patients))
            print("Number of no patients output: {}".format(no_patient_output))

    return 0


def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    BERN_STATIC_INFO_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/1a_hdf5_clean/v6b/static.h5"
    MIMIC_STATIC_INFO_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/merged/181023/static.h5"
    TYPICAL_WEIGHT_DICT_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/stephanie/typical_weight_dict.npy"
    MEDIAN_BMI_DICT_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/stephanie/median_bmi_dict.npy"
    META_VARENCODING_MAP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_varencoding_map_v6.pickle"
    VARENCODING_MAP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/varencoding_map_v6.pickle"
    IMPUTATION_PARAM_DICT_REDUCED="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/imputation_reduced_v6b"
    IMPUTATION_PARAM_DICT="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/imputation_v6b"
    META_NORMALVAL_MAP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_normalval_map_v6.pickle"
    NORMALVAL_MAP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/normalval_map_v6.pickle"
    BERN_REDUCED_MERGED_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3_merged/v6b_top18_upsampled_downsampled_rev2_new/reduced" 
    BERN_MERGED_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3_merged/v6b_top18_upsampled_downsampled_rev2_new" 
    MIMIC_REDUCED_MERGED_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/merged/181023/reduced"
    MIMIC_MERGED_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/merged/181023"
    TEMPORAL_DATA_SPLIT_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/temporal_split_180918.pickle"
    BERN_PID_BATCH_MAP_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle"
    MIMIC_PID_BATCH_MAP_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/id_lists/chunks_181023.pickle"
    MIMIC_ALL_PID_LIST_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/pids_with_endpoint_data.csv.181103"

    # Output paths
    BERN_IMPUTED_REDUCED_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_downsample_upsample_no_impute/reduced" 
    BERN_IMPUTED_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_downsample_upsample_no_impute" 
    MIMIC_IMPUTED_REDUCED_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023/reduced"
    MIMIC_IMPUTED_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023"
    LOG_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log"

    # Input paths
    parser.add_argument("--bern_static_info_path", default=BERN_STATIC_INFO_PATH, help="Path of static info to be loaded for the Bern data-set")
    parser.add_argument("--mimic_static_info_path", default=MIMIC_STATIC_INFO_PATH, help="Path of the static info to be loaded for the MIMIC data-set")
    parser.add_argument("--typical_weight_dict_path", default=TYPICAL_WEIGHT_DICT_PATH,
                        help="Dictionary of typical heights and weights that we use for our imputation schema")
    parser.add_argument("--median_bmi_dict_path", default=MEDIAN_BMI_DICT_PATH,
                        help="Dictionary of gender-specific estimated BMIs (used for weight imputation")
    parser.add_argument("--meta_varencoding_map_path", default=META_VARENCODING_MAP_PATH, 
                        help="Dictionary mapping variable IDs to variable encoding to distinguish between processing categories")
    parser.add_argument("--varencoding_map_path", default=VARENCODING_MAP_PATH,
                        help="Dictionary mapping variableIDs to variable types/encoding")
    parser.add_argument("--imputation_param_dict_reduced", default=IMPUTATION_PARAM_DICT_REDUCED, 
                        help="Directory where imputation parameters are stored, for the reduced data version")
    parser.add_argument("--imputation_param_dict", default=IMPUTATION_PARAM_DICT, 
                        help="Directory where imputation parameters are stored, for the standard data version")
    parser.add_argument("--meta_normalval_map_path", default=META_NORMALVAL_MAP_PATH, 
                        help="Dictionary mapping meta-variable IDs to normal values")
    parser.add_argument("--normalval_map_path", default=NORMALVAL_MAP_PATH, 
                        help="Dictionary mapping variableIDs to normal values")
    parser.add_argument("--bern_reduced_merged_path", default=BERN_REDUCED_MERGED_PATH, help="Current version of dimensionality-reduced data for the Bern data-set")
    parser.add_argument("--bern_merged_path", default=BERN_MERGED_PATH, help="Current version of the merged data for the Bern data-set")
    parser.add_argument("--mimic_reduced_merged_path", default=MIMIC_REDUCED_MERGED_PATH, help="Current version of the dim-reduced data for the MIMIC data-set")
    parser.add_argument("--mimic_merged_path", default=MIMIC_MERGED_PATH, help="Current version of the un-reduced data for the MIMIC data-set")
    parser.add_argument("--temporal_data_split_binary", default=TEMPORAL_DATA_SPLIT_BINARY, help="Location of the temporal split descriptor")
    parser.add_argument("--bern_pid_batch_map_binary", default=BERN_PID_BATCH_MAP_BINARY, help="Location of the PID-batch map for the Bern data-set")
    parser.add_argument("--mimic_pid_batch_map_binary", default=MIMIC_PID_BATCH_MAP_BINARY, help="Location of the PID-batch map for the MIMIC data-set")
    parser.add_argument("--mimic_all_pid_list_path", default=MIMIC_ALL_PID_LIST_PATH, help="Location of the all PID list of the MIMIC data set that should be considered")    

    # Output paths
    parser.add_argument("--bern_imputed_reduced_dir", default=BERN_IMPUTED_REDUCED_DIR, help="Where to store the imputed data for reduced mode for the Bern data-set")
    parser.add_argument("--bern_imputed_dir", default=BERN_IMPUTED_DIR, help="Where to store the imputed data for non-reduced mode for the Bern data-set")
    parser.add_argument("--mimic_imputed_reduced_dir", default=MIMIC_IMPUTED_REDUCED_DIR, help="Where to store the imputed data for reduced mode for the MIMIC data-set")
    parser.add_argument("--mimic_imputed_dir", default=MIMIC_IMPUTED_DIR, help="Where to store the imputed data for non-reduced mode for the MIMIC data-set")
    parser.add_argument("--log_dir", default=LOG_DIR, help="Location of the log directory")

    # Arguments
    parser.add_argument("--dataset", default="bern", help="For which data-set should we run imputation?")
    parser.add_argument("--batch_idx", type=int, default=10, help="On which batch should imputation be run?")
    parser.add_argument("--split_key", default="temporal_5", help="On which split should imputation be run?")
    parser.add_argument("--data_mode", default="reduced", help="Should dim-reduced data be used?")
    parser.add_argument("--imputed_dset_id", default="/imputed", help="Dataset key for imputed data")
    parser.add_argument("--hdf_comp_level", default=5, type=int, help="Compression level to store HDF5 files")
    parser.add_argument("--hdf_comp_alg", default="blosc:lz4", help="Compression algorithm to use")
    parser.add_argument("--run_mode", default="INTERACTIVE", help="Should job be run in batch or interactive mode")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode without writing data to disk")
    parser.add_argument("--grid_period", default=300.0, help="Imputation grid period")
    parser.add_argument("--max_grid_length_days", default=28, help="Number of days after which the grid should be cut off")
    parser.add_argument("--imputation_mode", default="complex", help="Which imputation schema should be used?")

    args=parser.parse_args()
    configs=vars(args)

    assert(configs["run_mode"] in ["CLUSTER", "INTERACTIVE"])
    assert(configs["data_mode"] in ["reduced", "non_reduced"])
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    split_key=configs["split_key"]
    dim_reduced_str=configs["data_mode"]
    batch_idx=configs["batch_idx"]

    if configs["run_mode"]=="CLUSTER":
        sys.stdout=open(os.path.join(configs["log_dir"],"IMPUTE_{}_{}_{}.stdout".format(split_key,dim_reduced_str, batch_idx)),'w')
        sys.stderr=open(os.path.join(configs["log_dir"],"IMPUTE_{}_{}_{}.stderr".format(split_key,dim_reduced_str, batch_idx)),'w')

    impute_one_batch(configs)




