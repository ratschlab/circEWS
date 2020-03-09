
''' ML feature generation, saving X/y for supervised learning'''

import psutil
import os
import os.path
import random
import timeit
import gc
import multiprocessing as mp
import time
import sys
import argparse

import tables
import warnings
warnings.simplefilter('ignore', tables.NaturalNameWarning)

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as sp_stats

import circews.classes.feat_gen as bern_tf_features
import circews.classes.feat_gen_nan as bern_tf_features_nan
import circews.functions.util.filesystem as mlhc_fs
import circews.functions.util.io as mlhc_io

def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(),dtype=np.float64) >=0).all()


def save_ml_input(configs):
    split_key=configs["split_key"]
    reduced_data_str=configs["data_mode"]
    label_key=configs["label_key"]
    lhours=configs["lhours"]
    rhours=configs["rhours"]
    batch_idx=configs["batch_idx"]

    assert(reduced_data_str in ["reduced", "non_reduced"])
    
    if reduced_data_str=="reduced":
        dim_reduced_data=True
    else:
        dim_reduced_data=False

    print("Job SPLIT: {}, REDUCED?: {}, LABEL: {}, INTERVAL: [{},{}], BATCH: {}".format(split_key, dim_reduced_data, label_key, lhours, rhours, batch_idx),flush=True)

    output_base_key="{}_{}_{}".format(label_key, float(lhours), float(rhours))

    if configs["dataset"]=="bern":
        imputed_base_path=configs["bern_imputed_path"]
        label_base_path=configs["bern_label_path"]
        output_base_path=configs["bern_output_features_path"]
    elif configs["dataset"]=="mimic":
        imputed_base_path=configs["mimic_imputed_path"]
        label_base_path=configs["mimic_label_path"]
        output_base_path=configs["mimic_output_features_path"]
    else:
        print("ERROR: Invalid data-set specified")
        sys.exit(1)

    if dim_reduced_data:
        fmat_dir=os.path.join(imputed_base_path,"reduced",split_key)
        lmat_path=os.path.join(label_base_path,"reduced", split_key, label_key,"{:.1f}To{:.1f}Hours".format(int(lhours),int(rhours)))
        ml_output_dir=os.path.join(output_base_path,"reduced",split_key,output_base_key)
        var_encoding_dict=mlhc_io.load_pickle(configs["meta_varenc_map_path"])
        var_parameter_dict=mlhc_io.load_pickle(os.path.join(configs["meta_varprop_map_path"],"interval_median_{}.pickle".format(split_key)))
        pharma_dict=np.load(configs["pharma_acting_period_map_path"],allow_pickle=True).item()
    else:
        fmat_dir=os.path.join(imputed_base_path,split_key)
        lmat_path=os.path.join(label_base_path,split_key, label_key, "{:.1f}To{:.1f}Hours".format(int(lhours),int(rhours)))
        ml_output_dir=os.path.join(output_base_path,split_key,output_base_key)
        var_encoding_dict=mlhc_io.load_pickle(configs["varenc_map_path"])
        var_parameter_dict=mlhc_io.load_pickle(os.path.join(configs["varprop_map_path"], "interval_median_{}.pickle".format(split_key)))

    data_split=mlhc_io.load_pickle(configs["temporal_split_path"])[split_key]

    if configs["dataset"]=="bern":
        all_pids=data_split["train"]+data_split["val"]+data_split["test"]
    elif configs["dataset"]=="mimic":
        all_pids=list(map(int, mlhc_io.read_list_from_file(configs["mimic_all_pid_list_path"])))

    if configs["verbose"]:
        print("Number of patient IDs: {}".format(len(all_pids),flush=True))

    if configs["dataset"]=="bern":
        batch_map=mlhc_io.load_pickle(configs["bern_pid_map_path"])["chunk_to_pids"]
    elif configs["dataset"]=="mimic":
        batch_map=mlhc_io.load_pickle(configs["mimic_pid_map_path"])["chunk_to_pids"]

    pids_batch=batch_map[batch_idx]
    selected_pids=list(set(pids_batch).intersection(all_pids))
    print("Number of selected PIDs for this batch: {}".format(len(selected_pids)),flush=True)
    batch_path=os.path.join(fmat_dir,"batch_{}.h5".format(batch_idx))
    n_skipped_patients=0

    if not os.path.exists(batch_path):
        print("WARNING: No input data for batch, skipping...",flush=True)
        print("Generated path: {}".format(batch_path))
        return 0

    if configs["missing_values_mode"]=="finite":
        tf_model=bern_tf_features.Features(dim_reduced_data=dim_reduced_data, 
                                           impute_grid_unit=configs["impute_grid_period_secs"],
                                           dataset=configs["dataset"])
    elif configs["missing_values_mode"]=="missing":
        tf_model=bern_tf_features_nan.FeaturesWithMissingVals(dim_reduced_data=dim_reduced_data, 
                                                              impute_grid_unit=configs["impute_grid_period_secs"],
                                                              dataset=configs["dataset"])
    else:
        print("ERROR: Invalid missing value mode specified...")
        sys.exit(1)
        
    tf_model.set_varencoding_dict(var_encoding_dict)
    tf_model.set_varparameters_dict(var_parameter_dict)
    tf_model.set_pharma_dict(pharma_dict)
    X_output_dir=os.path.join(ml_output_dir,"X")
    y_output_dir=os.path.join(ml_output_dir,"y")
    
    try:
        if not configs["debug_mode"]:
            mlhc_fs.create_dir_if_not_exist(X_output_dir, recursive=True)
            mlhc_fs.create_dir_if_not_exist(y_output_dir, recursive=True)
    except:
        print("WARNING: Race condition when creating directory from different jobs...")

    lab_path=os.path.join(lmat_path,"batch_{}.h5".format(batch_idx))

    if not os.path.exists(lab_path):
        print("WARNING: No input label data for batch, skipping...",flush=True)
        return 0

    first_write=True
    t_begin=timeit.default_timer()

    for pidx,pid in enumerate(selected_pids):
        df_pat=pd.read_hdf(batch_path,mode='r',where="PatientID={}".format(pid))
        df_label_pat=pd.read_hdf(lab_path,mode='r',where="PatientID={}".format(pid))

        if df_pat.shape[0]==0 or df_label_pat.shape[0]==0:
            n_skipped_patients+=1
            continue

        assert(df_pat.shape[0]==df_label_pat.shape[0])
        assert(is_df_sorted(df_pat,"AbsDatetime"))
        assert(is_df_sorted(df_label_pat,"AbsDatetime"))
        df_X, df_y=tf_model.transform(df_pat, df_label_pat, pid=pid)

        if df_X is None or df_y is None:
            print("WARNING: Features could not be generated for PID: {}".format(pid), flush=True)
            n_skipped_patients+=1
            continue

        assert(df_X.shape[0]==df_y.shape[0])
        assert(df_X.shape[0]==df_pat.shape[0])

        if first_write:
            open_mode='w'
        else:
            open_mode='a'

        if not configs["debug_mode"]:
            df_X.to_hdf(os.path.join(X_output_dir,"batch_{}.h5".format(batch_idx)),"/{}".format(pid),format="fixed",append=False,mode=open_mode,
                        complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"])
            df_y.to_hdf(os.path.join(y_output_dir,"batch_{}.h5".format(batch_idx)),"/{}".format(pid),format="fixed",append=False,mode=open_mode,
                        complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"])

        first_write=False

        print("Job {}: {:.2f} %".format(batch_idx, (pidx+1)/len(selected_pids)*100),flush=True)
        t_current=timeit.default_timer()
        tpp=(t_current-t_begin)/(pidx+1)
        eta_minutes=(len(selected_pids)-(pidx+1))*tpp/60.0
        print("ETA [minutes]: {:.2f}".format(eta_minutes),flush=True)
        print("Number of skipped patients: {}".format(n_skipped_patients),flush=True)

    return 0

def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    BERN_PID_MAP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle"
    MIMIC_PID_MAP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/id_lists/chunks_181023.pickle"
    TEMPORAL_SPLIT_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/temporal_split_180918.pickle"
    MIMIC_ALL_PID_LIST_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/pids_with_endpoint_data.csv.181103"
    BERN_IMPUTED_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5b_binarized/binarized_v6b_downsample_upsample_rev2" 
    MIMIC_IMPUTED_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023"
    BERN_LABEL_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/6_labels/targets_v6b_downsample_upsample_binarized_rev2" 
    MIMIC_LABEL_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/labels/targets_181023"
    META_VARENC_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/meta_varencoding_map_v6.pickle"
    VARENC_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/varencoding_map_v6.pickle"
    META_VARPROP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/imputation_reduced_v6b"
    VARPROP_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/imputation_v6b"
    PHARMA_ACTING_PERIOD_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/stephanie/pharma_acting_period_v6_meta.npy"

    # Output paths
    LOG_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log"
    BERN_OUTPUT_FEATURES_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/v6b_downsample_upsample_binarized_rev2" 
    MIMIC_OUTPUT_FEATURES_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/ml_input/181023"

    # Arguments
    parser.add_argument("--run_mode", default="INTERACTIVE", help="Should the code be run in cluster mode?")
    parser.add_argument("--split_key", default="temporal_5", help="For which split should the ML input be produced?")
    parser.add_argument("--data_mode", default="reduced", help="Which data version should be used?")
    parser.add_argument("--label_key", default="AllLabels", help="Which label should be used for the ML input?")
    parser.add_argument("--lhours", default=0, type=float, help="Left boundary of the future horizon in hours")
    parser.add_argument("--rhours", default=8, type=float, help="Right boundary of the future horizon in hours")
    parser.add_argument("--impute_grid_period_secs", default=300, type=int, help="What is the base period of the grid?")
    parser.add_argument("--batch_idx", default=41, type=int, help="Which of the patient batches should be processed in this call?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Should debug mode be used?")
    parser.add_argument("--hdf_comp_level", default=5, help="HDF compression level for new data-sets")
    parser.add_argument("--hdf_comp_alg", default="blosc:lz4", help="HDF compression algorithm for new data-sets")
    parser.add_argument("--verbose", default=False, action="store_true", help="Should verbose logging messages be printed?")
    parser.add_argument("--dataset", default="bern", help="For which data-set should features be computed?")
    parser.add_argument("--missing_values_mode", default="finite", help="In mode <finite>, finite input values are expected, <missing> allows NANs")

    # Paths
    parser.add_argument("--log_dir", default=LOG_DIR, help="Logging directory for cluster mode")
    parser.add_argument("--bern_pid_map_path", default=BERN_PID_MAP_PATH, help="Path of the patient ID batch map")
    parser.add_argument("--mimic_pid_map_path", default=MIMIC_PID_MAP_PATH, help="Path of the patient ID batch map")
    parser.add_argument("--temporal_split_path", default=TEMPORAL_SPLIT_PATH, help="Path of the temporal split descriptor")
    parser.add_argument("--bern_imputed_path", default=BERN_IMPUTED_PATH, help="Path of the impute directory from which to load the input? [Bern data-set]")
    parser.add_argument("--mimic_imputed_path", default=MIMIC_IMPUTED_PATH, help="Path of the impute directory from which to load the input? [MIMIC data-set]")
    parser.add_argument("--bern_label_path", default=BERN_LABEL_PATH, help="Path of the label directory/version from which to load the input? [Bern data-set]")
    parser.add_argument("--mimic_label_path", default=MIMIC_LABEL_PATH, help="Path of the label directory/version from which to load the input? [MIMIC data-set]")
    parser.add_argument("--bern_output_features_path", default=BERN_OUTPUT_FEATURES_PATH, help="Path of the output directory where to store features for the Bern data-set")
    parser.add_argument("--mimic_output_features_path", default=MIMIC_OUTPUT_FEATURES_PATH, help="Path of the output directory where to store features for the MIMIC data-set")
    parser.add_argument("--meta_varenc_map_path", default=META_VARENC_PATH, help="Location of the meta variable encoding map to use")
    parser.add_argument("--varenc_map_path", default=VARENC_PATH, help="Location of the variable encoding map to use")
    parser.add_argument("--meta_varprop_map_path", default=META_VARPROP_PATH, help="Location of the meta variable property map to use")
    parser.add_argument("--varprop_map_path", default=VARPROP_PATH, help="Location of the variable property map to use")
    parser.add_argument("--pharma_acting_period_map_path", default=PHARMA_ACTING_PERIOD_PATH, help="Location of the pharma acting period map")
    parser.add_argument("--mimic_all_pid_list_path", default=MIMIC_ALL_PID_LIST_PATH, help="List of all PIDs that should be processed for the MIMIC data-set")

    args=parser.parse_args()
    configs=vars(args)
    return configs


if __name__=="__main__":
    configs=parse_cmd_args()
    
    run_mode=configs["run_mode"]
    split_key=configs["split_key"]
    dim_reduced_str=configs["data_mode"]
    label_key=configs["label_key"]
    lhours=configs["lhours"]
    rhours=configs["rhours"]
    batch_idx=configs["batch_idx"]
    
    if configs["dataset"]=="bern":
        output_features_path=configs["bern_output_features_path"]
    elif configs["dataset"]=="mimic":
        output_features_path=configs["mimic_output_features_path"]

    if run_mode=="CLUSTER":
        sys.stdout=open(os.path.join(configs["log_dir"],"FEATGEN_{}_{}_{}_{}_{}_{}_{}.stdout".format(split_key,dim_reduced_str, label_key,
                                                                                                     lhours,rhours, batch_idx,output_features_path.split("/")[-1])),'w')
        sys.stderr=open(os.path.join(configs["log_dir"],"FEATGEN_{}_{}_{}_{}_{}_{}_{}.stderr".format(split_key,dim_reduced_str, label_key,
                                                                                                     lhours,rhours, batch_idx,output_features_path.split("/")[-1])),'w')

    save_ml_input(configs)


    


