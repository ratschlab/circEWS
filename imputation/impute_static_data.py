'''
Imputes static data of all patients and saves back to file-system
'''

import os
import os.path
import sys
import argparse
import ipdb

import numpy as np
import pandas as pd

import circews.classes.imputer_static as bern_tf_impute_static
import circews.functions.util.io as mlhc_io

def impute_static_data(dim_reduced_data, split_key,configs=None):
    data_split=mlhc_io.load_pickle(configs["temporal_data_split_binary_path"])[split_key]
    train=data_split["train"]
    val=data_split["val"]
    test=data_split["test"]

    if configs["dataset"]=="bern":
        all_pids=train+val+test
        reduced_output_base_dir=configs["bern_imputed_reduced_path"]
        output_base_dir=configs["bern_imputed_path"]
    elif configs["dataset"]=="mimic":
        all_pids=map(int, mlhc_io.read_list_from_file(configs["mimic_all_pid_list_path"]))
        reduced_output_base_dir=configs["mimic_imputed_reduced_path"]
        output_base_dir=configs["mimic_imputed_path"]

    df_static_bern=pd.read_hdf(configs["bern_static_info_path"], mode='r')
    df_static_mimic=pd.read_hdf(configs["mimic_static_info_path"], mode='r')
    df_train=df_static_bern[df_static_bern["PatientID"].isin(train)]

    if configs["dataset"]=="bern":
        df_all=df_static_bern[df_static_bern["PatientID"].isin(all_pids)]
    elif configs["dataset"]=="mimic":
        df_all=df_static_mimic[df_static_mimic["PatientID"].isin(all_pids)]

    tf_model=bern_tf_impute_static.StaticDataImputer(dataset=configs["dataset"])
    df_static_imputed=tf_model.transform(df_all, df_train=df_train)

    if dim_reduced_data:
        base_dir=os.path.join(reduced_output_base_dir,split_key)
    else:
        base_dir=os.path.join(output_base_dir,split_key)

    assert(df_static_imputed.isnull().sum().values.sum()==0)

    if not configs["debug_mode"]:
        df_static_imputed.to_hdf(os.path.join(base_dir, "static.h5"),'data',complevel=configs["hdf_comp_level"], complib=configs["hdf_comp_alg"])

def parse_cmd_args():

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--temporal_data_split_binary_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/temporal_split_180918.pickle",
                        help="Path where temporal splits are stored")
    parser.add_argument("--bern_static_info_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/1a_hdf5_clean/v6b/static.h5",
                        help="Path where static information for the ICU Bern data-set is stored")
    parser.add_argument("--mimic_static_info_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/merged/181023/static.h5",
                        help="Path where static information for the MIMIC data-set is stored")
    parser.add_argument("--mimic_all_pid_list_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/pids_with_endpoint_data.csv.181103",
                        help="Path where the list of all relevant MIMIC PIDs is stored?")

    # Output paths
    parser.add_argument("--bern_imputed_reduced_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_top18_downsample_MH/reduced" ,
                        help="Path where merged imputed data is stored for the Bern data-set")
    parser.add_argument("--bern_imputed_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_top18_downsample_MH",
                        help="Path where imputed data is stored for the Bern data-set, non-reduced")
    parser.add_argument("--mimic_imputed_reduced_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023/reduced",
                        help="Path where the MIMIC imputed data is stored, reduced version")
    parser.add_argument("--mimic_imputed_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023", 
                        help="Path where the MIMIC imputed data is stored, non-reduced version")

    # Parameters
    parser.add_argument("--hdf_comp_level", type=int, default=5, help="HDF compression level")
    parser.add_argument("--hdf_comp_alg", default="blosc:lz4", help="HDF compression algorithm")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode stores nothing to disk")
    parser.add_argument("--dataset", default="bern", help="For which data-set should static imputation be run?")

    args=parser.parse_args()
    configs=vars(args)

    configs["DATA_MODES"]=["reduced"]
    configs["SPLIT_MODES"]=["held_out","temporal_1","temporal_2","temporal_3","temporal_4","temporal_5"]
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()

    for dim_reduced_str in configs["DATA_MODES"]:
        for split_key in configs["SPLIT_MODES"]:
            dim_reduced_data=(dim_reduced_str=="reduced")
            print("Imputing static data for reduced data: {} on split: {}".format(dim_reduced_data, split_key))
            impute_static_data(dim_reduced_data, split_key, configs=configs)
