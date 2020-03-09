'''
Generates data splits, we hold-out the last 10 % of patients
in absolute time with respect to admission time, and the 5 temporal 
splits are defined by absolute time of admission.
'''

import random
import csv
import argparse

import numpy as np
import pandas as pd

import circews.functions.util.io as mlhc_io
import circews.functions.static as bern_static

def generate_patient_splits(configs):

    if configs["dataset"]=="bern":
        all_pids=list(map(int,mlhc_io.read_list_from_file(configs["bern_pid_included_base_list_path"],skip_header=True)))
    elif configs["dataset"]=="mimic":
        all_pids=list(map(int, mlhc_io.read_list_from_file(configs["mimic_all_pid_list_path"])))

    print("Initial number of PIDs: {}".format(len(all_pids)))

    held_out_test_set_ratio=configs["held_out_test_set_ratio"]

    if configs["dataset"]=="mimic":
        random.seed(configs["random_state"])
        random.shuffle(all_pids)
        split_point=int(len(all_pids)*held_out_test_set_ratio)
        held_out_test=all_pids[split_point:]
        rem_pids=all_pids[:split_point]
        held_out_train=rem_pids[:int(configs["mimic_training_set_ratio"]/(configs["mimic_training_set_ratio"]+configs["mimic_val_set_ratio"])*len(rem_pids))]
        held_out_val=rem_pids[int(configs["mimic_training_set_ratio"]/(configs["mimic_training_set_ratio"]+configs["mimic_val_set_ratio"])*len(rem_pids)):]
        assert(len(held_out_train)+len(held_out_val)+len(held_out_test)==len(all_pids))
        train_split_point=configs["mimic_training_set_ratio"]
        val_split_point=configs["mimic_val_set_ratio"]+configs["mimic_training_set_ratio"]
        out_dict={}
        local_dict={}
        local_dict["train"]=held_out_train
        local_dict["val"]=held_out_val
        local_dict["test"]=held_out_test
        out_dict["held_out"]=local_dict

        for sidx in range(configs["mimic_n_replicates"]):
            random.shuffle(rem_pids)
            train_pids=rem_pids[:int(train_split_point*len(rem_pids))]
            val_pids=rem_pids[int(train_split_point*len(rem_pids)):int(val_split_point*len(rem_pids))]
            test_pids=rem_pids[int(val_split_point*len(rem_pids)):]
            assert(len(rem_pids)==len(train_pids)+len(val_pids)+len(test_pids))
            local_dict={}
            local_dict["train"]=train_pids
            local_dict["val"]=val_pids
            local_dict["test"]=test_pids
            out_dict["random_{}".format(sidx)]=local_dict

        if not configs["debug_mode"]:
            mlhc_io.save_pickle(out_dict, configs["mimic_data_split_binary_path"])

        return

    with_endpoint_pids=[]
    adm_lookup_dict={}

    with open(configs["bern_endpoint_list_path"], 'r') as fp:
        csv_fp=csv.reader(fp,delimiter=',')
        for pid,_ in csv_fp:
            with_endpoint_pids.append(int(pid.strip()))

    all_pids=list(set(all_pids).intersection(set(with_endpoint_pids)))
    print("Number of PIDs after no endpoint exclusion: {}".format(len(all_pids)))

    first_date_map=[]
    df_patient_full=pd.read_hdf(configs["bern_general_data_table_path"], mode='r')
    static_pids=list(df_patient_full.PatientID.unique())

    all_pids=list(set(all_pids).intersection(set(static_pids)))
    print("Number of PIDs after excluding PIDs without static information: {}".format(len(all_pids)))

    for pidx,pid in enumerate(all_pids):
        if (pidx+1) % 1000 == 0:
            print("Patient {}/{}".format(pidx+1,len(all_pids)))
        adm_time=bern_static.lookup_admission_time(pid, df_patient_full)
        first_date_map.append((pid,adm_time))
        adm_lookup_dict[pid]=adm_time

    print("Generating temporal splits...")
    sorted_pids_with_times=list(sorted(first_date_map, key=lambda item: item[1]))
    sorted_pids=list(map(lambda item: item[0], sorted_pids_with_times))
    split_point=int(len(sorted_pids)*held_out_test_set_ratio)
    held_out_test=sorted_pids[split_point:]
    print("Number of PIDs in super held-out-test set: {}".format(len(held_out_test)))
    rem_pids=sorted_pids[:split_point]
    print("Number of remaining PIDs for temporal splits: {}".format(len(rem_pids)))
    abs_time_T=sorted_pids_with_times[split_point][1]
    current_T=abs_time_T
    T_minus_3=current_T-np.timedelta64(3*365*24,'h')
    first_adm_time=sorted_pids_with_times[0][1]
    K=T_minus_3-first_adm_time
    long_format_dict={}
    out_dict={}
    for pid in sorted_pids:
        long_format_dict[pid]=[]

    for tsplit_idx in range(5):
        local_dict={}
        half_year_T=current_T-np.timedelta64(182*24+12,'h')
        begin_year_T=current_T-np.timedelta64(365*24,'h')
        begin_all_T=begin_year_T-K
        test_idxs=list(map(lambda item: item[0], filter(lambda item: item[1]>=half_year_T and item[1]<current_T, sorted_pids_with_times)))
        val_idxs=list(map(lambda item: item[0], filter(lambda item: item[1]>=begin_year_T and item[1]<half_year_T, sorted_pids_with_times)))
        train_idxs=list(map(lambda item: item[0], filter(lambda item: item[1]>=begin_all_T and item[1]<begin_year_T, sorted_pids_with_times)))
        local_dict["train"]=train_idxs
        local_dict["val"]=val_idxs
        local_dict["test"]=test_idxs
        
        for pid in train_idxs:
            long_format_dict[pid].append("train")
        for pid in val_idxs:
            long_format_dict[pid].append("val")
        for pid in test_idxs:
            long_format_dict[pid].append("test")
        for pid in set(sorted_pids).difference(set(train_idxs+val_idxs+test_idxs)):
            long_format_dict[pid].append("-")

        out_dict["temporal_{}".format(5-tsplit_idx)]=local_dict
        current_T=half_year_T

    local_dict={}
    local_dict["train"]=rem_pids
    local_dict["val"]=held_out_test[:int(0.5*len(held_out_test))]
    local_dict["test"]=held_out_test[int(0.5*len(held_out_test)):]
    out_dict["held_out"]=local_dict

    for pid in local_dict["train"]:
        long_format_dict[pid].append("train")
    for pid in local_dict["val"]:
        long_format_dict[pid].append("val")
    for pid in local_dict["test"]:
        long_format_dict[pid].append("test")
    for pid in set(sorted_pids).difference(set(local_dict["train"]+local_dict["val"]+local_dict["test"])):
        long_format_dict[pid].append("-")

    train_ratio=0.8
    val_ratio=0.1
    test_ratio=0.1

    for rdx in range(5):
        print("Generating exploration split: {}/{}".format(rdx+1,5))
        random.shuffle(rem_pids)
        explore_train=rem_pids[:int(train_ratio*len(rem_pids))]
        explore_val=rem_pids[int(train_ratio*len(rem_pids)):int((train_ratio+val_ratio)*len(rem_pids))]
        explore_test=rem_pids[int((train_ratio+val_ratio)*len(rem_pids)):]
        local_dict={}
        local_dict["train"]=explore_train
        local_dict["val"]=explore_val
        local_dict["test"]=explore_test
        out_dict["exploration_{}".format(rdx+1)]=local_dict

        for pid in local_dict["train"]:
            long_format_dict[pid].append("train")
        for pid in local_dict["val"]:
            long_format_dict[pid].append("val")
        for pid in local_dict["test"]:
            long_format_dict[pid].append("test")
        for pid in set(sorted_pids).difference(set(local_dict["train"]+local_dict["val"]+local_dict["test"])):
            long_format_dict[pid].append("-")

    pid_keys=list(sorted(long_format_dict.keys()))

    if configs["debug_mode"]:
        return

    with open(configs["bern_temporal_data_split_text_path"],'w') as fp:
        csv_fp=csv.writer(fp, delimiter='\t')
        csv_fp.writerow(["pid","adm_time"]+["temporal_{}".format(split_idx) for split_idx in range(5,0,-1)]+["held_out"]+["exploration_{}".format(base_idx+1) for base_idx in range(5)])
        for pid in pid_keys:
            csv_fp.writerow([pid,adm_lookup_dict[pid]]+long_format_dict[pid])

    mlhc_io.save_pickle(out_dict, configs["bern_temporal_data_split_binary_path"])

def parse_cmd_args():

    # Input paths
    BERN_PID_INCLUDED_BASE_LIST_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_filtered.csv"
    BERN_ENDPOINT_LIST_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/pipeline_diagnostics/patients_with_endpoints_v6b.txt"
    BERN_GENERAL_DATA_TABLE_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/1_hdf5_consent/180704/generaldata.h5"
    MIMIC_ALL_PID_LIST_PATH="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/pids_with_endpoint_data.csv.181003"

    # Output paths
    BERN_TEMPORAL_DATA_SPLIT_TEXT="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/aux_exploration_split_190105.tsv"
    BERN_TEMPORAL_DATA_SPLIT_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/aux_exploration_split_190105.pickle"
    MIMIC_DATA_SPLIT_BINARY="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/aux_exploration_split_190105.pickle"

    parser=argparse.ArgumentParser()

    # Paths 
    parser.add_argument("--bern_pid_included_base_list_path", default=BERN_PID_INCLUDED_BASE_LIST_PATH, 
                        help="Base list of included PIDs from which we subtract for the Bern data-set")
    parser.add_argument("--bern_endpoint_list_path", default=BERN_ENDPOINT_LIST_PATH, help="PIDs that have endpoints and should be included")
    parser.add_argument("--bern_general_data_table_path", default=BERN_GENERAL_DATA_TABLE_PATH,
                        help="General data table from the original DBMS")
    parser.add_argument("--bern_temporal_data_split_text_path", default=BERN_TEMPORAL_DATA_SPLIT_TEXT, help="Temporal data split descriptor (text format) for the Bern data-set")
    parser.add_argument("--bern_temporal_data_split_binary_path", default=BERN_TEMPORAL_DATA_SPLIT_BINARY, help="Temporal data split descriptor (binary format) for the Bern data-set")
    parser.add_argument("--mimic_data_split_binary_path", default=MIMIC_DATA_SPLIT_BINARY, help="Temporal data split descriptor (binary format) for the MIMIC data-set")
    parser.add_argument("--mimic_all_pid_list_path", default=MIMIC_ALL_PID_LIST_PATH, help="Path for the PIDs of interest in the MIMIC data-set")

    # Arguments
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debugging mode, no output to file-system")
    parser.add_argument("--dataset", default="bern", help="For which data-set should we create splits?")
    parser.add_argument("--held_out_test_set_ratio", type=float, default=0.9, help="Ratio of data to retain after excluding held-out PIDs")
    parser.add_argument("--mimic_n_replicates", type=int, default=5, help="How many replicates should be produced for the MIMIC data-set?")
    parser.add_argument("--mimic_training_set_ratio", type=float, default=0.6, help="Ratio of entire data-set that should be used for training")
    parser.add_argument("--mimic_val_set_ratio", type=float, default=0.2, help="Ratio of entire data set that should be used for validation")
    parser.add_argument("--random_state", type=int, default=2018, help="Random seed for generation of splits for MIMIC")

    args=parser.parse_args()
    configs=vars(args)
    return configs
    

if __name__=="__main__":
    configs=parse_cmd_args()
    generate_patient_splits(configs)
