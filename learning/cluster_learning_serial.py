''' 
Cluster dispatch script for ML.
'''

import subprocess
import os
import os.path
import sys
import argparse
import itertools
import random
import ipdb
import glob
import csv
import json

import circews.functions.util.filesystem as mlhc_fs

def cluster_learning_serial(configs):
    LABEL_SCHEMAS=[(0,8)] 
    ALL_TASKS=["WorseStateFromZero"]
    random.seed(configs["random_seed"])
    job_index=0
    subprocess.call(["source activate default_py36"],shell=True)
    mem_in_mbytes=configs["mbytes_per_job"]
    n_cpu_cores=configs["num_cpu_cores"]
    n_compute_hours=configs["hours_per_job"]
    is_dry_run=configs["dry_run"]
    ml_model=configs["ml_model"]
    col_desc=configs["col_desc"]
    bad_hosts=["lm-a2-003","lm-a2-004"]

    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log files...")
        for logf in os.listdir(configs["log_dir"]):
            os.remove(os.path.join(configs["log_dir"], logf))

    for reduce_config in configs["DATA_CONFIGS"]:
        for split_key in configs["SPLIT_CONFIGS"]:

            if reduce_config=="reduced":
                split_dir=os.path.join(configs["pred_dir"], "reduced",split_key)
            else:
                split_dir=os.path.join(configs["pred_dir"],split_key)

            if not is_dry_run:
                mlhc_fs.create_dir_if_not_exist(split_dir)

            for task_key in ALL_TASKS:
                for lhours,rhours in LABEL_SCHEMAS:

                    if configs["xinrui_subsample"]:
                        output_base_key="{}_{}_{}_{}_{}_xinrui".format(task_key, float(lhours), float(rhours), col_desc, ml_model)
                    else:
                        output_base_key="{}_{}_{}_{}_{}_full".format(task_key, float(lhours), float(rhours), col_desc, ml_model)

                    pred_output_dir=os.path.join(split_dir,output_base_key)

                    if not is_dry_run:
                        mlhc_fs.create_dir_if_not_exist(pred_output_dir)

                    print("Fit ML model for split {} with reduced data: {}, task: {} [{},{}], ML model: {}".format(split_key,reduce_config,task_key,
                                                                                                                   lhours,rhours, ml_model))

                    job_name="mlfit_{}_{}_{}_{}_{}_{}_{}".format(configs["col_desc"],split_key,reduce_config,task_key,lhours,rhours,ml_model)
                    log_result_file=os.path.join(configs["log_dir"],"{}_RESULT.txt".format(job_name))
                    mlhc_fs.delete_if_exist(log_result_file)

                    if ml_model=="lightgbm":
                        cmd_line=" ".join(["bsub", "-R" , "span[hosts=1]", "-R", "rusage[mem={}]".format(mem_in_mbytes), 
                                           "-n", "{}".format(n_cpu_cores), "-r",
                                           "-W", "{}:00".format(n_compute_hours), 
                                           "-J","{}".format(job_name), "-o", log_result_file,
                                           " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),                                           
                                           "python3", configs["compute_script_path"], "--run_mode CLUSTER",
                                           "--ml_model {}".format(ml_model), "--split_key {}".format(split_key), "--data_mode {}".format(reduce_config),
                                           "--special_development_split {}".format(configs["special_development_split"]),
                                           "--column_set {}".format(configs["col_desc"]), ("--add_shapelets" if configs["add_shapelets"] else ""),
                                           ("--negative_subsampling" if configs["negative_subsampling"] else ""), ("--use_catboost" if configs["use_catboost"] else ""),
                                           ("--50percent_sample_train" if configs["50percent_sample_train"] else ""),
                                           ("--20percent_sample_train" if configs["20percent_sample_train"] else ""),
                                           ("--10percent_sample_train" if configs["10percent_sample_train"] else ""),
                                           ("--5percent_sample_train" if configs["5percent_sample_train"] else ""),
                                           ("--1percent_sample_train" if configs["1percent_sample_train"] else ""),
                                           ("--0.1percent_sample_train" if configs["0.1percent_sample_train"] else ""),
                                           ("--decision_tree_mode" if configs["decision_tree_mode"] else ""),
                                           ("--logreg_mode" if configs["logreg_mode"] else ""),
                                           ("--mlp_mode" if configs["mlp_mode"] else ""), 
                                           ("--decision_tree_baseline" if configs["decision_tree_baseline"] else ""),
                                           ("--1percent_sample" if configs["1percent_sample"] else ""),
                                           "--dataset {}".format(configs["dataset"]),
                                           "--mimic_split_key {}".format(configs["mimic_split_key"]),
                                           ("--special_year {}".format(configs["special_year"]) if configs["special_year"] is not None else ""),
                                           "--special_test_set {}".format(configs["special_test_set"]),
                                           "--task_key {}".format(task_key), "--lhours {}".format(lhours), 
                                           "--rhours {}".format(rhours), "--ml_model {}".format(ml_model)])

                    elif ml_model=="logreg":
                        cmd_line=" ".join(["bsub", "-R" , "span[hosts=1]", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-R", "rusage[ngpus_excl_t=1]",
                                           "-n", "{}".format(n_cpu_cores), "-r",
                                           "-W", "{}:00".format(n_compute_hours), 
                                           "-J","{}".format(job_name), "-o", log_result_file, "python3", configs["compute_script_path"], "--run_mode CLUSTER",
                                           "--ml_model {}".format(ml_model), "--split_key {}".format(split_key), "--data_mode {}".format(reduce_config),
                                           "--column_set {}".format(configs["col_desc"]), "--logreg_alpha {}".format(best_hps["alpha"]),
                                           "--task_key {}".format(task_key), "--lhours {}".format(lhours), 
                                           "--rhours {}".format(rhours)])

                    assert(" rm " not in cmd_line)
                    job_index+=1

                    if configs["dry_run"]:
                        print("CMD: {}".format(cmd_line))
                    else:
                        subprocess.call([cmd_line], shell=True)

                        if configs["debug_mode"]:
                            sys.exit(0)
                        

    print("Generated {} jobs...".format(job_index))

def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    COMPUTE_SCRIPT_PATH="/cluster/home/mhueser/git/projects/2016/ICUscore/mhueser/scripts/learning/learning_serial.py"

    # Output paths
    PRED_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/8_predictions/181108"
    LOG_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log"

    # Arguments
    parser.add_argument("--mbytes_per_job", default=80000, type=int, help="How many mbytes should be allocated per job?")
    parser.add_argument("--hours_per_job", default=120, type=int, help="How many compute hours should be requested per job?")
    parser.add_argument("--num_cpu_cores", default=1, type=int, help="How many CPU cores should be used per job?")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Should a dry-run be used?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debugging mode, run only one job")
    parser.add_argument("--ml_model", default="lightgbm", help="ML model for which to dispatch jobs")
    parser.add_argument("--col_desc", default="normal_model_shapelets_50_percent", help="Feature column set to use")
    parser.add_argument("--random_seed", type=int, default=2018, help="Random seed to use")
    parser.add_argument("--hp_grid_size", type=int, default=10, help="Number of random points on the HP grid to draw")
    parser.add_argument("--xinrui_subsample", default=False, action="store_true", help="Should the sub-sample by Xinrui be used?")
    parser.add_argument("--preserve_logs", default=False, action="store_true", help="Should logging files be preserved?")
    parser.add_argument("--add_shapelets", default=False ,action="store_true", help="Should shapelets be added?") 
    parser.add_argument("--negative_subsampling", default=False, action="store_true", help="Should we use negative sub-sampling?")
    parser.add_argument("--1percent_sample", default=False, action="store_true", help="Should a 1 % sample of train/val be used, for debugging")
    parser.add_argument("--0.1percent_sample_train", default=False, action="store_true", help="Should we use 0.1 % random patients from the training set?")
    parser.add_argument("--1percent_sample_train", default=False, action="store_true", help="Should we use 1 % random patients from the training set?")
    parser.add_argument("--5percent_sample_train", default=False, action="store_true", help="Should we use 5 % random patients from the training set?")
    parser.add_argument("--10percent_sample_train", default=False, action="store_true", help="Should we use 10 % random patients from the training set?")
    parser.add_argument("--20percent_sample_train", default=False, action="store_true", help="Should we use 20 % random patients from the training set?")
    parser.add_argument("--50percent_sample_train", default=False, action="store_true", help="Should we use 50 % random patients from the training set?")
    parser.add_argument("--use_catboost", default=False, action="store_true", help="Should we use the Catboost library?")
    parser.add_argument("--dataset", default="bern", help="For which data-set should we perform testing?")
    parser.add_argument("--special_test_set", default="NONE", help="Should a test-set from another split be used")
    parser.add_argument("--decision_tree_baseline", default=False, action="store_true",
                        help="Should a very simple decision tree baseline be trained?")
    parser.add_argument("--mimic_split_key", default="held_out", help="MIMIC split key to use")
    parser.add_argument("--decision_tree_mode", default=False, action='store_true', help="Train a simple decision tree instead of LightGBM model")
    parser.add_argument("--logreg_mode", default=False, action="store_true", help="Train a simple logistic regression model instead of LightGBM")
    parser.add_argument("--mlp_mode", default=False, action="store_true", help="Train a MLP model instead of LightGBM")
    parser.add_argument("--special_development_split", default="NONE", help="Provide non-default if a special development split should be loaded")

    # Temporal generalization experiment
    parser.add_argument("--special_year", default=None, help="Special year to use for the temporal generalization experiment for train/val sets")

    # Paths
    parser.add_argument("--pred_dir", default=PRED_DIR, help="Predictions output directory")
    parser.add_argument("--log_dir", default=LOG_DIR, help="Logging base directory")
    parser.add_argument("--compute_script_path", default=COMPUTE_SCRIPT_PATH, help="Script to dispatch")

    args=parser.parse_args()
    configs=vars(args)

    configs["DATA_CONFIGS"]=["reduced"]
    configs["SPLIT_CONFIGS"]=["held_out","temporal_1","temporal_2","temporal_3","temporal_4","temporal_5"]
    
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    cluster_learning_serial(configs)
