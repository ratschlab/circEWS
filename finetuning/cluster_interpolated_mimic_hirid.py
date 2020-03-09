''' An experiment testing linearly interpolating the predictions of the MIMIC and HIRID models for fine-tuning'''

import argparse
import ipdb
import random
import os
import os.path
import pickle
import csv
import glob
import subprocess
import sys

import circews.functions.util.filesystem as mlhc_fs

def cluster_interpolated_mimic_hirid(configs):
    bad_hosts=["lm-a2-003","lm-a2-004"]
    job_index=0

    for ip_coeff in configs["IP_COEFFS"]:
        for val_set in configs["VAL_SETS"]:
            job_name="ipjob_{}_{}".format(ip_coeff,val_set)
            log_result_file=os.path.join(configs["log_dir"],"{}_RESULT.txt".format(job_name))
            mlhc_fs.delete_if_exist(log_result_file)

            cmd_line=" ".join(["bsub", "-R" , "span[hosts=1]", "-R", "rusage[mem={}]".format(configs["mbytes_per_job"]), 
                               "-n", "{}".format(configs["num_cpu_cores"]), "-r",
                               "-W", "{}:00".format(configs["hours_per_job"]), 
                               "-J","{}".format(job_name), "-o", log_result_file,
                               " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),                                           
                               "python3", configs["compute_script_path"], "--run_mode CLUSTER", ("--debug_mode" if configs["debug_mode"] else ""),
                               "--val_type {}".format(val_set), "--ip_coeff {}".format(ip_coeff)])
            job_index+=1

            assert(" rm " not in cmd_line)

            if configs["dry_run"]:
                print("CMD: {}".format(cmd_line))
            else:
                subprocess.call([cmd_line],shell=True)

            if configs["debug_mode"]:
                sys.exit(0)

    print("Number of generated jobs: {}".format(job_index))

def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--compute_script_path",default="/cluster/home/mhueser/git/projects/2016/ICUscore/mhueser/scripts/calibration/interpolated_mimic_hirid.py",help="Script to execute")

    # Output paths
    parser.add_argument("--log_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log", help="Logging base directory")

    # Arguments
    parser.add_argument("--mbytes_per_job", default=16000, type=int, help="How many mbytes should be allocated per job?")
    parser.add_argument("--hours_per_job", default=24, type=int, help="How many compute hours should be requested per job?")
    parser.add_argument("--num_cpu_cores", default=1, type=int, help="How many CPU cores should be used per job?")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Should a dry-run be used?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode with only job")

    configs=vars(parser.parse_args())

    configs["IP_COEFFS"]=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    configs["VAL_SETS"]=["val","test"]
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    cluster_interpolated_mimic_hirid(configs)
