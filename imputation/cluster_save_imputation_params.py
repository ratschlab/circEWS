'''
Cluster dispatcher for the script <save_imputation_params.py>
'''

import subprocess
import os
import os.path
import sys
import argparse

import circews.functions.util.filesystem as mlhc_fs

def cluster_save_imputation_params(configs):
    compute_script_path=configs["compute_script_path"]
    job_index=0
    mem_in_mbytes=configs["mem_in_mbytes"]
    n_cpu_cores=configs["n_cpu_cores"]
    n_compute_hours=configs["n_compute_hours"]
    bad_hosts=["lm-a2-002","lm-a2-003","lm-a2-004"]

    for reduce_config in configs["reduce_configs"]:
        for split_key in configs["split_configs"]:
            
            print("Generating imputation parameters for split {} with reduced data: {}".format(split_key, reduce_config))
            job_name="imputationparams_{}_{}".format(split_key,reduce_config)
            log_result_file=os.path.join(configs["log_dir"],"{}_RESULT.txt".format(job_name))
            mlhc_fs.delete_if_exist(log_result_file)
            cmd_line=" ".join(["bsub", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "-r", "-W", "{}:00".format(n_compute_hours),
                               " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),
                               "-J","{}".format(job_name), "-o", log_result_file, "python3", compute_script_path, "--run_mode CLUSTER",
                               "--split_key {}".format(split_key), "--data_mode {}".format(reduce_config)])
            assert(" rm " not in cmd_line)
            job_index+=1

            if configs["dry_run"]:
                print(cmd_line)
            else:
                subprocess.call([cmd_line], shell=True)

def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--compute_script_path", default="/cluster/home/mhueser/git/projects/2016/ICUscore/mhueser/scripts/imputation/save_imputation_params.py",
                        help="Script to dispatch")
    
    # Output paths
    parser.add_argument("--log_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log", help="Logging directory")

    # Arguments
    parser.add_argument("--mem_in_mbytes", type=int, default=8000, help="Number of mbytes to request")
    parser.add_argument("--n_cpu_cores", type=int, default=1, help="Number of CPU cores to use")
    parser.add_argument("--n_compute_hours", type=int, default=24, help="Number of CPU hours to request")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Should script be run in dry-run mode")

    args=parser.parse_args()
    configs=vars(args)

    configs["reduce_configs"] = ["reduced"]
    configs["split_configs"] =  ["temporal_2"]
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    cluster_save_imputation_params(configs)


