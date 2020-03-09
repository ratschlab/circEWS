''' 
Cluster dispatch script for label generation
'''

import subprocess
import os
import os.path
import sys
import argparse

import circews.functions.util.filesystem as mlhc_fs

def cluster_labels(configs):
    ''' Computes labels for all possible impute data and label type combinations'''
    compute_script_path=configs["compute_script_path"]
    job_index=0
    subprocess.call(["source activate default_py36"],shell=True)
    mem_in_mbytes=configs["compute_mem"]
    n_cpu_cores=configs["compute_n_cores"]
    n_compute_hours=configs["compute_n_hours"]
    bad_hosts=["lm-a2-003","lm-a2-004"]

    if configs["dataset"]=="bern":
        label_base_path=configs["bern_output_label_path"]
    elif configs["dataset"]=="mimic":
        label_base_path=configs["mimic_output_label_path"]

    is_dry_run=configs["dry_run"]

    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log-files...")
        for logf in os.listdir(configs["log_dir"]):
            os.remove(os.path.join(configs["log_dir"], logf))

    for reduce_config in configs["DATA_MODES"]:
        for split_key in configs["SPLIT_MODES"]:

            if reduce_config=="reduced":
                split_base_dir=os.path.join(label_base_path, "reduced" , split_key)
            else:
                split_base_dir=os.path.join(label_base_path, split_key)

            mlhc_fs.create_dir_if_not_exist(split_base_dir)

            for label_key in ["AllLabels"]:

                if reduce_config=="reduced":
                    label_base_dir=os.path.join(label_base_path, "reduced", split_key, label_key)
                else:
                    label_base_dir=os.path.join(label_base_path, split_key, label_key)
                
                mlhc_fs.create_dir_if_not_exist(label_base_dir)

                for lhours,rhours in configs["LABEL_SCHEMAS"]:

                    for batch_idx in range(50):
            
                        print("Create label patient data for split {} with reduced data: {}, label: {} [{},{}], batch {}".format(split_key, reduce_config,label_key, lhours,rhours,batch_idx))
                        job_name="labelgen_{}_{}_{}_{}_{}_{}".format(split_key,reduce_config,label_key,lhours,rhours,batch_idx)
                        log_result_file=os.path.join(configs["log_dir"], "{}_RESULT.txt".format(job_name))
                        mlhc_fs.delete_if_exist(log_result_file)
                        cmd_line=" ".join(["bsub", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "-r", "-W", "{}:00".format(n_compute_hours),
                                           " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),                                                                             
                                           "-J","{}".format(job_name), "-o", log_result_file, "python3", compute_script_path, "--run_mode CLUSTER", "--split_key {}".format(split_key),
                                           "--data_mode {}".format(reduce_config), "--label_key {}".format(label_key), "--lhours {}".format(lhours), "--dataset {}".format(configs["dataset"]), 
                                           "--rhours {}".format(rhours), "--batch_idx {}".format(batch_idx)])
                        assert(" rm " not in cmd_line)
                        job_index+=1

                        if is_dry_run:
                            print("CMD: {}".format(cmd_line))
                        else:
                            subprocess.call([cmd_line], shell=True)

                            if configs["debug_mode"]:
                                sys.exit(0)

    print("Number of generated jobs: {}".format(job_index))

def parse_cmd_args():

    parser=argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--compute_n_hours", default=4, help="Number of compute hours to dispatch per job")
    parser.add_argument("--compute_n_cores", default=1, help="Number of compute cores to dispatch per job")
    parser.add_argument("--compute_mem", default=4000, help="Mbytes to allocate per job")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Should a dry run be run, without dispatching the jobs?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode?")
    parser.add_argument("--preserve_logs", default=False, action="store_true", help="Should logs be preserved?")
    parser.add_argument("--dataset", default="bern", help="For which data-set should label generation be run?")

    # Input paths
    parser.add_argument("--compute_script_path", default="/cluster/home/mhueser/git/projects/2016/ICUscore/mhueser/scripts/labels/label_gen.py", help="Script to execute per job")
    
    # Output paths
    parser.add_argument("--log_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log", help="Base dir for storing logging infos")
    parser.add_argument("--bern_output_label_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/6_labels/targets_v6b_downsample_upsample_binarized_rev2",
                        help="Base dir for storing output labels for the Bern data-set")
    parser.add_argument("--mimic_output_label_path", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/labels/targets_181023", 
                        help="Base dir for storing output labels for the MIMIC data-set")

    args=parser.parse_args()
    configs=vars(args)

    configs["DATA_MODES"]=["reduced"]
    configs["SPLIT_MODES"]=["held_out","temporal_1","temporal_2","temporal_3","temporal_4","temporal_5"]
    configs["LABEL_SCHEMAS"]=[(0,8)]
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    cluster_labels(configs)
