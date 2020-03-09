''' 
Cluster dispatch script for ML feature generation
'''

import subprocess
import os
import os.path
import sys
import argparse

import circews.functions.util.filesystem as mlhc_fs

def cluster_ml_input(configs):
    ''' Computes ML features in the current configuration on all possible label+imputed data configurations'''

    job_index=0
    subprocess.call(["source activate default_py36"],shell=True)
    mem_in_mbytes=configs["mbytes_per_job"]
    n_cpu_cores=1
    n_compute_hours=configs["hours_per_job"]
    is_dry_run=configs["dry_run"]
    bad_hosts=["lm-a2-003","lm-a2-004"]
    
    if configs["dataset"]=="bern":
        features_output_dir=configs["bern_features_dir"]
    elif configs["dataset"]=="mimic":
        features_output_dir=configs["mimic_features_dir"]
    else:
        print("ERROR: Invalid data-set specified..")
        sys.exit(1)

    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log files...")
        for logf in os.listdir(configs["log_dir"]):
            os.remove(os.path.join(configs["log_dir"], logf))

    for reduce_config in configs["DATA_SCHEMAS"]:

        for split_key in configs["SPLIT_SCHEMAS"]:

            if reduce_config=="reduced":
                split_dir=os.path.join(features_output_dir,"reduced",split_key)
            else:
                split_dir=os.path.join(features_output_dir,split_key)

            if not is_dry_run:
                mlhc_fs.create_dir_if_not_exist(split_dir)

            for label_key in ["AllLabels"]:

                for lhours,rhours in configs["LABEL_SCHEMAS"]:

                    output_base_key="{}_{}_{}".format(label_key, float(lhours), float(rhours))
                    ml_output_dir=os.path.join(split_dir,output_base_key)

                    if not is_dry_run:
                        mlhc_fs.create_dir_if_not_exist(ml_output_dir)

                    X_output_dir=os.path.join(ml_output_dir,"X")
                    y_output_dir=os.path.join(ml_output_dir,"y")

                    if not is_dry_run:
                        mlhc_fs.create_dir_if_not_exist(X_output_dir)
                        mlhc_fs.create_dir_if_not_exist(y_output_dir)

                    for batch_idx in range(50):
            
                        print("Create features for split {} with reduced data: {}, label: {} [{},{}], batch: {}".format(split_key,reduce_config,label_key, lhours,rhours, batch_idx))
                        job_name="featgen_{}_{}_{}_{}_{}_{}_{}".format(split_key,reduce_config,label_key,lhours,rhours,batch_idx,features_output_dir.split("/")[-1])
                        log_result_file=os.path.join(configs["log_dir"],"{}_RESULT.txt".format(job_name))
                        mlhc_fs.delete_if_exist(log_result_file)
                        cmd_line=" ".join(["bsub", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "-r", "-W", "{}:00".format(n_compute_hours),
                                           " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),                                                                                                                        
                                           "-J","{}".format(job_name), "-o", log_result_file, "python3", configs["compute_script_path"], "--run_mode CLUSTER", 
                                           "--dataset {}".format(configs["dataset"]), "--missing_values_mode {}".format(configs["missing_values_mode"]),
                                           "--split_key {}".format(split_key), "--data_mode {}".format(reduce_config), "--label_key {}".format(label_key), "--lhours {}".format(lhours), 
                                           "--rhours {}".format(rhours), "--batch_idx {}".format(batch_idx)])
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
    COMPUTE_SCRIPT_PATH="/cluster/home/mhueser/git/projects/2016/ICUscore/mhueser/scripts/features/save_ml_input.py"

    # Output paths
    BERN_FEATURES_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/v6b_downsample_upsample_binarized_rev2" # DOWNSAMPLED/UPSAMPLED BINARIZED EXPERIMENT
    MIMIC_FEATURES_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/ml_input/181023"
    LOG_DIR="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log"

    # Arguments
    parser.add_argument("--mbytes_per_job", default=4000, type=int, help="How many mbytes should be allocated per job?")
    parser.add_argument("--hours_per_job", default=120, type=int, help="How many compute hours should be requested per job?")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Should a dry-run be used?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debugging mode, run only one job")
    parser.add_argument("--preserve_logs", default=False, action="store_true", help="Preserve log files")
    parser.add_argument("--dataset", default="bern", help="For which data-set should we compute features?")
    parser.add_argument("--missing_values_mode", default="finite", help="Are missing values allowed in the input?")

    # Paths
    parser.add_argument("--bern_features_dir", default=BERN_FEATURES_DIR, help="Feature output directory for the Bern data-set")
    parser.add_argument("--mimic_features_dir", default=MIMIC_FEATURES_DIR, help="Feature output directory for the MIMIC data-set")
    parser.add_argument("--log_dir", default=LOG_DIR, help="Logging base directory")
    parser.add_argument("--compute_script_path", default=COMPUTE_SCRIPT_PATH, help="Script to dispatch")

    args=parser.parse_args()
    configs=vars(args)

    configs["LABEL_SCHEMAS"]=[(0,8)]
    configs["DATA_SCHEMAS"]=["reduced"]
    configs["SPLIT_SCHEMAS"]=["held_out","temporal_1","temporal_2","temporal_3","temporal_4","temporal_5"]
    
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    cluster_ml_input(configs)
