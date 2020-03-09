'''
Cluster dispatcher script for the main batched imputation script
'''

import subprocess
import os
import os.path
import sys
import argparse

import circews.functions.util.filesystem as mlhc_fs

def cluster_impute(configs):
    job_index=0
    subprocess.call(["source activate default_py36"],shell=True)
    mem_in_mbytes=configs["mem_in_mbytes"]
    n_cpu_cores=configs["n_cpu_cores"]
    n_compute_hours=configs["n_compute_hours"]
    compute_script_path=configs["compute_script_path"]
    bad_hosts=["lm-a2-003","lm-a2-004"]
    
    if configs["dataset"]=="bern":
        dset_base_dir=configs["bern_imputed_dir"]
        dset_base_dir_reduced=configs["bern_imputed_reduced_dir"]
    elif configs["dataset"]=="mimic":
        dset_base_dir=configs["mimic_imputed_dir"]
        dset_base_dir_reduced=configs["mimic_imputed_reduced_dir"]
    else:
        print("ERROR: Invalid data-set has been selected")

    for reduce_config in configs["data_configs"]:

        for split_key in configs["split_configs"]:

            if reduce_config=="reduced":
                base_dir=os.path.join(dset_base_dir_reduced,split_key)
            else:
                base_dir=os.path.join(dset_base_dir,split_key)

            if not configs["dry_run"]:
                mlhc_fs.create_dir_if_not_exist(base_dir)

            for batch_idx in range(50):

                if not configs["dry_run"]:
                    mlhc_fs.delete_if_exist(os.path.join(base_dir,"batch_{}.h5".format(batch_idx)))

                print("Impute patient data for split {} with reduced data: {}".format(split_key, reduce_config))
                job_name="imputation_{}_{}_{}".format(split_key,reduce_config,batch_idx)
                log_result_file=os.path.join(configs["log_dir"], "{}_RESULT.txt".format(job_name))
                
                if not configs["dry_run"]:
                    mlhc_fs.delete_if_exist(log_result_file)

                cmd_line=" ".join(["bsub", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "-r", "-W", "{}:00".format(n_compute_hours),
                                   " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),                                   
                                   "-J","{}".format(job_name), "-o", log_result_file, "python3", compute_script_path, "--run_mode CLUSTER", "--dataset {}".format(configs["dataset"]),
                                   "--imputation_mode {}".format(configs["imputation_mode"]), 
                                   "--split_key {}".format(split_key), "--data_mode {}".format(reduce_config), "--batch_idx {}".format(batch_idx)])
                assert(" rm " not in cmd_line)
                job_index+=1

                if configs["dry_run"]:
                    print(cmd_line)
                else:
                    subprocess.call([cmd_line], shell=True)


def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--compute_script_path", default="/cluster/home/mhueser/git/projects/2016/ICUscore/mhueser/scripts/imputation/impute_one_batch.py",
                        help="Compute script to dispatch")

    # Output paths
    parser.add_argument("--log_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log", help="Directory where to store log files")
    parser.add_argument("--bern_imputed_reduced_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_downsample_upsample_no_impute/reduced" ,
                        help="Where to store the imputed data for reduced mode for the Bern data-set")
    parser.add_argument("--bern_imputed_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_v6b_downsample_upsample_no_impute",
                        help="Where to store the imputed data for non-reduced mode for the Bern data-set")
    parser.add_argument("--mimic_imputed_reduced_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023/reduced", 
                        help="Where to store imputed data for MIMIC data-set")
    parser.add_argument("--mimic_imputed_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023",
                        help="Where to store the imputed data for non-reduced mode for the MIMIC data-set")

    # Default arguments
    parser.add_argument("--mem_in_mbytes", type=int, default=5000, help="Memory consumed by each batch")
    parser.add_argument("--n_cpu_cores", type=int, default=1, help="Number of CPU cores to use")
    parser.add_argument("--n_compute_hours", type=int, default=4, help="Number of compute hours per job")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Should the dispatcher be dry-run?")
    parser.add_argument("--dataset", default="bern", help="For which data-set should we perform imputation?")
    parser.add_argument("--imputation_mode", default="complex", help="Which imputation mode should be used?")

    args=parser.parse_args()
    configs=vars(args)

    # Constants
    configs["data_configs"]=["reduced"]

    if configs["dataset"]=="bern":
        configs["split_configs"]=["held_out","temporal_1","temporal_2","temporal_3","temporal_4","temporal_5"]
    else:
        configs["split_configs"]=["held_out"]

    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    cluster_impute(configs)
