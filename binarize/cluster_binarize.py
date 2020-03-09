''' Binarize one batch of imputed matrices'''

import argparse
import os
import os.path
import subprocess

import circews.functions.util.filesystem as mlhc_fs

def cluster_binarize(configs):
    compute_script_path=configs["compute_script_path"]
    job_index=0
    subprocess.call(["source activate default_py36"],shell=True)
    mem_in_mbytes=configs["compute_mem"]
    n_cpu_cores=configs["compute_n_cores"]
    n_compute_hours=configs["compute_n_hours"]
    bad_hosts=["lm-a2-003","lm-a2-004"]
    output_base_path=configs["binarized_dir"]
    is_dry_run=configs["dry_run"]

    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log-files...")
        for logf in os.listdir(configs["log_dir"]):
            os.remove(os.path.join(configs["log_dir"], logf))

    for reduce_config in configs["DATA_MODES"]:
        for split_key in configs["SPLIT_MODES"]:

            if reduce_config=="reduced":
                split_base_dir=os.path.join(output_base_path, "reduced" , split_key)
            else:
                split_base_dir=os.path.join(output_base_path, split_key)

            mlhc_fs.create_dir_if_not_exist(split_base_dir)

            for batch_idx in range(50):

                print("Create binarized patient data for split {} with reduced data: {}, batch {}".format(split_key, reduce_config,batch_idx))
                job_name="binarize_{}_{}_{}".format(split_key,reduce_config,batch_idx)
                log_result_file=os.path.join(configs["log_dir"], "{}_RESULT.txt".format(job_name))
                mlhc_fs.delete_if_exist(log_result_file)
                cmd_line=" ".join(["bsub", "-R", "rusage[mem={}]".format(mem_in_mbytes), "-n", "{}".format(n_cpu_cores), "-r", "-W", "{}:00".format(n_compute_hours),
                                   " ".join(['-R "select[hname!=\'{}\']"'.format(bad_host) for bad_host in bad_hosts]),                                                                             
                                   "-J","{}".format(job_name), "-o", log_result_file, "python3", compute_script_path, "--run_mode CLUSTER", "--split_key {}".format(split_key),
                                   "--data_mode {}".format(reduce_config),"--batch_idx {}".format(batch_idx)])
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

    # Input paths
    parser.add_argument("--compute_script_path", default="/cluster/home/mhueser/git/projects/2016/ICUscore/mhueser/scripts/binarize/binarize_one_batch.py", help="Script to execute per job")

    # Output paths
    parser.add_argument("--log_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/mhueser/log", help="Location of the log directory")
    parser.add_argument("--binarized_dir", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5b_binarized/binarized_v6b_downsample_upsample_rev2",
                        help="Binarized data-set with 1 for observed values, 0 otherwise")

    # Arguments
    parser.add_argument("--data_mode", default="reduced", help="Should dim-reduced data be used?")
    parser.add_argument("--compute_n_hours", default=4, help="Number of compute hours to dispatch per job")
    parser.add_argument("--compute_n_cores", default=1, help="Number of compute cores to dispatch per job")
    parser.add_argument("--compute_mem", default=16000, help="Mbytes to allocate per job")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Should a dry run be run, without dispatching the jobs?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode?" )   
    parser.add_argument("--preserve_logs", default=False, action="store_true", help="Preserve log files?")

    configs=vars(parser.parse_args())

    configs["DATA_MODES"]=["reduced"]
    configs["SPLIT_MODES"]=["held_out", "temporal_1","temporal_2","temporal_3","temporal_4","temporal_5"]
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    cluster_binarize(configs)


    
