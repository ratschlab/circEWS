'''
Generates a new PID list from the batch patient map and also saves binary 
versions of the PID and batch maps.
'''

import csv
import pickle
import argparse
import sys

def chunkfiles_to_binary(configs):
    fw_dict = {}
    rw_dict = {}
    pid_lst = []

    if configs["dataset"] == "bern":
        input_file = configs["bern_chunk_file_input"]
        output_file = configs["bern_chunk_file_output"]
    elif configs["dataset"] == "mimic":
        input_file = configs["mimic_chunk_file_input"]
        output_file = configs["mimic_chunk_file_output"]
    else:
        print("ERROR: Invalid data-set specified")
        sys.exit(1)

    with open(input_file, 'r') as fp:
        csv_fp = csv.reader(fp, delimiter=',')
        next(csv_fp)
        for pid, chunk_idx in csv_fp:
            chunk_key = int(chunk_idx.strip())
            pid_key = int(pid.strip())

            if chunk_key not in fw_dict:
                fw_dict[chunk_key] = []

            fw_dict[chunk_key].append(pid_key)
            rw_dict[pid_key] = chunk_key
            pid_lst.append(pid_key)

    pickle_obj = {"chunk_to_pids": fw_dict, "pid_to_chunk": rw_dict}

    with open(output_file, 'wb') as fp:
        pickle.dump(pickle_obj, fp)


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--bern_chunk_file_input", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.csv", 
                        help="Chunk file text version input for the Bern database")
    parser.add_argument("--mimic_chunk_file_input", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/chunks.csv.181023",
                        help="Chunk file text version input for the MIMIC database")

    # Output paths
    parser.add_argument("--bern_chunk_file_output", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle",
                        help="Chunk file pickle version output for the Bern database")
    parser.add_argument("--mimic_chunk_file_output", default="/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/id_lists/chunks_181023.pickle",
                        help="Chunk file pickle version output for the MIMIC database")

    # Arguments
    parser.add_argument("--dataset", default="bern", help="For which data-set should we construct in this run?")

    args = parser.parse_args()
    configs = vars(args)
    return configs



if __name__ == "__main__":
    configs=parse_cmd_args()
    chunkfiles_to_binary(configs)
