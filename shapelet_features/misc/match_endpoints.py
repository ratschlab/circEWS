#!/usr/bin/env python3
#
# The purpose of this script is to match time series of patients with
# their corresponding endpoints. The problem with these files is that
# different batches of time series correspond to different batches in
# the endpoints file. This script will match both files, resulting in
# one dictionary that stores the two files for each patient ID. If no
# matching files can be found, a key of 'None' will be stored.
#
# Environment:
#
# module load hdf5
# module load python_gpu/3.6.4

import h5py as h5
import pandas as pd

import gc
import glob
import os

# All batches of processed time series data. This directory is supposed
# to contain HDF5 files.
#
# TODO:
#   - are these the correct files?
#   - is this the output of the current GR pipeline?
batch_file_path = '/cluster/work/borgw/ICU_Bern/data/imputed/exploration'

# All `.h5` files in the batch file path.
batch_files = [f for f in glob.glob(os.path.join(batch_file_path, 'batch_*.h5'))]

# Contains endpoints plus other weird stuff for patients. The problem is
# that individual files do not match the patient IDs of the individual
# batches. Hence, the purpose of this script.
endpoint_file_path = '/cluster/work/borgw/ICU_Bern/data/endpoints'

# All `.h5` files in the endpoints file path.
endpoint_files = [f for f in glob.glob(os.path.join(endpoint_file_path, '*.h5'))]

# Maps a patient ID to its batch file, i.e. the file with the imputed
# data.
id_to_batch_file = dict()

for batch_file in batch_files:
    print('Processing {}...'.format(batch_file))

    # Data in each of these batches is stored as part of the 'imputed'
    # group.
    #

    data = pd.read_hdf(batch_file, 'imputed', columns=['PatientID'], mode='r')
    patient_ids = data['PatientID'].unique()

    for patient_id in patient_ids:
        if not patient_id in id_to_batch_file:
            id_to_batch_file[patient_id] = batch_file
        else:
            print('Patient with ID {} occurs in multiple batch files'.format(batch_file))

    # Explicitly removing all references to the data frame here. This is
    # just a workaround until we have figured out what is going wrong in
    # the other case.
    del data
    gc.collect()
