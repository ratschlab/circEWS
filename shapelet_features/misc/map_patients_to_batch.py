#!/usr/bin/env python3

import pandas as pd

import gc
import glob
import json
import os

batch_file_path = '/cluster/work/borgw/ICU_Bern/data/imputed/exploration'
batch_files = [f for f in glob.glob(os.path.join(batch_file_path, 'batch_*.h5'))]

# Maps a batch filename to a list of IDs of patients occurring in that
# file.
name_to_ids = dict()

for batch_file in batch_files:
    print('Processing {}...'.format(batch_file))

    # Use filename portion of the batch file (without any extension) to
    # represent the input file
    name = os.path.splitext(os.path.basename(batch_file))[0]

    data = pd.read_hdf(batch_file, 'imputed', mode='r')
    patient_ids = data['PatientID'].unique().tolist()

    name_to_ids[name] = list(patient_ids)

    # Explicitly removing all references to the data frame here. This is
    # just a workaround until we have figured out what is going wrong in
    # the other case.
    del data
    gc.collect()

with open('Output/batch_patient_ids.json', 'w') as f:
    json.dump(name_to_ids, f)
