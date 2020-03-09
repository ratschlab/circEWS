#!/usr/bin/env python3
#

import argparse
import glob
import logging
import os
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduced', type=str, help='Directory with imputed data')
    parser.add_argument('--output', type=str, help='Output directory', default='~/tmp')

    args = parser.parse_args()

    # Ensures that '~' can be used to denote the user's home directory
    # when specifying an output path.
    args.output = os.path.expanduser(args.output)

    # Keyword arguments for storing HDF5 files. These should be used
    # whenever an HDF5 file has to be written.
    hdf_kwargs = {
        'complevel': 5,
    }

    # Create output directory if it does not already exist; all errors
    # concerning this will be ignored.
    os.makedirs(args.output, exist_ok=True)

    batch_files = [f for f in sorted(glob.glob(os.path.join(args.reduced, 'reduced_*.h5')))]
    print(batch_files)
    pids = []

    for batch_file in batch_files:
        if 'external' in args.reduced:
            dfx = pd.read_hdf(batch_file, '/merged_clean')
        else:
            dfx = pd.read_hdf(batch_file, '/reduced')
        print(dfx['PatientID'].unique())
        for pid in dfx['PatientID'].unique():
            pids.append(pid)
            pdf = dfx[dfx['PatientID'] == pid]
            print(pid, len(pdf))
            X = pd.HDFStore(os.path.join(args.output, 'reduced_' + str(int(pid)) + '.h5'), mode='w', **hdf_kwargs)
            pd.concat([pdf]).to_hdf(X, '/reduced', format='table', data_columns=['PatientID'], **hdf_kwargs)
            X.close()

    print(pids)
