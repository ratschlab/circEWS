#!/usr/bin/env python3
#
# Example call: bsub python extract_all_data.py /cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/temporal_split_180918.tsv --imputed-directory /cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_180918/reduced/ --output-directory /cluster/work/borgw/Bern_ICU_Sanctuary/v6b_all_splits/

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

# https://stackoverflow.com/questions/40055835/removing-elements-from-an-array-that-are-in-another-array/40055928


def in1d_dot_approach(A, B):
    cumdims = (np.maximum(A.max(), B.max()) + 1)**np.arange(B.shape[1])
    return A[~np.in1d(A.dot(cumdims), B.dot(cumdims))]


def decompose_path(path):
    '''
    Recursively decomposes a path into individual folders and file names
    that are encountered while traversing it.
    '''

    folders = []

    # Initialize the list of folders with the basename, provided such
    # a thing exists.
    basename = os.path.basename(path)
    if basename:
        folders.append(basename)

    while True:
        path = os.path.dirname(path)
        basename = os.path.basename(path)

        if basename:
            folders.append(basename)
        else:
            break

    folders.reverse()
    return folders


def get_date_from_path(path, prefix=''):
    '''
    Attempts to parse a date portion, specified as YYMMDD, from a path.
    This function looks for one folder within the decomposed path  that
    matches the specification. An optional prefix can be used to search
    for folders of the form `prefixYYMMDD`.

    Raises an exception if no date can be found.
    '''

    folders = decompose_path(path)
    re_date = r'^' + prefix + '(\d{6})$'

    for folder in folders:
        m = re.match(re_date, folder)
        if m:
            return m.group(1)

    raise RuntimeError('Unable to find expected date portion in path {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imputed-directory', type=str, help='Directory with imputed data')
    parser.add_argument('--output-directory', type=str, help='Output directory', default='~/tmp')
    parser.add_argument('SPLIT', type=str, help='Split file to use')

    args = parser.parse_args()

    # Ensures that '~' can be used to denote the user's home directory
    # when specifying an output path.
    args.output_directory = os.path.expanduser(args.output_directory)

    # Create output directory if it does not already exist; all errors
    # concerning this will be ignored.
    os.makedirs(args.output_directory, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logging = logging.getLogger(os.path.basename(__file__))
    logging.info('Reading split file {}'.format(args.SPLIT))

    split_date = get_date_from_path(os.path.splitext(os.path.basename(args.SPLIT))[0], prefix='temporal_split_')
    logging.info('Date of split information is {}'.format(split_date))

    patients = np.array([])
    batch_counter = 0
    patient_per_batch_counter = 0
    patient_per_batch = 100

    df_all_batches = []

    for split_id in ['temporal_1', 'temporal_2', 'temporal_3', 'temporal_4', 'temporal_5']:
        full_split_df = pd.read_csv(args.SPLIT, sep='\t')
        full_split_df = full_split_df[['pid', split_id]]

        logging.info('Grouping patients according to ' + str(split_id))

        # The key of this dictionary will be either 'train', 'test', or
        # 'val'
        split_data = {
            split: data for split, data in full_split_df.groupby(split_id)
        }

        assert 'train' in split_data.keys()
        assert 'test' in split_data.keys()
        assert 'val' in split_data.keys()

        train_patients = split_data['train']['pid'].values.reshape((1, -1))[0]
        test_patients = split_data['test']['pid'].values.reshape((1, -1))[0]
        val_patients = split_data['val']['pid'].values.reshape((1, -1))[0]

        all_patients = np.append(np.append(train_patients, test_patients), val_patients)
        logging.info('Number patients in ' + str(split_id) + ' = ' + str(len(all_patients)))
        all_patients = [i for i in all_patients if i not in patients]
        logging.info('Number patients not processed before = ' + str(len(all_patients)))
        patients = np.append(patients, all_patients)

        # Reduce split data frame and store it in the output directory such
        # that it can be picked up by subsequent scripts.
        # full_split_df = full_split_df.query('pid in @patients')
        # if not os.path.isdir(os.path.join(args.output_directory, split_date)):
        #     os.makedirs(os.path.join(args.output_directory, split_date))

        # full_split_df_out = os.path.join(args.output_directory, split_date, 'split.tsv')
        # logging.info('Writing split file to {}'.format(full_split_df_out))
        # full_split_df.to_csv(full_split_df_out, sep='\t', index=False)

        # Date portion of the feature matrix path; will be used in the
        # subsequent steps to check validity.
        Xy_date = None

        # Keyword arguments for storing HDF5 files. These should be used
        # whenever an HDF5 file has to be written.
        hdf_kwargs = {
            'complevel': 5,
        }

        imp_dir = args.imputed_directory + '/' + str(split_id) + '/'
        imputed_date = get_date_from_path(imp_dir, prefix='imputed_')
        imputed_directory_out = os.path.join(args.output_directory, imputed_date, 'imputed')

        if imputed_date != split_date:
            logging.warning('Split date {} does not match date {} of imputed data; will continue nonetheless'.format(split_date, imputed_date))

        os.makedirs(imputed_directory_out, exist_ok=True)
        logging.info('Using output directory {} for imputed data'.format(imputed_directory_out))

        batch_files = [f for f in sorted(glob.glob(os.path.join(imp_dir, 'batch_*.h5')))]

        ################################################################
        # Store batch information
        ################################################################

        logging.info('Extracting data from batches')

        # Stores the patient IDs of all *processed* patients. The difference
        # between this set and `patients` should hopefully be small, or even
        # zero in the best case.
        processed_patients = set()

        for batch_file in batch_files:
            X_store = pd.HDFStore(batch_file, mode='r')

            logging.info('Processing {}...'.format(os.path.basename(batch_file)))

            X_batch = X_store.get('imputed')
            X_batch_grouped = X_batch.groupby('PatientID')

            for patient_id, data in X_batch_grouped:
                if patient_id in all_patients:
                    logging.debug('Storing patient {} for inclusion in new imputed matrix'.format(patient_id))
                    df_all_batches.append(data)
                    patient_per_batch_counter += 1

                    # Mark patient as being processed; note that this
                    # uses the original patient ID because we use the
                    # column of the respective data frame.
                    processed_patients.add(patient_id)

                    # Finally, store the data frame
                    if patient_per_batch_counter >= patient_per_batch:
                        logging.info('Storing batch {} in {}'.format(batch_counter, imputed_directory_out))
                        X = pd.HDFStore(os.path.join(imputed_directory_out, 'batch_' + str(batch_counter) + '.h5'), mode='w', **hdf_kwargs)
                        pd.concat(df_all_batches).to_hdf(X, 'imputed', format='table', data_columns=['PatientID'], **hdf_kwargs)
                        df_all_batches = []
                        patient_per_batch_counter = 0
                        batch_counter += 1
                        X.close()

            X_store.close()

    X = pd.HDFStore(os.path.join(imputed_directory_out, 'batch_' + str(batch_counter) + '.h5'), mode='w', **hdf_kwargs)
    pd.concat(df_all_batches).to_hdf(X, 'imputed', format='table', data_columns=['PatientID'], **hdf_kwargs)
    X.close()
