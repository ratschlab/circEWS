#!/usr/bin/env python3
#
# extract_reduced_data_set.py: extracts a smaller subset of all patients
# with a pre-defined size for a train, test, and validation data set.
#
# Example call: ./extract_reduced_data_set.py --Xy-directory /cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/7_ml_input/180830/reduced/temporal_5/AllLabels_0.0_8.0 /cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/temporal_split_180827.tsv --imputed-directory /cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_180827/reduced/temporal_5/ --output-directory /cluster/work/borgw/Bern_ICU_Sanctuary/v6/reduced/

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
    parser.add_argument('--n-train', type=int, help='Number of patients in training data set', default=1000)
    parser.add_argument('--n-test', type=int, help='Number of patients in test data set', default=200)
    parser.add_argument('--n-val', type=int, help='Number of patients in validation data set', default=200)
    parser.add_argument('--Xy-directory', type=str, help='Directory with X/y matrices')
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

    full_split_df = pd.read_csv(args.SPLIT, sep='\t')
    full_split_df = full_split_df[['pid', 'temporal_5']]

    logging.info('Grouping patients according to "temporal_5" status')

    # The key of this dictionary will be either 'train', 'test', or
    # 'val'
    split_data = {
        split: data for split, data in full_split_df.groupby('temporal_5')
    }

    assert 'train' in split_data.keys()
    assert 'test' in split_data.keys()
    assert 'val' in split_data.keys()

    train_patients = split_data['train']['pid'].values
    test_patients = split_data['test']['pid'].values
    val_patients = split_data['val']['pid'].values

    np.random.seed(42)

    logging.info('Selecting patients at random (seed = 42)')

    train_patients = list(np.random.choice(train_patients, args.n_train))
    test_patients = list(np.random.choice(test_patients, args.n_test))
    val_patients = list(np.random.choice(val_patients, args.n_val))
    patients = set(train_patients + test_patients + val_patients)

    assert len(train_patients) == args.n_train
    assert len(test_patients) == args.n_test
    assert len(val_patients) == args.n_val

    logging.info('Extracted {} patients (training), {} patients (test), and {} patients (validation)'.format(args.n_train, args.n_test, args.n_val))

    # Reduce split data frame and store it in the output directory such
    # that it can be picked up by subsequent scripts.
    full_split_df = full_split_df.query('pid in @patients')
    if not os.path.isdir(os.path.join(args.output_directory, split_date)):
        os.makedirs(os.path.join(args.output_directory, split_date))

    full_split_df_out = os.path.join(args.output_directory, split_date, 'split.tsv')
    logging.info('Writing split file to {}'.format(full_split_df_out))
    full_split_df.to_csv(full_split_df_out, sep='\t', index=False)

    # Date portion of the feature matrix path; will be used in the
    # subsequent steps to check validity.
    Xy_date = None

    # Keyword arguments for storing HDF5 files. These should be used
    # whenever an HDF5 file has to be written.
    hdf_kwargs = {
        'complevel': 5,
    }

    # Prepare a new set of feature matrices (full feature matrices)
    # based on the selected split.
    if args.Xy_directory:
        # The date portion of the X/y directory is *not* supposed to
        # contain any prefix, so we can easily extract its date.
        Xy_date = get_date_from_path(args.Xy_directory)
        Xy_directory_out = os.path.join(args.output_directory, Xy_date, 'reduced', os.path.basename(args.Xy_directory))

        if Xy_date != split_date:
            logging.warning('Split date {} does not match X/y date {}; will continue nonetheless'.format(split_date, Xy_date))

        logging.info('Using output directory {} and subordinate directories for X/y data'.format(Xy_directory_out))

        os.makedirs(Xy_directory_out, exist_ok=True)
        os.makedirs(os.path.join(Xy_directory_out, 'X'), exist_ok=True)
        os.makedirs(os.path.join(Xy_directory_out, 'y'), exist_ok=True)

        X_files = [f for f in sorted(glob.glob(os.path.join(args.Xy_directory, 'X/batch_*.h5')))]
        y_files = [f for f in sorted(glob.glob(os.path.join(args.Xy_directory, 'y/batch_*.h5')))]

        # Check that each `X` file has a corresponding `y` file and
        # remove files that have no counterpart.
        n_X_files = len(X_files)
        n_y_files = len(y_files)

        X_files = [X_file for X_file in X_files if os.path.basename(X_file) in map(os.path.basename, y_files)]
        y_files = [y_file for y_file in y_files if os.path.basename(y_file) in map(os.path.basename, X_files)]

        if n_X_files != len(X_files):
            logging.info('Removed {} X files because they have no matching y file'.format(n_X_files - len(X_files)))

        if n_y_files != len(y_files):
            logging.info('Removed {} y files because they have no matching X file'.format(n_y_files - len(y_files)))

        assert len(X_files) == len(y_files)

        logging.info('Processing {} X/y files'.format(len(X_files)))

        X = pd.HDFStore(os.path.join(Xy_directory_out, 'X/batch_0_reduced.h5'), mode='w', **hdf_kwargs)
        y = pd.HDFStore(os.path.join(Xy_directory_out, 'y/batch_0_reduced.h5'), mode='w', **hdf_kwargs)

        # Stores the patient IDs of all *processed* patients. The difference
        # between this set and `patients` should hopefully be small, or even
        # zero in the best case.
        processed_patients = set()

        for X_file, y_file in zip(X_files, y_files):
            # The `HDFStore` class does *not* open these files in
            # read-only mode by default, which may cause problems
            # with locking.
            y_store = pd.HDFStore(y_file, mode='r')
            X_store = pd.HDFStore(X_file, mode='r')

            logging.info('Processing {}...'.format(os.path.basename(X_file)))

            # Only take patients that are represented in *both* files
            # because we will be writing spurious data otherwise.
            batch_patients = set(X_store.keys()).intersection(set(y_store.keys()))

            # Convert patient IDs to `str` and prepend a '/' in order to match
            # the format of the keys in the imputed file.
            patients_str_keys = ['/' + str(patient) for patient in patients]

            for patient_id in sorted(batch_patients.intersection(patients_str_keys)):

                # Need to *remove* the leading '/' again in order to be
                # consistent with the key format.
                processed_patients.add(int(patient_id[1:]))

                logging.debug('Storing patient {} in new matrix'.format(patient_id[1:]))

                X_tmp = X_store.get(patient_id)
                y_tmp = y_store.get(patient_id)

                if X is None:
                    X = pd.DataFrame().reindex_like(X_tmp)
                    logging.info('Created columns for X from first entry')
                    logging.info('Columns: {}'.format(X.columns))

                if y is None:
                    y = pd.DataFrame().reindex_like(y_tmp)
                    logging.info('Created columns for y from first entry')
                    logging.info('Columns: {}'.format(y.columns))

                X_tmp.to_hdf(X, patient_id, **hdf_kwargs)
                y_tmp.to_hdf(y, patient_id, **hdf_kwargs)

            X_store.close()
            y_store.close()

        X.close()
        y.close()

        n_patients = len(patients)
        n_processed_patients = len(processed_patients)

        assert n_processed_patients <= n_patients

        logging.info('Processed {}/{} patients for reduced feature matrix creation'.format(n_processed_patients, n_patients))

        if n_patients != n_processed_patients:
            missing_patients = patients.difference(processed_patients)
            logging.warning('The following patients could not be processed because they were not found: {}'.format(missing_patients))

    # Prepare a new set of imputed data files based on the selected
    # split.
    if args.imputed_directory:
        imputed_date = get_date_from_path(args.imputed_directory, prefix='imputed_')
        imputed_directory_out = os.path.join(args.output_directory, imputed_date, 'imputed')

        if Xy_date and imputed_date != Xy_date:
            logging.warning('X/y date {} does not match date {} of imputed data; will continue nonetheless'.format(Xy_date, imputed_date))

        if imputed_date != split_date:
            logging.warning('Split date {} does not match date {} of imputed data; will continue nonetheless'.format(split_date, imputed_date))

        os.makedirs(imputed_directory_out, exist_ok=True)
        logging.info('Using output directory {} for imputed data'.format(imputed_directory_out))

        batch_files = [f for f in sorted(glob.glob(os.path.join(args.imputed_directory, 'batch_*.h5')))]
        static_file = os.path.join(args.imputed_directory, 'static.h5')

        ################################################################
        # Store batch information
        ################################################################

        logging.info('Extracting data from batches')

        X = pd.HDFStore(os.path.join(imputed_directory_out, 'batch_0_reduced.h5'), mode='w', **hdf_kwargs)

        # Stores data over all batches; this is required because
        # individual patients are collated within a single group
        # so that we cannot write them per batch
        df_all_batches = []

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
                if patient_id in patients:
                    logging.debug('Storing patient {} for inclusion in new imputed matrix'.format(patient_id))
                    df_all_batches.append(data)

                    # Mark patient as being processed; note that this
                    # uses the original patient ID because we use the
                    # column of the respective data frame.
                    processed_patients.add(patient_id)

            # Finally, store the data frame in the specified group;
            # I have not found a way to append instead.
            if df_all_batches:
                pd.concat(df_all_batches).to_hdf(X, 'imputed', format='table', data_columns=['PatientID'], **hdf_kwargs)

            X_store.close()

        X.close()

        n_patients = len(patients)
        n_processed_patients = len(processed_patients)

        assert n_processed_patients <= n_patients

        logging.info('Processed {}/{} patients for reduced imputed matrix creation'.format(n_processed_patients, n_patients))

        if n_patients != n_processed_patients:
            missing_patients = patients.difference(processed_patients)
            logging.warning('The following patients could not be processed because they were not found: {}'.format(missing_patients))

        ################################################################
        # Store static information
        ################################################################

        logging.info('Extracting data from static file')

        X_static = pd.HDFStore(os.path.join(imputed_directory_out, 'static.h5'), mode='w', **hdf_kwargs)
        static_data = pd.read_hdf(static_file, 'data', mode='r')

        # Again, we need to collect all patients prior to storing them
        # in the file.
        df_all_batches = []

        processed_patients_static = set()

        for patient_id, data in static_data.groupby('PatientID'):
            if patient_id in patients:
                logging.debug('Storing static data for patient {}'.format(patient_id))
                df_all_batches.append(data)
                processed_patients_static.add(patient_id)

        if df_all_batches:
            pd.concat(df_all_batches).to_hdf(X_static, 'data', format='table', data_columns=['PatientID'], **hdf_kwargs)

        logging.info('Processed {}/{} patients for static data'.format(len(processed_patients_static), n_patients))

        if len(processed_patients_static) != len(processed_patients):
            logging.warning('Static patient information does not match imputed patient information!')
            logging.warning('The following patients are missing: {}'.format(processed_patients.symmetric_difference(processed_patients_static)))

        X_static.close()
