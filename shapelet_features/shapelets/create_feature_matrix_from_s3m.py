
#!/usr/bin/env python3
import os
import json
import random
import logging

import pandas as pd
import numpy as np

from collections import defaultdict

from multiprocessing import JoinableQueue
from utils.cli import md5sum, nonexistant_file, existing_file
from utils.mp import HDF5Reader, HDF5Aggregator, PatientProcessor, ParallelPipeline, MyManager


differences = False
logging = logging.getLogger(os.path.basename(__file__))


def get_rolling_window_distances(arr, shapelet):
    if len(arr) == len(shapelet):
        return np.linalg.norm(arr - shapelet) ** 2
    else:
        normalization_constant = max(len(arr), len(shapelet)) / min(len(arr), len(shapelet))

        # w.l.o.g., let $x$ be the smaller series
        x = pd.Series(arr) if len(arr) < len(shapelet) else shapelet
        y = pd.Series(shapelet) if len(arr) < len(shapelet) else arr

        unnormalized_distance = y.rolling(len(x)).apply(lambda z: np.linalg.norm(x - z) ** 2, raw=True).min()
        return unnormalized_distance * normalization_constant


def process_patient(patient_id, patient_data, shapelets, counting_range, transforms):
    patient_data.set_index('AbsDatetime', inplace=True, drop=False)
    result_df = patient_data[['PatientID', 'RelDatetime']].copy()

    k = 0  # For dist-hist
    if differences:
        print(differences)

    # Calculate the squared euclidean distance of each fragment of a rolling window of length of shapelet
    stepsize = '30m'
    data_stepsize = '5m'
    tds = pd.to_timedelta(stepsize)
    tdc = pd.to_timedelta(counting_range)
    scale = int(tds / pd.to_timedelta(data_stepsize))
    for var in shapelets.keys():
        if var in patient_data.keys():
            for shapelet_id, shapelet in enumerate(shapelets[var]):
                dist_col_name = '{}_{}_dist'.format(var, shapelet_id)
                shapelet_length = len(shapelet['shapelet'])
                # differences=True #Delete for normal use
                if differences:
                    patient_data[dist_col_name] = patient_data[var].diff()
                    patient_data[dist_col_name][0] = 0
                    patient_data[dist_col_name] = patient_data[dist_col_name].rolling(shapelet_length)\
                        .apply(lambda x: get_rolling_window_distances(x, shapelet['shapelet']), raw=True)
                else:
                    patient_data[dist_col_name] = patient_data[var].rolling(shapelet_length)\
                        .apply(lambda x: get_rolling_window_distances(x, shapelet['shapelet']), raw=True)
                for transform in transforms:
                    result_col_name = '{}_{}_{}'.format(var, shapelet_id, transform)
                    # Count occurences of shapelet in last
                    # @rolling_window_size.
                    if transform == 'counts':
                        rolling_view = (patient_data[dist_col_name] <= shapelet['threshold']).rolling(counting_range)
                        result_df[result_col_name] = rolling_view.sum().astype('uint8')
                    elif transform == 'normalized_counts':
                        rolling_view = (patient_data[dist_col_name] <= shapelet['threshold']).rolling(counting_range)
                        result_df[result_col_name] = rolling_view.mean().astype('float16')
                    elif transform == 'min':
                        result_df[result_col_name] = patient_data[dist_col_name].rolling(counting_range).min().astype('float16')
                    elif transform == 'distance':
                        result_df[result_col_name] = patient_data[dist_col_name].astype('float16')
                    elif transform == 'dist-set':
                        result_df[result_col_name] = patient_data[dist_col_name].astype('float16')
                        k = 1
                        while k * tds <= tdc:
                            result_df[result_col_name + str(k)] = np.nan
                            if len(patient_data.index.values) > k * scale:
                                result_df.iloc[scale * k:, result_df.columns.get_loc(result_col_name + str(k))] = patient_data.iloc[:-(scale * k), patient_data.columns.get_loc(dist_col_name)].values

                            k = k + 1
                    else:
                        raise NotImplementedError('transform {}'.format(transform))

    logging.info('Number of iterations over while {}'.format(k))

    # As we moved to hdf5, the index is actually perserved
    # We want to explicitly save `AbsDatetime` as a column
    # Calling reset_index does this for us as it converts the index to a column
    result_df.reset_index(inplace=True)

    # Sort the columns such that the following are the first in the DataFrame
    first_columns = ['PatientID', 'AbsDatetime', 'RelDatetime']
    result_df = result_df[first_columns + [col for col in result_df.columns if col not in first_columns]]
    return patient_id, result_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('timeseries_files', nargs='+', type=existing_file, help='Time series files in HDF5 format')
    parser.add_argument('--s3m-shapelets', nargs='+', type=existing_file, help='Paths to json files containing s3m detected shapelets')
    parser.add_argument('--n-shapelets', type=int, default=None, help='Extract features for n shapelets for each provided json file. '
                                                                      'Default behaviour is to take the first n shapelets in the json file, which corresponds to the most significant shapelets.')
    parser.add_argument('--random', default=False, action='store_true', help='Instead of taking the first n shapelets, take n randomly selected shapelets from the s3m output json.')
    parser.add_argument('--counting-range', type=str, default='4h', help='Time over which we should accumulate shapelet counts for each time point (default: "4h")')
    parser.add_argument('--transforms', nargs='+', choices=['counts', 'normalized_counts', 'min', 'distance', 'dist-set'], default=['distance'], help='Tranforms to use for computation of shapelet features')
    parser.add_argument('--output', required=True, type=nonexistant_file, help='Output file in hdf5 format')
    parser.add_argument('--n-readers', type=int, default=2, help='Number of reader processes to start')
    parser.add_argument('--n-workers', type=int, default=15, help='Number of worker processes to start')

    args = parser.parse_args()

    # Build dict containing shaplet information
    shapelets_dict = defaultdict(list)
    for shapelet_file in args.s3m_shapelets:
        if 'IMPUTED_STATUS_CUM_COUNT' in shapelet_file:
            print(shapelet_file)
            differences = True
        basename = os.path.basename(shapelet_file)
        varname = basename.split('_')[0]
        with open(shapelet_file, 'r') as f:
            try:
                json_dict = json.load(f)
            except Exception as e:
                logging.error('Error processing shapelet information file {}: {}'.format(shapelet_file, e))
                logging.debug('Skipping file')
                continue

            # Maybe we should warn in case there are multiple files referring to the same variable
            if args.n_shapelets:
                if args.random:
                    # if random flag is passed, we dont want to take the most significant shapelets, but random significant ones
                    random.shuffle(json_dict['shapelets'])

                if args.n_shapelets > len(json_dict['shapelets']):
                    # Print a warning message, as python does not raise an error if the end index of a slice is out of range
                    logging.warn('%s contains less than %d shapelets! Only %d shaplets will be used for extraction!', shapelet_file, args.n_shapelets, len(json_dict['shapelets']))

                shapelets_dict[varname].extend(json_dict['shapelets'][:args.n_shapelets])
            else:
                shapelets_dict[varname].extend(json_dict['shapelets'])

    if len(shapelets_dict) == 0:
        logging.error('None of the provided JSON file contained significant shapelets. Nothing to do here; will exit.')
        import sys
        sys.exit(1)

    # Setup queues and the results dictionary
    # Queue for input files in HDF5 format
    inputfile_queue = JoinableQueue()
    # These variables are required to track the results at one place and collect error messages
    manager = MyManager()
    manager.start()
    progressbar = manager.tqdm(total=len(args.timeseries_files))
    results = manager.dict()
    error_list = manager.list()

    # Setup readers, workers and aggregators according to command line parameters
    pipeline = ParallelPipeline(steps=[
        (HDF5Reader, {'hdf5_group': 'imputed', 'progressbar': progressbar}, args.n_readers, 300),
        (PatientProcessor, {'processing_function': process_patient, 'error_list': error_list,
                            'function_args': {
                                'shapelets': shapelets_dict,
                                'counting_range': args.counting_range,
                                'transforms': args.transforms}
                            }, args.n_workers),
        # It seems that the Aggregator is taking very long to finish its input queue. Why?
        (HDF5Aggregator, {'output': args.output, 'error_list': error_list}, 1)
    ], input_queue=inputfile_queue)

    # Start all processes and setup intermediate queues
    pipeline.run()

    # Add input file to queue
    for f in args.timeseries_files:
        inputfile_queue.put(f)

    pipeline.wait_for_completion()
    progressbar.close()

    # TODO: Write the errors out into a json file to allow later analysis
    if len(error_list) > 0:
        logging.error('The following errors occurred during processing:')
        for e in error_list:
            logging.error(e)

    logging.debug('Writing results to output file {}'.format(args.output + '.json'))

    run_summary = {
        'time_series_files': {fname: md5sum(fname) for fname in args.timeseries_files},
        'shapelet_files': {fname: md5sum(fname) for fname in args.s3m_shapelets},
        'counting_range': args.counting_range,
        'n_shapelets': args.n_shapelets,
        'random': args.random,
        'shapelets': shapelets_dict
    }
    with open(args.output + '.json', 'w') as f:
        json.dump(run_summary, f)
