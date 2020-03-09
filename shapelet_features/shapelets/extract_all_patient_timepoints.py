#!/usr/bin/env python3

import logging
import json
import os
import numpy as np

from multiprocessing import JoinableQueue
# Activate below to see debugging messages
# logging.basicConfig(level=logging.DEBUG)
from utils.cli import existing_file, nonexistant_file
from utils.mp import HDF5Reader, PatientProcessor, Aggregator, ParallelPipeline, MyManager, DateTimeJSONEncoder


def process_patient(patient_id, patient_data):
    """
    Extract all timepoints from a patient used for resampling of the endpoints.
    """
    # print('Processing patient', patient_id, patient_data['AbsDatetime'].tolist())
    return {str(patient_id): patient_data['AbsDatetime'].tolist()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('timeseries_files', nargs='+', type=existing_file, help='Time series files in HDF5 format')
    parser.add_argument('--output', required=True, type=nonexistant_file, help='Output file in json format')
    parser.add_argument('--n-readers', type=int, default=2, help='Number of reader processes to start')
    parser.add_argument('--n-workers', type=int, default=1, help='Number of worker processes to start')

    args = parser.parse_args()

    # Setup queues and the results dictionary
    # Queue for input files in HDF5 format
    inputfile_queue = JoinableQueue()
    # These variables are required to track the results at one place and collect error messages
    manager = MyManager()
    manager.start()
    progressbar = manager.tqdm(total=len(args.timeseries_files))
    results = manager.dict()
    error_list = manager.list()

    logging = logging.getLogger(os.path.basename(__file__))

    # Setup readers, workers and aggregators according to command line parameters
    pipeline = ParallelPipeline(steps=[
        (HDF5Reader, {'hdf5_group': 'imputed', 'progressbar': progressbar, 'columns': ['PatientID', 'AbsDatetime']}, args.n_readers, 300),
        (PatientProcessor, {'processing_function': process_patient, 'error_list': error_list}, args.n_workers),
        # It seems that the Aggregator is taking very long to finish its input queue. Why?
        (Aggregator, {'output_dict': results, 'error_list': error_list}, 1)
    ], input_queue=inputfile_queue)

    # Start all processes and setup intermediate queues
    pipeline.run()

    # Add input file to queue
    for f in args.timeseries_files:
        inputfile_queue.put(f)

    pipeline.wait_for_completion()
    progressbar.close()

    if len(error_list) > 0:
        logging.error('Several errors occurred during processing:')
        for e in error_list:
            logging.error(e)

        logging.error('Writing errors to {}'.format(args.output + '.log'))

        with open(args.output + '.log', 'w') as f:
            for e in error_list:
                print(e, file=f)

    logging.debug('Writing results to output file')
    with open(args.output, 'w') as f:
        json.dump(results.copy(), f, indent=2, cls=DateTimeJSONEncoder)
