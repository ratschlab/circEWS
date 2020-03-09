#!/usr/bin/env python3
#
# Environment:
#
# module load hdf5
# module load python_gpu/3.6.4

import json

from multiprocessing import JoinableQueue

from utils.cli import existing_file, nonexistant_file
from utils.mp import ProcessingError, HDF5Reader, PatientProcessor, Aggregator, ParallelPipeline, MyManager


def process_patient(patient_id, patient_data, timepoints):
    import numpy as np
    import pandas as pd
    patient_data.set_index('Datetime', inplace=True, drop=False)
    # If timepoints are provided, reindex such that the endpoints match the imputed timeseries
    if timepoints:
        try:
            cur_timepoints = pd.to_datetime(timepoints[str(patient_id)])
            patient_data = patient_data.reindex(index=cur_timepoints, method='nearest')
            patient_data['Datetime'] = patient_data.index
        except KeyError:
            return ProcessingError(patient_id, ProcessingError.Type.CannotAlignDatetimes)
    # s3m endpoint is one if one of the events is set
    patient_data['s3m_endpoint'] = (patient_data[['event1', 'event2', 'event3']].sum(axis=1) > 0).astype(int)

    # Set those to NaN where all events are NaN
    nans = patient_data[['event1', 'event2', 'event3']].isnull().all(axis=1)
    patient_data.loc[nans, 's3m_endpoint'] = np.nan

    # remove rows where endpoint nan:
    pat_clean = patient_data[np.isfinite(patient_data['s3m_endpoint'])]
    # in case there is no row left continue with next patient
    if len(pat_clean) == 0:
        # print('Jumping Patient {} as no non-Nan endpoint available'.format(patient_id))
        return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint)

    # give consecutive fragments of identical s3m_endpoint label same id (e.g. 1,1,1,2,2,2,1,1,3,3..)
    consecutive_fragments = (pat_clean.s3m_endpoint != pat_clean.s3m_endpoint.shift()).cumsum()

    fragments = pat_clean.groupby(consecutive_fragments, as_index=False).apply(lambda df: pd.Series({
        's3m_endpoint': df.s3m_endpoint.iloc[0],
        'Start': df.Datetime.iloc[0],
        'End': df.Datetime.iloc[-1]
    }))

    return {str(patient_id): fragments.to_dict(orient='list')}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint_files', nargs='+', type=existing_file, help='Endpoint files in HDF5 format')
    parser.add_argument('--output', required=True, type=nonexistant_file, help='Output file in json format')
    parser.add_argument('--n-readers', type=int, default=3, help='Number of reader processes to start')
    parser.add_argument('--n-workers', type=int, default=20, help='Number of worker processes to start')
    parser.add_argument('--align-timepoints', type=existing_file, default=None, help='Json file containing the patientID to timepoint mapping')

    args = parser.parse_args()

    # Setup queues and the results dictionary
    # Queue for input files in HDF5 format
    inputfile_queue = JoinableQueue()
    # These variables are required to track the results at one place and collect error messages
    manager = MyManager()
    manager.start()
    results = manager.dict()
    error_list = manager.list()

    if args.align_timepoints:
        with open(args.align_timepoints, 'r') as f:
            timepoints = json.load(f)
    else:
        timepoints = None

    # Setup readers, workers and aggregators according to command line parameters
    progress_monitor = manager.tqdm(total=len(args.endpoint_files))
    pipeline = ParallelPipeline(steps=[
        (HDF5Reader, {'hdf5_group': 'endpoints', 'progressbar': progress_monitor}, args.n_readers, 300),
        (PatientProcessor, {'processing_function': process_patient, 'error_list': error_list, 'function_args': {'timepoints': timepoints}}, args.n_workers),
        (Aggregator, {'output_dict': results, 'error_list': error_list}, 1)
    ], input_queue=inputfile_queue)
    # Start all processes and setup intermediate queues
    pipeline.run()

    # Add input file to queue
    for f in args.endpoint_files:
        inputfile_queue.put(f)

    pipeline.wait_for_completion()
    progress_monitor.close()


    if len(error_list) > 0:
        print('Errors occurred during processing:')
        for e in error_list:
            print(e)

    print('Writing results to output file')
    with open(args.output, 'w') as f:
        json.dump(results.copy(), f, indent=2, default=str)

