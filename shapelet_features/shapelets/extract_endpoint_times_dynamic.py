#!/usr/bin/env python3
#
# The purpose of this script is to extract windows of certain duration
# and with a certain delta to a potential even switch. This results in
# a set of fragments per patient that can be used to evaluate a model,
# or extract shapelets.

import json

from multiprocessing import JoinableQueue

import numpy as np
import pandas as pd

from utils.cli import existing_file, nonexistant_file
from utils.mp import ProcessingError, HDF5Reader, PatientProcessor, Aggregator, ParallelPipeline, MyManager, DateTimeJSONEncoder


def process_patient(patient_id, patient_data, duration, dt, timepoints):
    """
    Apply dynamic endpoint extraction process to a single patient
    patient_id: the patient id
    patient_data: a pandas dataframe of the endpoints with similar structure to the HDF5 endpoint file
    case_window_size: number of datapoints to extract before onset (label change from 0 to 1)
    distance_onset_extraction_window: distance between registered onset and the extraction window
    minimum_onset_duration: Minimum number of non-zero endpoints required for a switch from 0 to !=0 to be treated as an onset
    """
    patient_data.set_index('Datetime', inplace=True, drop=False)
    # If timepoints are provided, reindex such that the endpoints match the imputed timeseries
    if timepoints:
        try:
            cur_timepoints = pd.to_datetime(timepoints[str(patient_id)])
            patient_data = patient_data.reindex(index=cur_timepoints, method='nearest')
            # for old_index, new_index in zip(patient_data.index, patient_new.index):
            #    print(old_index, '-->', new_index)
            patient_data['Datetime'] = patient_data.index
        except KeyError:
            return ProcessingError(patient_id, ProcessingError.Type.CannotAlignDatetimes)

    # s3m endpoint is one if one of the events is set
    patient_data['s3m_endpoint'] = (patient_data[['event1', 'event2', 'event3']].sum(axis=1) > 0).astype(int)
    # Set those to -1 where all events are NaN; this makes it possible
    # to detect consecutive segments consisting of NaNs
    nans = patient_data[['event1', 'event2', 'event3']].isnull().all(axis=1)
    patient_data.loc[nans, 's3m_endpoint'] = -1

    # Get all available endpoints. This is required in order to catch
    # segments that consist exclusively of NaNs.
    available_endpoints = patient_data['s3m_endpoint'].unique()

    # Check whether only invalid endpoints are available; if so, raise
    # an error because we will never be able to extract valid fragments
    if available_endpoints.max() < 0:
        return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='undefined', cause='all_endpoints_are_nan')

    # If the maximum value of the S3M endpoint is zero, the set of
    # fragments is guaranteed to represent a control. It is still
    # possible, though, that the time series contains some NaNs.
    #
    # NB: this assumes that NaNs have been assigned a negative
    # number
    is_control = available_endpoints.max() == 0

    # give consecutive fragments of identical s3m_endpoint label same id (e.g. 1,1,1,2,2,2,1,1,3,3..)
    consecutive_fragments = (patient_data.s3m_endpoint != patient_data.s3m_endpoint.shift()).cumsum()

    # generate a list where each element contains a fragment of consecutive endpoints
    # Groupby returns tuples of (groupvalue, data). We only need the data here
    # Thus the generator expression only extracts the data (f[1]) and discards the groupvalue
    fragments_without_groupvalue = (f[1] for f in patient_data.groupby(consecutive_fragments, as_index=False))
    fragments = list(fragments_without_groupvalue)

    if is_control:
        # Remove all fragments that contain a NaN endpoint; or rather:
        # only keep those that have an endpoint of zero if we are
        # looking for controls.
        fragments = filter(lambda a: a['s3m_endpoint'].iloc[0] == 0, fragments)

        # Take the *longest* fragment of the control
        fragment = sorted(fragments, key=len)[-1]

        # Check to be sure that this patient is not in !=0 state for complete stay
        # and that his stay is long enough to be considered
        # print(fragment.Datetime.iloc[0], fragment.Datetime.iloc[-1])
        # TODO: Split this in two clauses for correct error handling
        if fragment['s3m_endpoint'].iloc[0] == 0 and (fragment.Datetime.iloc[-1] - duration) >= fragment.Datetime.iloc[0]:
            # Take a random sample from the trajectory and return
            possible_starts = np.where(
                fragment.Datetime + duration <= fragment.Datetime.iloc[-1]
            )[0]
            start_index = np.random.choice(possible_starts)
            start_time = fragment.Datetime.iloc[start_index]
            # TODO: Test if this more elegant implementation would work
            # selected_fragment = fragment[start_time:(start_time + duration)]
            selected_fragment = fragment[np.logical_and(fragment.Datetime >= start_time, fragment.Datetime <= (start_time + duration))]
            return {
                str(patient_id): {
                    's3m_endpoint': [0],  # Endpoint 0 means we don't have a switch from 0 to !=0
                    'Start': [selected_fragment.Datetime.iloc[0]],
                    'End': [selected_fragment.Datetime.iloc[-1]]
                }
            }
        # Handle error in case of insufficient data or NoValidEndpoint
        else:
            fragment_duration = (fragment.Datetime.iloc[-1] - fragment.Datetime.iloc[0])
            return ProcessingError(patient_id, ProcessingError.Type.InsufficientData, duration=fragment_duration, patienttype='control')

    # Handle cases with invalid endpoints caused by a missing switch
    if len(fragments) < 2:
        return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='case', cause='no_switch')

    # Only extract one window for each patient and not every possible window before a state switch
    for possible_pre_onset, possible_post_onset in zip(fragments, fragments[1:]):

        # Skip a pair of fragments if they constitute a switch from NaN
        # to something else (or vice versa)
        if possible_pre_onset['s3m_endpoint'].iloc[0] < 0 or \
           possible_post_onset['s3m_endpoint'].iloc[0] < 0:
            continue

        if possible_pre_onset['s3m_endpoint'].iloc[0] == 0 and \
                possible_post_onset['s3m_endpoint'].iloc[0] == 1:
            # TODO: Move this check into separate function to enhance readability
            if (possible_pre_onset.Datetime.iloc[-1] - duration - dt) >= (possible_pre_onset.Datetime.iloc[0]):  # Check also if we have sufficient data
                # Find first time for which the distance to onset is sufficiently large
                # TODO: Maybe make similar to previous condition (using np.where) to enhance readability
                start_index = (possible_pre_onset.Datetime + duration + dt >= possible_pre_onset.Datetime.iloc[-1]).values.argmax()
                start_time = possible_pre_onset.Datetime.iloc[start_index]
                # TODO: Check if datetime based indexing works here
                fragment = possible_pre_onset[np.logical_and(possible_pre_onset.Datetime >= start_time, possible_pre_onset.Datetime <= (start_time + duration))]
                return {
                    str(patient_id): {
                        's3m_endpoint': [1],  # Endpoint 1 means we found a switch from 0 endpoint to !=0 endpoint
                        'Start': [fragment.Datetime.iloc[0]],
                        'End': [fragment.Datetime.iloc[-1]]
                    }
                }
            else:
                fragment_duration = (possible_pre_onset.Datetime.iloc[-1] - possible_pre_onset.Datetime.iloc[0])
                return ProcessingError(patient_id, ProcessingError.Type.InsufficientData, duration=fragment_duration, patienttype='case')
        else:
            return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='case', cause='invalid_direction')

    # Processed all fragments and did not encounter a valid switch;
    # giving up...
    return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='case', cause='no_valid_switch')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint_files', nargs='+', type=existing_file, help='Endpoint files in HDF5 format')
    parser.add_argument('--duration', type=float, required=True, help='Duration to extract in hours')
    parser.add_argument('--dt', type=float, default=.0, help='Time in hours between detected label switch and the extracted window')
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
        (PatientProcessor, {'processing_function': process_patient, 'error_list': error_list, 'function_args': {'duration': pd.DateOffset(hours=args.duration), 'dt': pd.DateOffset(hours=args.dt), 'timepoints': timepoints}}, args.n_workers),
        (Aggregator, {'output_dict': results, 'error_list': error_list}, 1)
    ], input_queue=inputfile_queue)

    # Add input file to queue
    for f in args.endpoint_files:
        inputfile_queue.put(f)

    # Start all processes and setup intermediate queues
    pipeline.run()

    pipeline.wait_for_completion()
    progress_monitor.close()

    # TODO: Write the errors out into a json file to allow later analysis
    if len(error_list) > 0:
        print('Errors occurred during processing:')
        for e in error_list:
            print(e)

    print('Writing results to output file')
    with open(args.output, 'w') as f:
        json.dump(results.copy(), f, indent=2, cls=DateTimeJSONEncoder)


# #!/usr/bin/env python3
# #
# # The purpose of this script is to extract windows of certain duration
# # and with a certain delta to a potential even switch. This results in
# # a set of fragments per patient that can be used to evaluate a model,
# # or extract shapelets.

# import json

# from multiprocessing import JoinableQueue

# import numpy as np
# import pandas as pd

# from utils.cli import existing_file, nonexistant_file
# from utils.mp import ProcessingError, HDF5Reader, PatientProcessor, Aggregator, ParallelPipeline, MyManager, DateTimeJSONEncoder


# def process_patient(patient_id, patient_data, duration, dt, timepoints):
#     """
#     Apply dynamic endpoint extraction process to a single patient
#     patient_id: the patient id
#     patient_data: a pandas dataframe of the endpoints with similar structure to the HDF5 endpoint file
#     case_window_size: number of datapoints to extract before onset (label change from 0 to 1)
#     distance_onset_extraction_window: distance between registered onset and the extraction window
#     minimum_onset_duration: Minimum number of non-zero endpoints required for a switch from 0 to !=0 to be treated as an onset
#     """
#     patient_data['Datetime'] = pd.to_datetime(patient_data['Datetime'])
#     patient_data.set_index('Datetime', inplace=True, drop=False)
#     # If timepoints are provided, reindex such that the endpoints match the imputed timeseries
#     if str(patient_id) in list(timepoints.keys()):
#         # if timepoints:
#         try:
#             cur_timepoints = pd.to_datetime(timepoints[str(patient_id)])
#             patient_data = patient_data.reindex(index=cur_timepoints, method='nearest')
#             # for old_index, new_index in zip(patient_data.index, patient_new.index):
#             #    print(old_index, '-->', new_index)
#             # patient_data['Datetime'] = patient_data.index
#             # patient_data['AbsDatetime'] = patient_data.index
#         except KeyError:
#             return ProcessingError(patient_id, ProcessingError.Type.CannotAlignDatetimes)

#         # s3m endpoint is one if one of the events is set
#         patient_data['s3m_endpoint'] = (patient_data[['event1', 'event2', 'event3']].sum(axis=1) > 0).astype(int)
#         # Set those to -1 where all events are NaN; this makes it possible
#         # to detect consecutive segments consisting of NaNs
#         nans = patient_data[['event1', 'event2', 'event3']].isnull().all(axis=1)
#         patient_data.loc[nans, 's3m_endpoint'] = -1

#         # Get all available endpoints. This is required in order to catch
#         # segments that consist exclusively of NaNs.
#         available_endpoints = patient_data['s3m_endpoint'].unique()

#         # Check whether only invalid endpoints are available; if so, raise
#         # an error because we will never be able to extract valid fragments
#         # if available_endpoints.max() < 0:
#         #     return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='undefined', cause='all_endpoints_are_nan')

#         # If the maximum value of the S3M endpoint is zero, the set of
#         # fragments is guaranteed to represent a control. It is still
#         # possible, though, that the time series contains some NaNs.
#         #
#         # NB: this assumes that NaNs have been assigned a negative
#         # number
#         is_control = len(available_endpoints) <= 1

#         # give consecutive fragments of identical s3m_endpoint label same id (e.g. 1,1,1,2,2,2,1,1,3,3..)
#         consecutive_fragments = (patient_data.s3m_endpoint != patient_data.s3m_endpoint.shift()).cumsum()

#         # generate a list where each element contains a fragment of consecutive endpoints
#         # Groupby returns tuples of (groupvalue, data). We only need the data here
#         # Thus the generator expression only extracts the data (f[1]) and discards the groupvalue
#         fragments_without_groupvalue = (f[1] for f in patient_data.groupby(consecutive_fragments, as_index=False))
#         fragments = list(fragments_without_groupvalue)

#         if is_control:
#             # print('control')
#             # print(patient_data['s3m_endpoint'].values.tolist())
#             # print(available_endpoints, is_control)
#             # Remove all fragments that contain a NaN endpoint; or rather:
#             # only keep those that have an endpoint of zero if we are
#             # looking for controls.
#             fragments = filter(lambda a: a['s3m_endpoint'].iloc[0] <= 0, fragments)

#             # Take the *longest* fragment of the control
#             fragment = sorted(fragments, key=len)[-1]

#             # Check to be sure that this patient is not in !=0 state for complete stay
#             # and that his stay is long enough to be considered
#             # print(fragment.Datetime.iloc[0], fragment.Datetime.iloc[-1])
#             # TODO: Split this in two clauses for correct error handling
#             # if fragment['s3m_endpoint'].iloc[0] == 0 and (fragment.Datetime.iloc[-1] - duration) >= fragment.Datetime.iloc[0]:
#             if fragment['s3m_endpoint'].iloc[0] < 1 and (fragment.Datetime.iloc[-1] - duration) >= fragment.Datetime.iloc[0]:
#                 # Take a random sample from the trajectory and return
#                 possible_starts = np.where(
#                     fragment.Datetime + duration <= fragment.Datetime.iloc[-1]
#                 )[0]
#                 start_index = np.random.choice(possible_starts)
#                 start_time = fragment.Datetime.iloc[start_index]
#                 # TODO: Test if this more elegant implementation would work
#                 # selected_fragment = fragment[start_time:(start_time + duration)]
#                 selected_fragment = fragment[np.logical_and(fragment.Datetime >= start_time, fragment.Datetime <= (start_time + duration))]
#                 return {
#                     str(patient_id): {
#                         's3m_endpoint': [0],  # Endpoint 0 means we don't have a switch from 0 to !=0
#                         'Start': [selected_fragment.Datetime.iloc[0]],
#                         'End': [selected_fragment.Datetime.iloc[-1]]
#                     }
#                 }
#             # Handle error in case of insufficient data or NoValidEndpoint
#             else:
#                 fragment_duration = (fragment.Datetime.iloc[-1] - fragment.Datetime.iloc[0])
#                 return ProcessingError(patient_id, ProcessingError.Type.InsufficientData, duration=fragment_duration, patienttype='control')

#         # Handle cases with invalid endpoints caused by a missing switch
#         if len(fragments) < 2:
#             return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='case', cause='no_switch')

#         # Only extract one window for each patient and not every possible window before a state switch
#         for possible_pre_onset, possible_post_onset in zip(fragments, fragments[1:]):
#             # print(possible_pre_onset['s3m_endpoint'].iloc[0], possible_post_onset['s3m_endpoint'].iloc[0])
#             # print(not( possible_pre_onset['s3m_endpoint'].iloc[0] < 0 or possible_post_onset['s3m_endpoint'].iloc[0] < 0))
#             # print(possible_pre_onset['s3m_endpoint'].iloc[0] == 0 and possible_post_onset['s3m_endpoint'].iloc[0] == 1)
#             # print(possible_pre_onset.Datetime.iloc[-1] - duration - dt >= possible_pre_onset.Datetime.iloc[0])

#             # IMPORTANT: The commented out part below was used for circEWS. For KDIGO, -1 in s3m_endpoint does not mean nan
#             # # Skip a pair of fragments if they constitute a switch from NaN
#             # # to something else (or vice versa)
#             # if possible_pre_onset['s3m_endpoint'].iloc[0] < 0 or \
#             #    possible_post_onset['s3m_endpoint'].iloc[0] < 0:
#             #     continue

#             # if possible_pre_onset['s3m_endpoint'].iloc[0] == 0 and \
#             #         possible_post_onset['s3m_endpoint'].iloc[0] == 1:

#             # Take any switch twoards a deterioration
#             if possible_pre_onset['s3m_endpoint'].iloc[0] < possible_post_onset['s3m_endpoint'].iloc[0]:

#                 # TODO: Move this check into separate function to enhance readability
#                 if possible_pre_onset.Datetime.iloc[-1] - duration - dt >= possible_pre_onset.Datetime.iloc[0]:  # Check also if we have sufficient data
#                     # Find first time for which the distance to onset is sufficiently large
#                     # TODO: Maybe make similar to previous condition (using np.where) to enhance readability
#                     start_index = (possible_pre_onset.Datetime + duration + dt >= possible_pre_onset.Datetime.iloc[-1]).values.argmax()
#                     start_time = possible_pre_onset.Datetime.iloc[start_index]
#                     # TODO: Check if datetime based indexing works here
#                     fragment = possible_pre_onset[np.logical_and(possible_pre_onset.Datetime >= start_time, possible_pre_onset.Datetime <= (start_time + duration))]
#                     # print('return ', fragment.Datetime.iloc[0], fragment.Datetime.iloc[-1])
#                     return {
#                         str(patient_id): {
#                             's3m_endpoint': [1],  # Endpoint 1 means we found a switch from 0 endpoint to !=0 endpoint
#                             'Start': [fragment.Datetime.iloc[0]],
#                             'End': [fragment.Datetime.iloc[-1]]
#                         }
#                     }
#                 else:
#                     fragment_duration = (possible_pre_onset.Datetime.iloc[-1] - possible_pre_onset.Datetime.iloc[0])
#                     return ProcessingError(patient_id, ProcessingError.Type.InsufficientData, duration=fragment_duration, patienttype='case')
#             else:
#                 return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='case', cause='invalid_direction')

#         # Processed all fragments and did not encounter a valid switch;
#         # giving up...
#         return ProcessingError(patient_id, ProcessingError.Type.NoValidEndpoint, patienttype='case', cause='no_valid_switch')
#     else:
#         return ProcessingError(patient_id, ProcessingError.Type.CannotAlignDatetimes)


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('endpoint_files', nargs='+', type=existing_file, help='Endpoint files in HDF5 format')
#     parser.add_argument('--duration', type=float, required=True, help='Duration to extract in hours')
#     parser.add_argument('--dt', type=float, default=.0, help='Time in hours between detected label switch and the extracted window')
#     parser.add_argument('--output', required=True, type=nonexistant_file, help='Output file in json format')
#     parser.add_argument('--n-readers', type=int, default=3, help='Number of reader processes to start')
#     parser.add_argument('--n-workers', type=int, default=20, help='Number of worker processes to start')
#     parser.add_argument('--align-timepoints', type=existing_file, default=None, help='Json file containing the patientID to timepoint mapping')

#     args = parser.parse_args()

#     # Setup queues and the results dictionary
#     # Queue for input files in HDF5 format
#     inputfile_queue = JoinableQueue()
#     # These variables are required to track the results at one place and collect error messages
#     manager = MyManager()
#     manager.start()
#     results = manager.dict()
#     error_list = manager.list()

#     if args.align_timepoints:
#         with open(args.align_timepoints, 'r') as f:
#             timepoints = json.load(f)
#     else:
#         timepoints = None

#     # Setup readers, workers and aggregators according to command line parameters
#     progress_monitor = manager.tqdm(total=len(args.endpoint_files))
#     pipeline = ParallelPipeline(steps=[
#         (HDF5Reader, {'hdf5_group': 'endpoints', 'progressbar': progress_monitor}, args.n_readers, 300),
#         (PatientProcessor, {'processing_function': process_patient, 'error_list': error_list, 'function_args': {'duration': pd.DateOffset(hours=args.duration), 'dt': pd.DateOffset(hours=args.dt), 'timepoints': timepoints}}, args.n_workers),
#         (Aggregator, {'output_dict': results, 'error_list': error_list}, 1)
#     ], input_queue=inputfile_queue)
#     #pd.to_timedelta(np.timedelta64(int(args.duration), 'h'))
#     # Add input file to queue
#     for f in args.endpoint_files:
#         inputfile_queue.put(f)

#     # Start all processes and setup intermediate queues
#     pipeline.run()

#     pipeline.wait_for_completion()
#     progress_monitor.close()

#     # TODO: Write the errors out into a json file to allow later analysis
#     if len(error_list) > 0:
#         print('Errors occurred during processing:')
#         for e in error_list:
#             print(e)

#     print('Writing results to output file')
#     # print(results)
#     with open(args.output, 'w') as f:
#         json.dump(results.copy(), f, indent=2, cls=DateTimeJSONEncoder)
