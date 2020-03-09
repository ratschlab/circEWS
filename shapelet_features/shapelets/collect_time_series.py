#!/usr/bin/env python3
#
# Environment:
#
# module load hdf5
# module load python_gpu/3.6.4

import csv
import gc
import json
import os
from tqdm import tqdm

import pandas as pd
import numpy as np

from collections import defaultdict
import logging
from utils.cli import nonexistant_file, existing_file
log = logging.getLogger('collect_time_series')


def process_batch_file(batch_file, selected_variables, event_times):
    log.info('Processing {}...'.format(batch_file))
    differences = False
    if np.any(['IMPUTED_STATUS_CUM_COUNT' in i for i in selected_variables]):
        differences = True

    # Data in each of these batches is stored as part of the 'imputed'
    # group.
    if 'binarized' in batch_file:
        try:
            data = pd.read_hdf(batch_file, mode='r', columns=['PatientID', 'AbsDatetime'] + selected_variables)
        except:
            data = pd.DataFrame(columns=['PatientID', 'AbsDatetime'] + selected_variables)
    else:
        data = pd.read_hdf(batch_file, 'imputed', mode='r', columns=['PatientID', 'AbsDatetime'] + selected_variables)
    data['AbsDatetime'] = pd.to_datetime(data['AbsDatetime'])

    # Check availability of variables and notify the user if some of the
    # selected ones are not present.
    available_variables = set(data.columns)
    missing_variables = available_variables - set(selected_variables)

    if missing_variables:
        log.warning('Some of the selected variables are missing in batch {}: {}'.format(batch_file, missing_variables))

    selected_variables = list(available_variables.intersection(selected_variables))
    results = defaultdict(list)

    for patient_id, patient_data in data.groupby('PatientID'):

        # Required for downstream processing
        patient_data.set_index('AbsDatetime', inplace=True, drop=False)

        # Skip all unknown patients for now
        if str(patient_id) not in event_times.keys():
            log.error('Unable to find patient {} in file. Skipping...'.format(patient_id))
            continue
        else:
            log.debug('Processing patient', patient_id)

        # Reads *all* fragments belonging to the current patient. Note
        # that these are only 'snippets' of the time series without an
        # attached variable (so far!)
        fragments = pd.DataFrame.from_dict(event_times[str(patient_id)])

        # Remove unnecessary variables
        time_series = patient_data[selected_variables]

        # Iterate over each fragment, i.e. over each row of the
        # output file.
        for _, fragment in fragments.iterrows():
            start = pd.to_datetime(fragment['Start'])
            end = pd.to_datetime(fragment['End'])
            label = fragment['s3m_endpoint']

            fragment_time_series = time_series.loc[start:end]
            times = fragment_time_series.index.to_series()

            # Sanity check: ensure that both data files have been
            # resampled in the same manner.
            if times.iloc[0] != start or times.iloc[-1] != end:
                log.warn('time specifications in fragments are not matching times from endpoints files: {} vs {} and {} vs {}'.format(times.iloc[0], start, times.iloc[-1], end))
            if differences:
                for selected_variable in selected_variables:
                    to_put = fragment_time_series[selected_variable].diff()
                    to_put[0] = 0
                    results[selected_variable].append([
                        str(patient_id),
                        str(start),
                        str(end),
                        str(label)] +
                        [str(x) for x in to_put])
            else:
                for selected_variable in selected_variables:
                    results[selected_variable].append([
                        str(patient_id),
                        str(start),
                        str(end),
                        str(label)] +
                        [str(x) for x in fragment_time_series[selected_variable]]
                    )

    # Explicitly removing all references to the data frame here. This is
    # just a workaround until we have figured out what is going wrong in
    # the other case.
    del data
    gc.collect()
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_files', nargs='+', type=existing_file, help='Endpoint files in HDF5 format')
    parser.add_argument('--variables', nargs='+', default=['v200'], help='Variables to extract from patient records')
    parser.add_argument('--event-times', required=True, type=existing_file, help='Event times files in json format')
    parser.add_argument('--output', required=True, type=nonexistant_file, help='Output file in csv format')
    parser.add_argument('--logfile', type=nonexistant_file, help='File in which to store log')

    args = parser.parse_args()
    loglevel = logging.INFO
    if args.logfile:
        logging.basicConfig(level=loglevel, filename=args.logfile)
    else:
        logging.basicConfig(level=loglevel)

    with open(args.event_times) as f:
        event_times = json.load(f)

    # Build output filename
    # If output is a directroy create a file for each variable within the dir
    # otherwise use output path as prefix to _variablename.csv
    output_is_dir = os.path.isdir(args.output) or args.output.endswith('/')
    if output_is_dir:
        if os.path.isdir(args.output):
            print('The output path specified already exists! Exising files will be overwritten!')
        # Dont accept existing directories to ensure files are not overwritten
        os.makedirs(args.output, exist_ok=True)
        glueing_char = ''
        if not args.output.endswith('/'):
            args.output += '/'
    else:
        glueing_char = '_'

    filename_mapping = {variable: '{}{}{}.csv'.format(args.output, glueing_char, variable)
                        for variable in args.variables}

    for batch_file in tqdm(args.batch_files, total=len(args.batch_files)):
        processed = process_batch_file(batch_file, args.variables, event_times)
        for var, values in processed.items():
            with open(filename_mapping[var], mode='a') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerows(values)


# #!/usr/bin/env python3
# #
# # Environment:
# #
# # module load hdf5
# # module load python_gpu/3.6.4

# import csv
# import gc
# import json
# import os
# from tqdm import tqdm
# import numpy as np

# import pandas as pd

# from collections import defaultdict
# import logging
# from utils.cli import nonexistant_file, existing_file
# log = logging.getLogger('collect_time_series')


# def process_batch_file(batch_file, selected_variables, event_times):
#     log.info('Processing {}...'.format(batch_file))

#     # Data in each of these batches is stored as part of the 'imputed'
#     # group.
#     data = pd.read_hdf(batch_file, 'imputed', mode='r', columns=['PatientID', 'AbsDatetime'] + selected_variables)
#     data['AbsDatetime'] = pd.to_datetime(data['AbsDatetime'])

#     # Check availability of variables and notify the user if some of the
#     # selected ones are not present.
#     available_variables = set(data.columns)
#     missing_variables = available_variables - set(selected_variables)

#     if missing_variables:
#         log.warning('Some of the selected variables are missing in batch {}: {}'.format(batch_file, missing_variables))

#     selected_variables = list(available_variables.intersection(selected_variables))
#     results = defaultdict(list)

#     for patient_id, patient_data in data.groupby('PatientID'):

#         # Required for downstream processing
#         patient_data.set_index('AbsDatetime', inplace=True, drop=False)

#         # Skip all unknown patients for now
#         if str(patient_id) not in event_times.keys():
#             log.error('Unable to find patient {} in file. Skipping...'.format(patient_id))
#             continue
#         else:
#             log.debug('Processing patient', patient_id)

#         # Reads *all* fragments belonging to the current patient. Note
#         # that these are only 'snippets' of the time series without an
#         # attached variable (so far!)
#         fragments = pd.DataFrame.from_dict(event_times[str(patient_id)])

#         # Remove unnecessary variables
#         time_series = patient_data[selected_variables]

#         # Iterate over each fragment, i.e. over each row of the
#         # output file.
#         for _, fragment in fragments.iterrows():
#             start = pd.to_datetime(fragment['Start'])
#             end = pd.to_datetime(fragment['End'])
#             label = fragment['s3m_endpoint']

#             fragment_time_series = time_series.loc[start:end]
#             times = fragment_time_series.index.to_series()

#             # Sanity check: ensure that both data files have been
#             # resampled in the same manner.
#             if times.iloc[0] != start or times.iloc[-1] != end:
#                 log.warn('time specifications in fragments are not matching times from endpoints files: {} vs {} and {} vs {}'.format(times.iloc[0], start,
#                                                                                                                                       times.iloc[-1], end))
#             if differences:
#                 for selected_variable in selected_variables:
#                     to_put = fragment_time_series[selected_variable].diff()
#                     to_put[0] = 0
#                     results[selected_variable].append([
#                         str(patient_id),
#                         str(start),
#                         str(end),
#                         str(label)] +
#                         [str(x) for x in to_put])
#             else:
#                 for selected_variable in selected_variables:
#                     results[selected_variable].append([
#                         str(patient_id),
#                         str(start),
#                         str(end),
#                         str(label)] +
#                         [str(x) for x in fragment_time_series[selected_variable]])

#     # Explicitly removing all references to the data frame here. This is
#     # just a workaround until we have figured out what is going wrong in
#     # the other case.
#     del data
#     gc.collect()
#     return results


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('batch_files', nargs='+', type=existing_file, help='Endpoint files in HDF5 format')
#     parser.add_argument('--variables', nargs='+', default=['v200'], help='Variables to extract from patient records')
#     parser.add_argument('--event-times', required=True, type=existing_file, help='Event times files in json format')
#     parser.add_argument('--output', required=True, type=nonexistant_file, help='Output file in csv format')
#     parser.add_argument('--logfile', type=nonexistant_file, help='File in which to store log')

#     args = parser.parse_args()
#     loglevel = logging.INFO
#     if args.logfile:
#         logging.basicConfig(level=loglevel, filename=args.logfile)
#     else:
#         logging.basicConfig(level=loglevel)

#     with open(args.event_times) as f:
#         event_times = json.load(f)

#     # Build output filename
#     # If output is a directroy create a file for each variable within the dir
#     # otherwise use output path as prefix to _variablename.csv
#     output_is_dir = os.path.isdir(args.output) or args.output.endswith('/')
#     if output_is_dir:
#         if os.path.isdir(args.output):
#             print('The output path specified already exists! Exising files will be overwritten!')
#         # Dont accept existing directories to ensure files are not overwritten
#         os.makedirs(args.output, exist_ok=True)
#         glueing_char = ''
#         if not args.output.endswith('/'):
#             args.output += '/'
#     else:
#         glueing_char = '_'

#     filename_mapping = {variable: '{}{}{}.csv'.format(args.output, glueing_char, variable)
#                         for variable in args.variables}

#     for batch_file in tqdm(args.batch_files, total=len(args.batch_files)):
#         processed = process_batch_file(batch_file, args.variables, event_times)
#         for var, values in processed.items():
#             with open(filename_mapping[var], mode='a') as f:
#                 csv_writer = csv.writer(f, delimiter=',')
#                 csv_writer.writerows(values)
