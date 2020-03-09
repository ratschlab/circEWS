import gc
import json
import logging
import os
import warnings

from datetime import datetime
import numpy as np
from enum import auto, Enum

from multiprocessing import Process, Event, JoinableQueue
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Empty, Full

from tqdm import tqdm


class MyManager(SyncManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Small hack to allow setting of the total number of steps


def set_total(self, total):
    self.total = total


tqdm.set_total = set_total
MyManager.register('tqdm', tqdm)


class DateTimeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return json.JSONEncoder.default(self, obj)


class ProcessingError():
    class Type(Enum):
        NoValidEndpoint = auto()
        PatientIdDuplicate = auto()
        InsufficientData = auto()
        CannotAlignDatetimes = auto()    # Error is set if we cannot find the patients in the patient timepoints file used for alignment.

    def __init__(self, patientid, errortype, **kwargs):
        self.patientid = patientid
        self.errortype = errortype
        self.additional_data = kwargs

    def __str__(self):
        base_str = 'Error: {:20} Patient: {}'.format(self.errortype.name, self.patientid)
        if self.additional_data:
            base_str += ' ' + str(self.additional_data)
        return base_str


class HDF5Reader(Process):
    def __init__(self, input_queue, output_queue, hdf5_group, progressbar, **read_kwargs):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.hdf5_group = hdf5_group
        self.progressbar = progressbar
        self.read_kwargs = read_kwargs
        self.exit = Event()
        super().__init__()

    def run(self):
        self.log = logging.getLogger(str(self))
        while True:
            try:
                next_file = self.input_queue.get(block=True, timeout=1)
            except Empty:
                if self.exit.is_set():
                    break
                else:
                    self.log.debug('Reached timeout while waiting for input')
                    continue
            self.progressbar.set_description('File: {}'.format(os.path.basename(next_file)))
            self.process_HDF5_file(next_file)
            self.input_queue.task_done()
            self.log.debug('Finished processing file {}'.format(os.path.basename(next_file)))
            self.progressbar.update()

    def terminate(self):
        self.exit.set()

    def process_HDF5_file(self, endpoint_file):
        import pandas as pd
        self.log.debug('Processing file {}'.format(os.path.basename(endpoint_file)))

        if 'binarized' in endpoint_file:
            try:
                data = pd.read_hdf(endpoint_file, mode='r')
            except:
                data = pd.DataFrame(columns=['PatientID'])
        else:
            data = pd.read_hdf(endpoint_file, self.hdf5_group, mode='r', **self.read_kwargs)
        self.log.debug('Sucessfully read {}'.format(os.path.basename(endpoint_file)))
        for patient_id, patient_data in data.groupby('PatientID', sort=True, as_index=False):
            while True:
                try:
                    self.output_queue.put((patient_id, patient_data), block=True, timeout=1)
                    break
                except Full:
                    self.log.debug('Reached timeout while trying to add to output queue')

        del data
        gc.collect()


import pickle
import os
import pandas as pd


def load_pickle(fpath):
    ''' Given a file path pointing to a pickle file, yields the object pickled in this file'''
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


fpath = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle'
bern_batch_map = load_pickle(fpath)["pid_to_chunk"]
all_columns = pd.read_hdf('/cluster/work/borgw/Bern_ICU_Sanctuary/v6b_held_out/Shapelet_min-max_dist-set_features.h5', key='/74353').columns.values


class HDF5Aggregator(Process):
    def __init__(self, input_queue, output, error_list):
        self.input_queue = input_queue
        self.output = output
        self.error_list = error_list
        self.exit = Event()
        super().__init__()

    def run(self):
        import pandas as pd
        from tables import NaturalNameWarning
        warnings.filterwarnings("ignore", category=NaturalNameWarning)

        if 'features' in self.output:

            if not os.path.exists(self.output[:-3] + '/'):
                os.makedirs(self.output[:-3] + '/')

            while True:
                try:
                    patient_id, res = self.input_queue.get(block=True, timeout=1)
                except Empty:
                    if self.exit.is_set():
                        break
                    else:
                        continue

                ofile = self.output[:-3] + '/batch_' + str(bern_batch_map[int(patient_id)]) + '.h5'

                for column in all_columns:
                    if column not in res.columns:
                        res[column] = np.nan
                res.fillna(10000000000000, inplace=True)
                res.to_hdf(ofile, "/p{}".format(patient_id), format="fixed", append=False, complevel=5, complib="blosc:lz4")
        else:
            with pd.HDFStore(self.output, 'w') as outputfile:
                while True:
                    try:
                        patient_id, res = self.input_queue.get(block=True, timeout=1)
                    except Empty:
                        if self.exit.is_set():
                            break
                        else:
                            continue
                    outputfile.put('/' + str(patient_id), res)
                    outputfile.flush()

                    self.input_queue.task_done()


class PatientProcessor(Process):
    def __init__(self, processing_function, input_queue, output_queue, error_list, function_args={}):
        self.processing_function = processing_function
        self.function_args = function_args
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.error_list = error_list
        self.exit = Event()
        super().__init__()

    def run(self):
        self.log = logging.getLogger(str(self))
        # Dont show timeout messages in beginning
        while True:
            try:
                patient_id, patient_data = self.input_queue.get(block=True, timeout=1)
            except Empty:
                # termination criterion, ensure that we want the processes to terminate
                # otherwise process might terminate if queue is temporary empty
                if self.exit.is_set():
                    break
                else:
                    self.log.debug('Reached timeout while waiting for input')
                    continue

            # process the patient
            res = self.processing_function(patient_id, patient_data, **self.function_args)

            # If there was a processing error, append it to the error list
            # otherwise pass it to the queue
            if isinstance(res, ProcessingError):
                self.error_list.append(res)
            else:
                while True:
                    try:
                        self.output_queue.put(res, block=True, timeout=1)
                        break
                    except Full:
                        self.log.debug('Reached timeout while trying to add to output queue')

            # Mark task as done
            self.input_queue.task_done()

    def terminate(self):
        self.exit.set()


class Aggregator(Process):
    def __init__(self, input_queue, output_dict, error_list):
        self.input_queue = input_queue
        self.output_dict = output_dict
        self.error_list = error_list
        self.exit = Event()
        super().__init__()

    def run(self):
        while True:
            try:
                res = self.input_queue.get(block=True, timeout=1)
            except Empty:
                if self.exit.is_set():
                    break
                else:
                    continue
            # Skip patient if already present in results dictionary
            patient_id = next(iter(res.keys()))
            if patient_id in self.output_dict.keys():
                self.error_list.append(ProcessingError(patient_id, ProcessingError.Type.PatientIdDuplicate))
                continue

            # Add results to output dictionary
            self.output_dict.update(res)
            self.input_queue.task_done()


class ParallelPipeline(object):
    def __init__(self, steps, input_queue):
        """Expects list of (processclass, process_parameters, n_processes) for steps and inputqueue for first step"""
        self.queues = [input_queue]
        self.processes = []
        # iterate over steps, exclude last one
        for i, step in enumerate(steps[:-1]):
            # Check if step in pipeline has defined bottleneck which limits the size of the output queue.
            # This can be used to synchonize the progress of the downstream processing steps in the pipeline with the current step
            # In our case, this makes the progressbar represent to total progress of processing.
            if len(step) == 3:
                cls, parameters, n_processes = step
                output_queue = JoinableQueue()
            elif len(step) == 4:
                cls, parameters, n_processes, bottleneck = step
                output_queue = JoinableQueue(maxsize=bottleneck)

            self.processes.append([cls(input_queue=self.queues[i], output_queue=output_queue, **parameters) for j in range(n_processes)])
            self.queues.append(output_queue)
        agg_cls, agg_parameters, n_aggregators = steps[-1]
        self.processes.append([agg_cls(input_queue=self.queues[-1], **agg_parameters) for i in range(n_aggregators)])

    def run(self):
        # run all pipeline processes
        for process_array in self.processes:
            for p in process_array:
                p.start()

    def wait_for_completion(self):
        for i, (input_queue, processes) in enumerate(zip(self.queues, self.processes)):
            input_queue.join()
            logging.info('Joined stage {} of pipeline'.format(i))
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
