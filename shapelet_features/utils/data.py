import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


class DataLoader:
    def __init__(self, classification_matrix, splitfile, variable_filter, splitfile_column='temporal_5',
                 label_column='Y', meta_columns=['PatientID', 'AbsDatetime', 'RelDatetime']):
        self.classification_matrix = classification_matrix
        self.splitfile = splitfile
        self.splitfile_column = splitfile_column
        self.label_column = label_column
        self.meta_columns = meta_columns
        self.split_df = self._get_split()
        self.log = logging.getLogger(self.__class__.__name__)
        self.variable_filter = variable_filter

    def _get_split(self):
        return pd.read_csv(self.splitfile, sep='\t', index_col=0)[self.splitfile_column]

    def _get_patients_from_split(self, criterion):
        return self.split_df[self.split_df == criterion].index

    def _read_from_classification_matrix(self, patients, batch_size=2000):
        self.log.debug('Reading {} patients from classification matrix'.format(len(patients)))
        # This reads patients from a dataframe in a batched fashion in order to
        # keep the memory footprint as small as possible
        num_sections = int(np.ceil(len(patients) / batch_size))
        collected_data = None
        selected_columns = None
        with pd.HDFStore(self.classification_matrix, 'r') as f, tqdm(total=len(patients)) as progressbar:
            for patients_batch in np.array_split(patients, num_sections):
                batch_fragments = []
                for patient in patients_batch:
                    try:
                        progressbar.set_description('Reading patient {}'.format(patient))
                        patient = str(patient)
                        d = f.get(patient)
                        if selected_columns is None:
                            # Label column and meta_columns should always be extracted!
                            # Create set and convert back to list to ensure we dont have duplicates
                            # This could occur for example if meta columns are not filtered with the variable filter
                            selected_columns = list(set(
                                self.meta_columns + self.variable_filter(d.columns) + [self.label_column]
                            ))
                        batch_fragments.append(d[selected_columns])
                    except Exception as e:
                        progressbar.write('Unable to read patient {} from classification matrix'.format(patient))
                        #self.log.error('Exception occurred reading patient \'{}\' from classification matrix file: {}'.format(patient, type(e)))
                    finally:
                        progressbar.update()
                if len(batch_fragments) > 0:
                    progressbar.set_description('Joining fragments')
                    if collected_data is None:
                        collected_data = pd.concat(batch_fragments, ignore_index=True)
                    else:
                        collected_data = pd.concat([collected_data] + batch_fragments, ignore_index=True)
        return collected_data

    def get_classification_data(self, data_class):
        training_patients = self._get_patients_from_split(data_class)
        data = self._read_from_classification_matrix(training_patients)
        meta_data = data[self.meta_columns].copy()
        labels = data[self.label_column].copy()
        # Remove columns that should not be part of the training data

        # These should never end up in the regular training data
        data.drop(columns=['PatientID', 'AbsDatetime', self.label_column], inplace=True)
        # Construct a list of columns that should be dropped to ensure memory is freed instead of copied
        # Sets are faster when using the `in` operation
        keep_columns = set(self.variable_filter(data.columns))
        drop_columns = [col for col in data.columns if col not in keep_columns]
        data.drop(columns=drop_columns, inplace=True)
        return data, meta_data, labels

    def training_data(self):
        return self.get_classification_data('train')

    def validation_data(self):
        return self.get_classification_data('val')

    def testing_data(self):
        return self.get_classification_data('test')


class HDF5ShapeletFeatureMatrixWrapper(object):
    def __init__(self, filename):
        self.filename = filename
        self.logger = logging.getLogger('HDF5ShapeletFeatureMatrixWrapper')

    def query(self, patients):
        """This function returns a pandas dataframe of the patients in the query"""
        extracted_regions = []
        with pd.HDFStore(self.filename, 'r') as f:
            for patient in patients:
                try:
                    patient = str(patient)
                    extracted_regions.append(f.get(patient))
                except Exception as e:
                    self.logger.error('Exception occurred reading patient \'{}\' from shapelet features file: {}'.format(patient, type(e)))
        if len(extracted_regions) > 0:
            return pd.concat(extracted_regions, axis=0)
        else:
            self.logger.warning('None of the patients {} were found in the input file!'.format(patients))
            return pd.DataFrame()


class VariableFilter:
    def __init__(self, keep_variables=None, drop_variables=None):
        self.log = logging.getLogger(self.__class__.__name__)

        if keep_variables and drop_variables:
            raise ValueError('Cannot both keep and drop variables at the same time')

        # If none are specified, keep all variables by default. This is
        # realized by specifying an empty list of variables to drop.
        if not keep_variables and not drop_variables:
            drop_variables = []

        if keep_variables:
            self.variables = keep_variables
            self.drop = False
        else:
            # This case also covers no variables being passed at all (both None)
            # The matches criterion will always return False such that no variable is dropped
            self.variables = drop_variables
            self.drop = True

    def _matches(self, column_name):
        '''
        Checks whether a given column name is a substring of at least
        one variable of the list of filter variables.
        '''
        if 'vm' in self.variables:
            return any([True if re.search(variable, column_name) else False for variable in self.variables])
        else:
            return any([True if variable.upper().lower() == column_name.upper().lower() else False for variable in self.variables])

    def __call__(self, variable_names):
        '''
        Returns all variables that are *kept*, i.e. the ones that
        survive the filter operation.
        '''

        if self.drop:
            filtered_variable_names = [name for name in variable_names if not self._matches(name)]
        else:
            filtered_variable_names = [name for name in variable_names if self._matches(name)]
        removed_variables = set(variable_names) - set(filtered_variable_names)
        self.log.debug('Removing columns {} due to variable filter'.format(', '.join(removed_variables)))

        return filtered_variable_names


class NANRemover:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def __call__(self, feature_matrix):
        '''
        Removes NaNs from a feature matrix and gives some informative output
        if necessary.
        '''

        n_rows_before = len(feature_matrix)

        # Remove invalid data from rows; we need this because no
        # classifier is capable of dealing with those data.
        feature_matrix = feature_matrix.dropna()

        n_rows_after = len(feature_matrix)

        if n_rows_after != n_rows_before:
            self.log.info('Removed {} rows of {} rows due to NaNs'.format(n_rows_before - n_rows_after, n_rows_before))

        return feature_matrix


class StaticVariableMerger:
    def __init__(self, static_variables_path, variable_filter):
        self.static_variables_path = static_variables_path
        self.variable_filter = variable_filter
        self.static_variables_df = self._get_static_variables()
        self.log = logging.getLogger(self.__class__.__name__)

    def _get_static_variables(self):
        static_df = pd.read_hdf(self.static_variables_path)
        selected_static_variables = self.variable_filter(static_df.columns)
        # Ensures that 'PatientID' is always part of the selected variables
        # because we need it for the merge.
        if 'PatientID' not in selected_static_variables:
            selected_static_variables = ['PatientID'] + selected_static_variables
        return static_df[selected_static_variables]

    def __call__(self, other_data):
        if len(self.static_variables_df.columns) > 1:
            self.log.debug('Merging static variables into feature matrix')
            feature_matrix = pd.merge(other_data, self.static_variables_df, on='PatientID')
        else:
            self.log.debug('No static variables have been selected')
            feature_matrix = other_data

        # This ensures that we get a standard `RangeIndex` instead of
        # something more complex. Otherwise, we cannot save a matrix
        # in HDF5 format.
        feature_matrix.reset_index(inplace=True, drop=True)
        return feature_matrix
