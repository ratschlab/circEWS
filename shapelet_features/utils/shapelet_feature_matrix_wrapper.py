import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)


class ShapeletFeatureMatrixWrapper(object):
    def __init__(self, filename, patient_col='PatientID'):
        self.filename = filename
        self.patient_col = patient_col
        self.logger = logging.getLogger('ShapeletFeatureMatrixWrapper')
        self.lookup_dict = self._construct_lookup_table()
        print("Created lookup dict!")

    def _construct_lookup_table(self):
        lookup_dict = {}
        with open(self.filename, 'r') as f:
            row_counter = 0
            start_of_p = 0
            prev_line = 0

            line = f.readline()
            self.header = line.strip().split(',')
            self.patient_col_index = self.header.index(self.patient_col)
            start_of_p = f.tell()
            line = f.readline()
            current_p = line.split(',')[self.patient_col_index]

            while line != '':
                this_line = line.split(',')
                if this_line[self.patient_col_index] == current_p:
                    # We are still in a consecutive fragment of patient data
                    # Increase the number of rows associated with the patient
                    row_counter += 1
                else:
                    # We encountered a new patient and thus need to store
                    # the information about the previous patient
                    logging.debug('Adding patient {} to lookup dict'.format(current_p))
                    if int(current_p) in lookup_dict.keys():
                        self.logger.error('The PatientID \'{}\' already exists in the lookup_dict! '
                                          'This indicates that the data of an individual patient is not consecutive in the input file!'
                                          .format(current_p))

                    lookup_dict[int(current_p)] = (start_of_p, row_counter + 1)
                    current_p = this_line[self.patient_col_index]
                    row_counter = 0
                    start_of_p = prev_line
                prev_line = f.tell()
                line = f.readline()

            if current_p != -1:
                lookup_dict[current_p] = (start_of_p, row_counter + 1)

        return lookup_dict

    def query(self, patients):
        """This function returns a pandas dataframe of the patients in the query"""
        extracted_regions = []
        with open(self.filename, 'r') as f:
            for patient in patients:
                try:
                    patient = int(patient)
                    start_pos, nrows = self.lookup_dict[patient]
                    f.seek(start_pos, 0)
                    extracted_regions.append(pd.read_csv(f, nrows=nrows, names=self.header, header=None))
                except Exception as e:
                    self.logger.error('Exception occured reading patient \'{}\' from shapelet features file: {}'.format(patient, type(e)))
        if len(extracted_regions) > 0:
            return pd.concat(extracted_regions, axis=0, ignore_index=True)
        else:
            self.logger.warning('None of the patients {} were found in the input file!'.format(patients))
            return pd.DataFrame(columns=self.header)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read a shaplelet features file without loading everything into memory')
    parser.add_argument('hdf5_file')
    parser.add_argument('--query', nargs='+', type=int, default=[])
    args = parser.parse_args()

    wrapper = HDF5ShapeletFeatureMatrixWrapper(args.hdf5_file)
    print(wrapper.query(args.query))

