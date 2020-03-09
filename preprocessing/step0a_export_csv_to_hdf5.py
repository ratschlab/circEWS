import pandas as pd
import numpy as np

from os.path import join

import sys
sys.path.append('../utils')
import preproc_utils

class CsvLoaderMain:
    def __init__(self, data_path):
        self.data_path = data_path

    def LoadCsv2HDF5(self, tbl_name, write_path = './'):
        if tbl_name in ['monvals', 'comprvals', 'dervals']: 
            fields = ['Value', 'VariableID', 'PatientID', 'Datetime', 'Status', 
                      'Entertime' if tbl_name=='monvals' else 'EnterTime']
            dtype = {'Value': np.float64, 'VariableID': np.int64, 'PatientID': np.int64, 
                     'Datetime': np.str_, 'Status': np.int64, 'EnterTime': np.str_}

        elif tbl_name == 'generaldata':
            fields = ['PatientID', 'birthYear', 'Sex', 'AdmissionTime', 'Status', 'PatGroup']
            dtype = {'PatientID': np.int64, 'birthYear': np.int64, 'Sex': np.str_, 
                     'AdmissionTime': np.str_, 'Status': np.int64, 'PatGroup': np.int64}

        elif tbl_name == 'labres':
            fields = ['ResultID', 'Value', 'VariableID', 'PatientID', 'SampleTime', 'Status', 'EnterTime']
            dtype = {'Value': np.float64, 'ResultID': np.int64, 'VariableID': np.int64, 
                     'PatientID': np.int64, 'SampleTime': np.str_, 'Status': np.int64, 'EnterTime': np.str_}

        elif tbl_name == 'observrec':
            fields = ['Value', 'VariableID', 'PatientID', 'DateTime', 'Status', 'EnterTime']
            dtype = {'Value': np.float64, 'VariableID': np.int64, 'PatientID': np.int64, 
                     'DateTime': np.str_, 'Status': np.int64, 'EnterTime': np.str_}

        elif tbl_name == 'pharmarec':
            fields = ['CumulDose', 'GivenDose', 'Rate', 'PharmaID', 'InfusionID', 'Route', 
                      'Status', 'PatientID', 'DateTime', 'EnterTime']
            dtype = {'CumulDose': np.float64, 'GivenDose': np.float64, 'Rate': np.float64, 
                     'PharmaID': np.int64, 'InfusionID': np.int64, 'Route': np.int64, 
                     'Status': np.int64, 'PatientID': np.int64, 'DateTime': np.str_,
                     'EnterTime': np.str_}
        else:
            raise Exception('Wrong table name.')

        filepath_csv = join(self.data_path, 'expot-%s.csv'%tbl_name)
        filepath_hdf5 = join(write_path, '%s.h5'%tbl_name)

        pID_set = set(preproc_utils.get_consent_patient_ids().tolist())

        if tbl_name in ['monvals', 'comprvals', 'dervals']:
            chunksize = 10 ** 7
            iter_csv = pd.read_csv(filepath_csv, encoding='cp1252', na_values='(null)', 
                                   sep=';', low_memory=True, usecols=fields, dtype=dtype, 
                                   chunksize=chunksize)
            vID_set = []
            pID_consent_set = []
            for i, chunk in enumerate(iter_csv):
                print(i)
                chunk.Datetime = pd.to_datetime(chunk.Datetime)
                pID_consent = set(chunk.PatientID.unique()) & pID_set
                is_consent = [(x in pID_consent) for x in chunk.PatientID.tolist()]
                chunk = chunk[is_consent]
                for col in chunk.columns:
                    if 'time' in col.lower():
                        chunk[col] = pd.to_datetime(chunk[col])
                vID_set.extend(chunk.VariableID.unique().tolist())
                pID_consent_set.extend(chunk.PatientID.unique().tolist())
                chunk.to_hdf(filepath_hdf5, 'raw_import', append=True, complevel=5, 
                             complib='zlib', data_columns=True, format='table')
            print('Number of patients = %d'%len(np.unique(pID_consent_set)))
            print('Number of variables = %d'%len(np.unique(vID_set)))
        else:
            df = pd.read_csv(filepath_csv, encoding='cp1252', na_values='(null)', sep=';', 
                             low_memory=True, usecols=fields, dtype=dtype)        
            pID_consent = set(df.PatientID.unique()) & pID_set
            is_consent = [(x in pID_consent) for x in df.PatientID.tolist()]
            df = df[is_consent]
            for col in df.columns:
                if 'time' in col.lower():
                    df[col] = pd.to_datetime(df[col])
            print('Number of patients = %d'%len(df.PatientID.unique()))
            if tbl_name == 'pharmarec':
                print('Number of variables = %d'%len(df.PharmaID.unique()))
            elif tbl_name != 'generaldata':
                print('Number of variables = %d'%len(df.VariableID.unique()))
            df.to_hdf(filepath_hdf5, 'raw_import', complevel=5, complib='zlib', data_columns=True, format='table')
        print('%s is created.'%filepath_hdf5)

read_path = r'/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/0_csv_exports'
write_path = r'/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/1_hdf5_consent/%s'%preproc_utils.data_version

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('tbl_name')
    args = parser.parse_args()
    tbl_name = args.tbl_name
    l = CsvLoaderMain(read_path)
    l.LoadCsv2HDF5(tbl_name, write_path)
