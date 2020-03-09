# sanity check that the dimensionality reduction didnt damage data integrity
# author: stephanie hyland

import pandas as pd
import glob as glob
import ipdb
import sys

import paths
folder = paths.root_dir + '3_merged/v6b/'
reduced_folder = folder  + 'reduced/'

#v6
check_columns = {'MAP': [['v110'], ['vm5']],
        'HR': [['v200'], ['vm1']],
        'a_Lac': [['v24000524'], ['vm136']],
        'v_Lac': [['v24000732', 'v24000485'], ['vm146']]}

columns_of_interest = ['endpoint_status', 'PatientID', 'Datetime']
columns_of_uninterest = ['event1', 'event2', 'event3',
        'maybe_event1', 'maybe_event2', 'maybe_event3',
        'probably_not_event1', 'probably_not_event2', 'probably_not_event3']

DEBUGSTOP = None

def check_file(f):
    hdf_name = f.split('/')[-1]
    print('checking', hdf_name)
    df = pd.read_hdf(folder + hdf_name, columns=['PatientID', 'Datetime'], stop=DEBUGSTOP)
    df_reduced = pd.read_hdf(folder + 'reduced/reduced_' + hdf_name, columns=['PatientID', 'Datetime'], stop=DEBUGSTOP)
    try:
        assert (df['Datetime'] == df_reduced['Datetime']).all()
    except AssertionError:
        print('ERROR: differing datetimes!')
        return False
    try:
        assert (df['PatientID'] == df_reduced['PatientID']).all()
    except AssertionError:
        print('ERROR: differing patient IDs!')
        return False
    del df
    del df_reduced
    for k, v in check_columns.items():
        print('comparing variable', k)
        df = pd.read_hdf(folder + hdf_name, columns=['PatientID', 'Datetime'] + v[0], stop=DEBUGSTOP)
        df_reduced = pd.read_hdf(folder + 'reduced/reduced_' + hdf_name, columns=['PatientID', 'Datetime'] + v[1], stop=DEBUGSTOP)
        try:
            assert df.loc[:, v[0]].median(axis=1).dropna().shape == df_reduced.loc[:, v[1]].dropna().median(axis=1).shape
        except AssertionError:
            print('ERROR: dataframes have different NA patterns')
            print(df.loc[:, v[0]].median(axis=1).dropna().shape, df_reduced.loc[:, v[1]].dropna().median(axis=1).shape)
            return False
        differences = (df.loc[:, v[0]].median(axis=1).dropna() != df_reduced.loc[:, v[1]].median(axis=1).dropna())
        try:
            assert sum(differences) == 0
        except AssertionError:
            print('ERROR: dataframes differ in', sum(differences), 'places')
            return False
        print('All okay!')
        del df
        del df_reduced
    return True

files = glob.glob(folder + '*.h5')
idx = int(sys.argv[1])

reduced_is_good = check_file(files[idx])
status_file = open('status.txt', 'a')
if reduced_is_good:
    print(files[idx], 'is good')
    status_file.write(files[idx] + ',good\n')
else:
    print(files[idx], 'is bad')
    status_file.write(files[idx] + ',bad\n')
status_file.close()
