#!/usr/bin/env python
import pandas as pd
import numpy as np
from time import time

import os
import ipdb
import sys
sys.path.append('../utils')
sys.path.append('./pharma')
import preproc_utils
from pharmarec_processing import pivot_pharma_table

def LoadRawHDF5Values(tbl_name, 
                      patient_id, 
                      variable_ids=None,
                      no_duplicates=True,
                      verbose=False):
    """
    Load data of selected patients from the HDF file of a table.

    Parameters:
    tbl_name: the name of the table (string)
    patient_id: the selected patient ID
    verbose: bool; while true, print the information of the returned dataframe

    Returns:
    df: a dataframe consisting the selected data
    """

    
    input_path = os.path.join(datapath, '1a_hdf5_clean', version, 'duplicates_removed', tbl_name)
    filename = [f for f in os.listdir(input_path) if '_%d_'%pid_chunkfile_index.loc[patient_id].ChunkfileIndex in f][0]
    df = pd.read_hdf(os.path.join(input_path, filename), where='PatientID = %d'%patient_id, mode='r')

    # If variables of interest are specified, then only select columns of interest and discard
    # the rest
    if variable_ids is not None:
        if tbl_name == 'pharmarec':
            variables_intersect = set(variable_ids) & set(df['PharmaID'].tolist())
        else:
            variables_intersect = set(variable_ids) & set(df['VariableID'].tolist())

            
        if len(variables_intersect) == 0:
            df = df.loc[[]]
        else:
            if tbl_name == 'pharmarec':
                df = df[df['PharmaID'].isin(variables_intersect)]
            else:
                df = df[df['VariableID'].isin(variables_intersect)]

    # Rename columns to the same name
    df = df.rename(columns={'PharmaID': 'VariableID', 'GivenDose': 'Value'})

    if tbl_name == 'pharmarec':
        df = df[['PatientID', 'Datetime', 'VariableID', 'InfusionID', 'Rate', 'Value', 'Status']]
    else:
        df = df[['PatientID', 'Datetime', 'VariableID', 'Value']]
        
    if verbose:
        if len(df) > 0:
            print('patient ', patient_id, 
                  '; # records ', len(df), 
                  '; # variables ', len(df.VariableID.unique()))
        else:
            print('No data of interest was found for patient ', patient_id)
        
    return df
                    
def remove_invalid_chars(variable_name):
    """
    Remove chars that should not appear in the column name of a dataframe.
    """
    for x in [' ', '-', '(', ')', ',', '/', ':']:
        if x in variable_name:
            variable_name = variable_name.replace(x,'_')
    for x in ['.', '*', '+']:
        if x in variable_name:
            variable_name = variable_name.replace(x, '')
    return variable_name

def table2matrix(tbl_name, df_tbl, variables_of_interest):
    """
    Export data from table to a feature matrix, where each column represents a variable.
    
    Parameter:
    df_tbl: the dataframe containing these columns: Datetime, PatientID, VariableID and Value
    variables_of_interest: a table of the mapping between their variable IDs and variable 
    names of variables that we are interested in.
    
    Returns:
    df_mat: the feature matrix, whose columns are associated with variables.
    """

    # If we choose to use the original name of the variables instead of IDs as the column names,
    # we need to remove some of the invalid chars from the variable names. 
    voi_tmp = variables_of_interest.copy()
    voi_tmp.VariableName.apply(lambda x: remove_invalid_chars(x))

    df_tbl = df_tbl.join(voi_tmp, on='VariableID', how='inner')
    
    if tbl_name == 'pharmarec':
        df_mat = pivot_pharma_table(df_tbl, switch='steph')
    else:
        df_tbl.drop('VariableID', axis=1, inplace=True)
        df_mat = pd.pivot_table(df_tbl, values='Value', index=['PatientID', 'Datetime'],
                                columns=['VariableName'])

    # Add the variables that are among the variables of interest but were not measured for the patients
    # in df_tbl. 
    missing_columns = set(voi_tmp.VariableName.tolist()) - set(df_mat.columns)
    for col in missing_columns:
        if tbl_name == 'pharmarec':
            df_mat[col] = 0
        else:
            df_mat[col] = np.nan    
    
    df_mat.reset_index(inplace=True)
    
    return df_mat


def main(tbl_name, index_chunk, output_to_disk=True):

    pid_list = preproc_utils.get_filtered_patient_ids()
    
    output_path = os.path.join(datapath, '2_pivoted', version, tbl_name)
    if output_to_disk and not os.path.exists(output_path):
        os.makedirs(output_path)
    pid_list = np.array(pid_chunkfile_index.index[pid_chunkfile_index.ChunkfileIndex==index_chunk])
    output_path = os.path.join(output_path, '%s__%d__%d--%d.h5'%(tbl_name, index_chunk, np.min(pid_list), np.max(pid_list)))

    if output_to_disk and os.path.exists(output_path):
        print('Already os.path.exists: %s.'%output_path)
        print('Please delete it manually if you want to reproduce a new file.')
        return -1

    # replace_name = False means the variableNames are vIDs (or pIDs for the pharma table)
    variables_of_interest = preproc_utils.voi_id_name_mapping(tbl_name, replace_name=False, version=version)
    vid_list = variables_of_interest.index.tolist()
    
    df_idx_start = 0
    num_pid = len(pid_list)
    for i in range(num_pid):
        t_total = 0
        
        t = time()
        df_tbl = LoadRawHDF5Values(tbl_name, pid_list[i], variable_ids=vid_list, verbose=True)
        t_read = time() - t
        print('Read time', t_read, 'secs')
        t_total += t_read

        if len(df_tbl) > 0:
            t = time()
            df_mat = table2matrix(tbl_name, df_tbl, variables_of_interest)
            t_pivot = time() - t
            print('Table-to-matrix transform time', t_pivot, 'secs')
            t_total += t_pivot

            if tbl_name != 'pharmarec':
                try:
                    assert(len(df_mat)==len(df_tbl.drop_duplicates(['Datetime'])))
                except AssertionError:
                    print('Timestamp number mismatches between the feature matrix and the table.')
                    pass
                    # import ipdb
                    # ipdb.set_trace()

                try:
                    assert(len(df_tbl.VariableID.unique())==np.sum(np.sum(pd.notnull(df_mat.iloc[:,2:]), axis=0)>0))
                except AssertionError:
                    print('Variable number mismatches between the feature matrix and the table')
                    ipdb.set_trace()


            sum_tbl_value = np.sum(df_tbl.Value)
            if tbl_name == 'pharmarec':
                sum_mat_value = 0
                non_zero_drugs = df_mat.dropna(axis=1, how='all').columns[2:]
                for col in non_zero_drugs:
                    df_drug = df_mat.loc[:, ['Datetime', col]]
                    df_drug.set_index('Datetime', inplace=True)
                    df_drug.sort_index(inplace=True)
                    df_drug.dropna(inplace=True)
                    if df_drug.shape[0] < 2:
                        if df_drug.values.sum() == 0:
                            # carry on
                            continue
                        else:
                            print('instantaneous drug?')
                            ipdb.set_trace()
                    # this seems to be converting back from rate into given doses to compare that the contents of the dataframe are unchanged (in aggregate, at least)
                    # time_difference calculates a time difference in hours, convert to minutes
                    # TODO somethign is a bit weird here, but ignore it for now
                    tdiff = ((df_drug.index[1:] - df_drug.index[:-1]).astype('timedelta64[s]').values)/60
                    total_dose = np.dot(df_drug.values[:-1].reshape(-1, ), tdiff)
                    sum_mat_value += total_dose
                    
            else:
                sum_mat_value = np.nansum(df_mat.iloc[:,2:])

            try:
                # NOTE: due to rescaling, this should not be true for many pharma records
                assert(np.abs(sum_tbl_value - sum_mat_value) < 1e-4)
            except AssertionError:
                # If the big difference was due to the large absolute values then we look at the the relative difference
                if sum_tbl_value != 0 : 
                    try:
                        assert(np.abs(sum_tbl_value - sum_mat_value) / sum_tbl_value < 1e-4)
                    except AssertionError:
                        print('The sum of values in the feature matrix does not match with the sum in the table.')
                        print('\t\t table:', sum_tbl_value)
                        print('\t\t matrix:', sum_mat_value)
                        if tbl_name == 'pharmarec':
                            print('... but this is expected behaviour due to drug rescaling')
                        else:
                            ipdb.set_trace()
                
            df_mat.set_index(np.arange(df_idx_start, df_idx_start+len(df_mat)), inplace=True)
            df_idx_start += len(df_mat)
            for col in df_mat.columns[2:]:
                df_mat[col] = df_mat[col].astype(float)
            if output_to_disk:
                t = time()
                df_mat.to_hdf(output_path, 'pivoted', append=True, complevel=5, 
                              complib='blosc:lz4', data_columns=['PatientID'], format='table')
                t_write = time() - t
                print('Write time', t_write, 'secs')
                t_total += t_write
                
        print('Patient %d / %d finished; Total runtime = %g sec'%(i, num_pid, t_total))
    return 1

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tbl_name')    
    parser.add_argument('-version')        
    parser.add_argument('--index_chunk', type=int, default=None)
    parser.add_argument('--output_to_disk', action='store_true')
    args = parser.parse_args()
    tbl_name = args.tbl_name
    version = args.version
    index_chunk = args.index_chunk
    output_to_disk = args.output_to_disk

    datapath = preproc_utils.get_datapath()
    pid_chunkfile_index = preproc_utils.get_chunking_info(version=version)

    main(tbl_name, index_chunk, output_to_disk)
