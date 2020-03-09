#!/usr/bin/ipython
# author: stephanie hyland
# Referring to the excel table, perform dimensionality reduction.

import pandas as pd
import ipdb
import numpy as np
import glob
from time import time
import sys

import paths

excel_path_var = paths.root_dir + 'misc_derived/ref_excel/varref_excel_v6.tsv'
excel_path_lab = paths.root_dir + 'misc_derived/ref_excel/labref_excel_v6.tsv'

in_version = 'v6b'

merged_dir = paths.root_dir + '/3_merged/' + in_version + '/'
merged_reduced_dir = merged_dir + 'reduced/'

def preproc_excel(excel_var, excel_lab):
    # process labs
    excel_lab['MetaVariableID'] = excel_lab['MetaVariableID'].astype('int')
    excel_lab.rename(columns={'VariableName': 'MetaVariableName'}, inplace=True)
    excel = pd.concat([excel_var, excel_lab], axis=0)
    # this is to delete the single row (ECMO) with NaN variable ID
    excel = excel.loc[excel['VariableID'] > 0, :]
    # change IDs to have p, v
    pharma_rows = excel['Type'] == 'Pharma'
    excel.loc[pharma_rows, 'VariableID'] = list(map(lambda x: 'p' + str(int(x)), excel.loc[pharma_rows, 'VariableID']))
    excel.loc[~pharma_rows, 'VariableID'] = list(map(lambda x: 'v' + str(int(x)), excel.loc[~pharma_rows, 'VariableID']))
    excel['Subgroup'] = excel['Subgroup'].fillna('Unknown')
    # convert fractions in text into floats, for flow rate
    fractions = excel.loc[excel['UnitConversionFactor'].notnull()&(excel['MetaVariableUnit'] == 'Flow rate'), 'UnitConversionFactor']
    fractions_as_floats = list(map(lambda x: np.float32(eval(x)), fractions))
    excel.loc[excel['UnitConversionFactor'].notnull()&(excel['MetaVariableUnit'] == 'Flow rate'), 'UnitConversionFactor'] = fractions_as_floats
    return excel

def get_mID_column(m_excel, df, IDs):
    """
    """
    parameter_unit = m_excel['MetaVariableUnit'].dropna().unique()
    if len(parameter_unit) == 0:
        #        print('WARNING: parameter has no units... assuming non drugs')
        parameter_unit = ['missing unit']
    assert len(parameter_unit) == 1
    if parameter_unit in ['[yes/no]', ' [yes/no]']:
        merge_logic = 'binary'
    elif parameter_unit == 'count of drugs':
        merge_logic = 'count presence'
    elif parameter_unit == 'Flow rate':
        merge_logic = 'scale drugs'
    else:
        merge_logic = 'non drugs'
        assert not 'Pharma' in m_excel['Type'].unique()

    # nothing to do here exactly (conversion to binary/count happens at the end)
    if len(IDs) == 1:
        try:
            mID_column = df[IDs[0]]
        except KeyError:
            print('WARNING: couldnt find', IDs[0], 'in data frame')
            mID_column = np.random.normal(size=df.shape[0])
        if merge_logic in ['binary', 'count presence']:
            mID_column = (mID_column > 0).astype(int)
    else:
        # now we have to merge
        if merge_logic == 'scale drugs':
            # need scaling factors
            scaling_factors = []
            for ID in IDs:
                scaling_factor = m_excel.loc[m_excel['VariableID'] == ID, 'UnitConversionFactor'].values[0]
                scaling_factors.append(scaling_factor)
            # sometimes there is no scaling factor - assume 1
            if np.isnan(scaling_factors).any():
                try:
                    assert np.isnan(scaling_factors).all()
                except AssertionError:
                    print(scaling_factors)
                    ipdb.set_trace()
                scaling_factors = [1]*len(IDs)
        else:
            scaling_factors = [1]*len(IDs)
        mID_column = merge_IDs(df, merge_logic, scaling_factors)
        if merge_logic == 'binary':
            mID_column = mID_column.astype(int)
    return mID_column

def merge_IDs(df, merge_logic, scaling_factors):
    if merge_logic == 'binary':
        columns = (df > 0).any(axis=1)
    elif merge_logic == 'count presence':
        columns = (df > 0).sum(axis=1, min_count=1)
    elif merge_logic == 'scale drugs':
        assert len(scaling_factors) == df.shape[1]
        columns = (df*scaling_factors).sum(axis=1, min_count=1)
    elif merge_logic == 'non drugs':
        # pandas median automatically deals with NaN
        # there is no min_count for median, it just produces NaN if there are only NaNs
        columns = df.median(axis=1)
    return columns

def merge_parameter(mID, excel, df_full):
    m_excel = excel.loc[excel['MetaVariableID'] == mID, :]
    assert len(m_excel['MetaVariableName'].dropna().unique()) == 1
    if 'Pharma' in m_excel['Type'].dropna().unique():
        assert len(m_excel['Type'].dropna().unique()) == 1
        mID = 'pm' + str(int(mID))
    else:
        mID = 'vm' + str(int(mID))
    parameter = m_excel['MetaVariableName'].dropna().unique()[0]
    IDs = m_excel['VariableID'].unique()
    print(mID, parameter)
    try:
        df = df_full.loc[:, IDs]
    except KeyError:
        print('WARNING: some of', IDs, 'missing in dataframe for parameter', parameter)
        ipdb.set_trace()
        return np.nan, mID
    mID_column = get_mID_column(m_excel, df, IDs)
    return mID_column, mID

#DEBUG_STOP = 5000

def process_chunk(chunkname, excel):
    print('Processing', chunkname)
    inpath = merged_dir + chunkname
    outpath = merged_reduced_dir + 'reduced_' + chunkname
    print('Reading in chunk', merged_dir + chunkname)
    chunk = pd.read_hdf(merged_dir + chunkname)
    mIDs = excel['MetaVariableID'].unique()
    df_reduced = chunk.loc[:, ['PatientID', 'Datetime']]
    # now run
    for mID in mIDs:
        # mID gets changed in here, to add a pm or vm
        mID_column, mID = merge_parameter(mID, excel, chunk)
        df_reduced[mID] = mID_column
    print('Saving to', outpath)
    df_reduced.to_hdf(outpath, 'reduced', append=False, complevel=5,
            complib='blosc:lz4', data_columns=['PatientID'], format='table')
    # csv is a lot quicker...
    #df_reduced.to_csv(merged_reduced_csv, mode='a')

def main(idx):
    #excel = pd.read_excel(excel_path)
    excel_var = pd.read_csv(excel_path_var, sep='\t', encoding='cp1252')
    excel_lab = pd.read_csv(excel_path_lab, sep='\t', encoding='cp1252')
    excel = preproc_excel(excel_var, excel_lab)
    #mIDs_to_ignore = [58, 57, 67, 74, 68]      # v5
    mIDs_to_ignore = []                         # v6

    files = glob.glob(merged_dir + '*h5')
    files = [f.split('/')[-1] for f in files]

    #for i, f in enumerate(sorted(files)):
    #    print(i, f)
    process_chunk(sorted(files)[idx], excel)

#idx = int(sys.argv[1])
#main(idx)
