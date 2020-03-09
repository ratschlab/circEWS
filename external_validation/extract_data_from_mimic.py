#!/usr/bin/ipython
# author: stephanie hyland
# data extraction and preprocessing functions for mimic

import pandas as pd
import numpy as np
import ipdb
import glob
import sys

import mimic_paths

# --- references --- #
def load_reference_tables(gdoc=True):
    if gdoc:
        ref_path = mimic_paths.GDOC_path
        try:
            d_reference = pd.read_csv(ref_path)
        except:
            print('ERROR: issue loading reference from', ref_path)
            return False
        #d_reference = d_reference.loc[d_reference['include'] == True, :]
        d_reference = d_reference.loc[:, ['include', 'ITEM_ID', 'varname (mimic?)', 'mID', 'table', 'merge logic']]
        d_reference.rename({'varname (mimic?)': 'varname'}, axis=1, inplace=True)
        d_reference['mID'] = d_reference['mID'].astype('str')
    else:
        # older version
        d_items = pd.read_csv(mimic_paths.D_ITEMS_path)
        d_labitems = pd.read_csv(mimic_paths.D_LABITEMS_path)
        d_items = d_items.loc[:, ['ITEM_ID', 'varname', 'mID', 'table']]
        d_labitems = d_labitems.loc[:, ['ITEM_ID', 'varname', 'mID']]
        d_labitems['table'] = 'labevents'
        d_reference = pd.concat([d_items, d_labitems])
        d_reference['mID'] = d_reference['mID'].astype('str')
        # create the "vm"+ column
        d_reference['mID_corrected'] = list(map(lambda x: 'vm' + str(x), d_reference['mID']))
        d_reference.loc[d_reference['mID'] == 'exclusion', 'mID_corrected'] = 'exclusion'
        d_reference.loc[d_reference['mID'] == 'static', 'mID_corrected'] = 'static'
        d_reference.loc[d_reference['mID'] == 'nan', 'mID_corrected'] = 'nan'
        d_reference.loc[d_reference['varname'].isin({'Epinephrine', 'Dobutamine', 'Norepinephrine', 'Vasopressin'}), 'mID_corrected'] = list(map(lambda x: 'pm' + str(x), d_reference.loc[d_reference['varname'].isin({'Epinephrine', 'Dobutamine', 'Norepinephrine', 'Vasopressin'}), 'mID']))
        # tidy up
        d_reference['mID_orig'] = d_reference['mID']
        d_reference['mID'] = d_reference['mID_corrected']
        del d_reference['mID_corrected']
    d_reference.rename({'ITEM_ID': 'ITEMID'}, axis=1, inplace=True)
    return d_reference

def get_icustayid_lookup():
    """
    using the chartevents table, grab ICUSTAY_ID, SUBJECT_ID, HADM_ID
    """
    df = pd.read_csv(mimic_paths.source_data + 'ICUSTAYS.csv')
    df = df.loc[:, ['SUBJECT_ID', 'ICUSTAY_ID', 'HADM_ID']]
    df.drop_duplicates(inplace=True)
    return df

def load_excel():
    """ hack for now """
    print('[load_excel] WARNING: this functionality should be refactored')
    sys.path.append('../dimensionality_reduction')
    from perform_dimensionality_reduction import excel_path_var, excel_path_lab, preproc_excel
    excel_var = pd.read_csv(excel_path_var, sep='\t', encoding='cp1252')
    excel_lab = pd.read_csv(excel_path_lab, sep='\t', encoding='cp1252')
    excel = preproc_excel(excel_var, excel_lab)
    # add mv/pm to excel mIDs
    pharma_vars = excel['Type'] == 'Pharma'
    excel.loc[pharma_vars,  'MetaVariableID'] = list(map(lambda x: 'pm' + str(x), excel.loc[pharma_vars, 'MetaVariableID']))
    excel.loc[~pharma_vars,  'MetaVariableID'] = list(map(lambda x: 'vm' + str(x), excel.loc[~pharma_vars, 'MetaVariableID']))
    return excel

def merge_ref_and_excel(d_ref=None, excel=None):
    if d_ref is None:
        d_ref = load_reference_tables()
    if excel is None:
        excel = load_excel()
    # we will join on mID
    d_ref.set_index('mID', inplace=True)
    excel_sub = excel.loc[:, ['MetaVariableID', 'MetaVariableName', 'LowerBound', 'UpperBound']] 
    excel_sub.drop_duplicates(inplace=True)
    excel_sub.rename(columns={'MetaVariableID': 'mID'}, inplace=True)
    excel_sub.set_index('mID', inplace=True)
    d_ref = d_ref.join(excel_sub, how='left')
    return d_ref

# --- misc utility --- #
def prep_table(tablename):
    """
    """
    if tablename == 'chartevents':
        table = open(mimic_paths.chartevents_path, 'r')
        var_idx = 4
    elif tablename == 'labevents':
        table = open(mimic_paths.labevents_path, 'r')
        var_idx = 3
    elif tablename == 'inputevents_mv':
        table = open(mimic_paths.inputevents_mv_path, 'r')
        var_idx = 6
    elif tablename == 'inputevents_cv':
        table = open(mimic_paths.inputevents_cv_path, 'r')
        var_idx = 5
    elif tablename == 'outputevents':
        table = open(mimic_paths.outputevents_path, 'r')
        var_idx = 5
    elif tablename == 'datetimeevents':
        table = open(mimic_paths.datetimeevents_path, 'r')
        var_idx = 4
    elif tablename == 'procedureevents_mv':
        table = open(mimic_paths.procedureevents_mv_path, 'r')
        var_idx = 6
    else:
        raise ValueError(tablename)
    return var_idx, table

def fix_variable_names(which='csvs'):
    """
    POSTHOC fix!
    add the vm/pm to the variable names (should have been done to the ID list)
    """
    ref = load_reference_tables()
    mapping = ref.loc[:, ['mID_orig', 'mID']].set_index('mID_orig')
    mapping = mapping['mID'].to_dict()
    if which == 'csvs':
        for csv_in_path in glob.glob(mimic_paths.csvs_dir + '*.csv'):
            print(csv_in_path)
            csv_out_path = csv_in_path + '.fixed'
            fi = open(csv_in_path, 'r')
            header = fi.readline()
            fo = open(csv_out_path, 'w')
            fo.write(header)
            for line in fi:
                sl = line.strip('\n').split(',')
                mid = sl[-1]
                new_mid = mapping[mid]
                sl[-1] = new_mid
                new_line = ','.join(sl) + '\n'
                fo.write(new_line)
            fi.close()
            fo.close()
    elif which == 'merged':
        raise NotImplementedError
        df_merged = pd.read_hdf(mimic_paths.merged_dir + 'merged.h5')
        print('Current columns:', df_merged.columns)
        df_merged.rename(columns=mapping, inplace=True)
        print('New columns:', df_merged.columns)
        # there is an error here, but there is also an issue with the merged data...
        df_merged.to_hdf(mimic_paths.merged_dir + 'merged_fixed.h5', 'merged', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
    else:
        raise ValueError(which)
    return True
   
def collect_pre_merged(version=''):
    df_list = []
    d_reference = load_reference_tables()
    for table in d_reference['table'].unique():
        try:
            print('reading', table)
            df = pd.read_hdf(mimic_paths.hdf5_dir + version + '/' + table + '_subset.h5', columns=['ICUSTAY_ID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'mID', 'ITEMID'])
            df_list.append(df)
        except TypeError:
            print('skipping', table)
            continue
    df_merged = pd.concat(df_list)
    return df_merged

# --- subset tables --- #
def subset_table(tablename, varlist, exclusion_list, version):
    """
    Subset the table to variables in the list, and not patients on the exclusion list
    """
    try:
        var_idx, table = prep_table(tablename)
    except:
        return False
    # subject id is always the second field
    subj_id = 1
    out_path = mimic_paths.csvs_dir +  version + '/' +  tablename + '_subset.csv'
    print('Writing to', out_path)
    table_subset = open(out_path, 'w')
    header = table.readline().strip('\n')
    table_subset.write(header + ',"mID"\n')

    written_lines = 0
    for (i, line) in enumerate(table):
        sl = line.strip('\n').split(',')
        subject_id = sl[subj_id]
        if str(subject_id) in exclusion_list:
            # skip this patient
            continue
        item_id = int(sl[var_idx])
        if item_id in varlist:
            mID = varlist[item_id]
            # write this row to the new csv
            table_subset.write(line.strip('\n') + ',' + str(mID) + '\n')
            written_lines += 1
        if i % 50000 == 0:
            print(i, written_lines)
            table_subset.flush()
    table_subset.close()
    print('Wrote', written_lines, 'lines to', out_path)
    return True

def meta_subset_tables(skiptable=None, version='', justtable=None):
    """
    just go through all the variables, pick them out of the relevant tables, which will get written to CSV
    rotating etc. comes later
    ... wrapper around "subset_table"
    """
    if not justtable is None:
        assert skiptable is None
    d_reference = load_reference_tables()
    exclusion_list = load_exclusion_list(version)

    if skiptable is None:
        skiptable = []
    for table in d_reference['table'].unique():
        if table in skiptable:
            continue
        if not justtable is None:
            if not table in justtable:
                continue
        d_table = d_reference.loc[d_reference['table'] == table, :]
        # this makes a ITEM_ID: mID dictionary, which we need for 'subset_table'
        varlist = d_table.loc[:, ['ITEMID', 'mID']].set_index('ITEMID').to_dict()['mID']
        print('Extracting', len(varlist), 'variables from table', table)
        subset_table(table, varlist, exclusion_list, version=version)
    return True

# --- convert to hdf5  --- #
def trim_unify_csvs(skiptable=[np.nan], version=''):
    """
    go through csvs, restricting to 
    ICUSTAY_ID, CHARTTIME, STORETIME, VALUE, mID
    convert to hdf5
    """
    d_reference = load_reference_tables()
    columns_of_interest = ['ICUSTAY_ID', 'ITEMID', 'SUBJECT_ID', 'CHARTTIME', 'STORETIME', 'VALUE', 'VALUENUM', 'mID']
    if skiptable is None:
        skiptable = []
    for table in d_reference['table'].unique():
        if table in skiptable:
            continue
        print('Trimming table', table, 'and converting to hdf5')
        if table == 'inputevents_mv':
            df_sub = deal_with_drugs(version=version)
            df_sub.rename(columns={'RATE': 'VALUENUM'}, inplace=True)
            df_sub.reset_index(inplace=True)
        elif table == 'procedureevents_mv':
            df_sub = deal_with_procedures(version=version)
            df_sub.reset_index(inplace=True)
        else:
            df = pd.read_csv(mimic_paths.csvs_dir + version + '/' + table + '_subset.csv', low_memory=False)
            for col in columns_of_interest:
                if not col in df.columns:
                    print('WARNING: table', table, 'is missing', col)
                    if col == 'ICUSTAY_ID':
                        assert 'HADM_ID' in df.columns
                        assert 'SUBJECT_ID' in df.columns
                        print('Joining from lookup...')
                        mapping = get_icustayid_lookup()
                        df = pd.merge(df, mapping, how='left', on=['HADM_ID', 'SUBJECT_ID'])             
                    if col == 'VALUENUM':
                        assert 'VALUE' in df.columns
                        print('creating VALUENUM column from VALUE in', table)
                        df['VALUENUM'] = df['VALUE']
            df_sub = df.loc[:, columns_of_interest]
        output_path = mimic_paths.hdf5_dir + version + '/' + table + '_subset.h5'
        df_sub.to_hdf(output_path, 'pivoted', append=False, complevel=5, complib='blosc:lz4', data_columns=['ICUSTAY_ID'], format='table')
    return True

# --- merging --- #
def merge_mid(mid, mdf, mdf_path):
    """
    build it
    """
    d_ref = load_reference_tables()
    merge_logic = d_ref.loc[d_ref['mID'] == mid, 'merge logic'].iloc[0]
    mdf_merged = []
    pids = mdf['PatientID'].unique()
    for pid in pids:
        pdf = mdf.loc[mdf['PatientID'] == pid, :]
        if merge_logic == 'median':
            pdf_merged = pdf.loc[:, ['Datetime', 'VALUENUM']].groupby('Datetime').median()
        elif merge_logic == 'binary':
            # check if any value at this timepoint is greater than 0
            # this looks strange, but it will preserve NAN values for us
            pdf_merged = pdf.loc[:, ['Datetime', 'VALUENUM']].groupby('Datetime').apply(lambda x: 1 if x['VALUENUM'].sum(min_count=1) > 0 else 0 if x['VALUENUM'].sum(min_count=1) == 0 else np.nan)
            # the above returns a Series for some reason
            try:
                pdf_merged_df = pd.DataFrame(pdf_merged, columns=['VALUENUM'])
                pdf_merged = pdf_merged_df
            except:
                ipdb.set_trace()
        elif merge_logic == 'sum':
            pdf_merged = pdf.loc[:, ['Datetime', 'VALUENUM']].groupby('Datetime').sum(min_count=1)
        else:
            raise ValueError(merge_logic)
        pdf_merged.rename(columns={'VALUENUM': mid}, inplace=True)
        pdf_merged['PatientID'] = pid
        #pdf_merged['mID'] = mid
        pdf_merged.reset_index(drop=False, inplace=True)
        #df_all.append(pdf_merged)
        mdf_merged.append(pdf_merged)
    mdf_merged = pd.concat(mdf_merged)
    # just for good measure
    mdf_merged['Datetime'] = pd.to_datetime(mdf_merged['Datetime'])
    mdf_merged.to_hdf(mdf_path, 'merged', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
    return mdf_merged


def merge_duplicates_and_pivot(df=None, version=''):
    """
    identify the same variable with identical patientID, charttime, value - merge somehow
    """
    d_ref = load_reference_tables()
    # now get the remaining mids
    mids = d_ref.loc[d_ref['include'] == True, 'mID'].unique()
    mdf_all = []
    print('Collecting mids...')
    for i, mid in enumerate(mids):
        if mid in ['exclusion', 'static', np.nan]:
            print('mID', mid, 'in exclusion list skipping...')
            continue
        mdf_path = mimic_paths.merged_dir + str(mid) + '.h5'
        try:
            mdf_merged = pd.read_hdf(mdf_path)
        except FileNotFoundError:
            if df is None:
                print('loading data...')
                df = prep_pre_merged(version)
            mdf = df.loc[df['mID'] == mid, :]
            if mdf.shape[0] == 0:
                print('mID', mid, 'not found in the data - skipping')
                mdf_merged = None
            else:
                mdf_merged = merge_mid(mid, mdf, mdf_path)
        if not mdf_merged is None:
            print('merging mid', mid, '(', d_ref.loc[d_ref['mID'] == mid, 'varname'].values, ')')
            mdf_all.append(mdf_merged)
    print('Merging...')
    df_merged = mdf_all[0]
    df_merged['Datetime'] = pd.to_datetime(df_merged['Datetime'])
    for i, df in enumerate(mdf_all[1:]):
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            print(i)
            df_merged = pd.merge(df_merged, df, how='outer', on=['PatientID', 'Datetime'])
        except:
            print('ERROR!')
            ipdb.set_trace()
    return df_merged

def convert_to_rate(df, version):
    """
    Given a long-form dataframe containing ONE ITEMID and potentially MANY patients, convert to rate
    This will happen during pre_merged
    """
    assert len(df['ITEMID'].unique()) == 1
    pids = df['PatientID'].unique()
    # load the static table
    static = pd.read_hdf(mimic_paths.merged_dir + version + '/' + 'static.h5')
    pdf_list = []
    for pid in pids:
        pdf = df.loc[df['PatientID'] == pid, :]
        pdf['Datetime'] = pd.to_datetime(pdf['Datetime'])
        pdf.set_index('Datetime', inplace=True)
        pdf.sort_index(inplace=True)
        try:
            total_dose_before = pdf['VALUENUM'].sum()
            entry_time = pd.to_datetime(static.loc[static['PatientID'] == pid, 'ADMITTIME'].iloc[0])
            hours_elapsed = np.concatenate([[(pdf.index[0] - entry_time)/np.timedelta64(1, 'h')],
                (pdf.index[1:] - pdf.index[:-1])/np.timedelta64(1, 'h')])
            amount_accumulated = pdf['VALUENUM']
            hourly_rate = amount_accumulated/hours_elapsed
            total_dose_after = hourly_rate.sum()
        except:
            ipdb.set_trace()
#        if not total_dose_before == total_dose_after:
#            ipdb.set_trace()
        pdf['RATE'] = hourly_rate
        pdf.reset_index('Datetime', inplace=True)
        pdf_list.append(pdf)
    df_out = pd.concat(pdf_list)
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df_out = df.merge(df_out, on=['PatientID', 'Datetime', 'mID', 'ITEMID'])
        df_out['VALUENUM'] = df_out['RATE'].copy()
        del df_out['RATE']
        assert df_out.shape[0] == df.shape[0]
        assert df.loc[:, ['PatientID', 'Datetime']].equals(df_out.loc[:, ['PatientID', 'Datetime']])
#        assert not df['VALUE'].equals(df_out['VALUE'])
    except AssertionError:
        ipdb.set_trace()
    #    df_out.loc[df_out['PatientID'] == pid, 'VALUE'] = hourly_rate  
    return df_out['VALUENUM']

def deal_with_special_cases(df, version):
    # non-numerical values... ... cases...
#    df.loc[df['VALUE'] == 'LESS THAN 10', 'VALUE'] = 9
    # RASS
#    # extract the numbers
#    df.loc[df['ITEMID'] == 228096, 'VALUE'] = pd.to_numeric(df.loc[df['ITEMID'] == 228096, 'VALUE'].str[:2])
    # now coerce everything to numeric
    na_pre = df['VALUENUM'].isna().sum()
    #values = df['VALUE'].astype('float')
    values = pd.to_numeric(df['VALUENUM'], errors='coerce')
    na_post = values.isna().sum()
    if na_post > na_pre:
        print('coerced', na_post - na_pre, '(', 100*(na_post - na_pre)/values.shape[0], '%) values to NA due to numeric conversion')
    df['VALUENUM'] = values
    # urine --> convert to rate
    df.loc[df['ITEMID'] == 227489, 'VALUENUM'] = convert_to_rate(df.loc[df['ITEMID'] == 227489, :], version)
    # temperature (F --> C)
    df.loc[df['ITEMID'] == 223761, 'VALUENUM'] = (df.loc[df['ITEMID'] == 223761, 'VALUENUM'] - 32)*(5.0/9.0)
    # WEIGHT - convert lbs to kg
    df.loc[df['ITEMID'] == 226531, 'VALUENUM'] = 0.453592*df.loc[df['ITEMID'] == 226531, 'VALUENUM']
    # glucose - convert from mg/dl to mmol/l
    glucose_vars = [225664, 220621, 226537, 50809, 50931]
    for gv in glucose_vars:
        df.loc[df['ITEMID'] == gv, 'VALUENUM'] = (1.0/18)*df.loc[df['ITEMID'] == gv, 'VALUENUM']
    # Hb - convert from mg/dl to mg/L
    hb_vars = [50811, 51222, 220228]
    for hv in hb_vars:
        df.loc[df['ITEMID'] == hv, 'VALUENUM'] = 10*df.loc[df['ITEMID'] == hv, 'VALUENUM']
    # creatinine
    creat_vars = [50912, 220615]
    for cv in creat_vars:
        df.loc[df['ITEMID'] == cv, 'VALUENUM'] = 88.42*df.loc[df['ITEMID'] == cv, 'VALUENUM']
    return df

def prep_pre_merged(version):
    df_merged = collect_pre_merged(version=version)
    print('WARNING: converting ICUstays to patients')
    df_merged.rename(columns={'ICUSTAY_ID': 'PatientID'}, inplace=True)
    # need to propagate this...
    df_merged.rename(columns={'CHARTTIME': 'Datetime'}, inplace=True)
    df_merged.dropna(how='all', inplace=True)
    df_merged.drop_duplicates(inplace=True)
    # deal with special cases
    df_merged = deal_with_special_cases(df_merged, version)
    ## remove invalid values
    d_ref = load_reference_tables()
    invalid_itemid = d_ref.loc[d_ref['include'] == False, 'ITEMID'].unique()
    # exclude invalid itemids
    df_merged = df_merged.loc[~df_merged['ITEMID'].isin(invalid_itemid), :]
    return df_merged

def merge_tables(version=''):
    """
    given hdf5s, pivot and merge them at the same time
    """
    df_merged = merge_duplicates_and_pivot(df=None, version=version)
    # Fluid balance OUT == hourly urine
    df_merged['vm32'] = df_merged['vm24']
    output_path = mimic_paths.merged_dir + version + '/reduced/merged.h5'
    df_merged.to_hdf(output_path, 'merged', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
    print('Wrote dataframe of shape', df_merged.shape, 'to', output_path)
    return True

# --- cleaning --- #
def remove_impossible_values(version='180817', step='merged'):
    """
    from the merged data
    """
    if not step == 'merged':
        raise NotImplementedError(step)
    # prepare the reference
    d_ref = merge_ref_and_excel()
    mids = d_ref.index.unique()

    df = pd.read_hdf(mimic_paths.merged_dir + version + '/reduced/merged.h5')
    df_clean = df.copy()
    for mid in mids:
        if not mid in df.columns:
            continue
        if mid in ['exclusion', 'static', 'nan']:
            continue
        lower = d_ref.loc[mid, 'LowerBound']
        upper = d_ref.loc[mid, 'UpperBound']
        try:
            lower = lower.median()
            upper = upper.median()
        except AttributeError:
            pass
        print('restricting', mid, 'to between [', lower, upper, ']')
        values = df[mid].copy()
        # just set impossible values to nan
        if np.isfinite(lower):
            values.loc[values < lower] = np.nan
        if np.isfinite(upper):
            values.loc[values > upper] = np.nan
        df_clean[mid] = values
        try:
            if np.isnan(lower) and np.isnan(upper):
                assert df_clean[mid].equals(df[mid])
            if np.isfinite(lower):
                assert (df_clean[mid].min() >= lower) or (df_clean[mid].isnull().mean() == 1)
            if np.isfinite(upper):
                assert (df_clean[mid].max() <= upper) or (df_clean[mid].isnull().mean() == 1)
        except AssertionError:
            ipdb.set_trace()
    df_clean.to_hdf(mimic_paths.merged_dir + version + '/reduced/merged_clean.h5',
            'merged_clean', append=False, complevel=5, complib='blosc:lz4',
            data_columns=['PatientID'], format='table')
    return True

# --- pharma --- #
def find_overlapping_drugs(df):
    """
    check to see if the same drug is ever given twice at once
    """
    df.sort_values(['ICUSTAY_ID', 'CHARTTIME'], inplace=True)
    for drugid in df['ITEMID'].unique():
        df_sub = df.loc[df['ITEMID'] == drugid, :]
        ipdb.set_trace()
        violations = df_sub.groupby('ICUSTAY_ID').apply(lambda x: (x['ENDTIME'].max() - x['STARTTIME'].min()) >= (x['ENDTIME'] - x['STARTTIME']).sum())
        if sum(violations) > 0:
            ipdb.set_trace()
            #TODO: these violations DEFINITELY happen, so we need to handle them, unfortunately
    return True

def deal_with_drugs(version):
    """
    The drugs are written like, STARTTIME, ENDTIME, with a rate in the middle...
    So we double it, make START = END, set the rate at the end to negative rate, concatenate it back, calculate cumulative sums...
    """
    df = pd.read_csv(mimic_paths.csvs_dir + version + '/inputevents_mv_subset.csv')
    df['STARTTIME'] = pd.to_datetime(df['STARTTIME'])
    df['ENDTIME'] = pd.to_datetime(df['ENDTIME'])
#    find_overlapping_drugs(df)
   
    df_ends = df.copy()
    df_ends['CHARTTIME'] = df_ends['ENDTIME']
    df_ends['RATE'] = -df_ends['RATE']

    df['CHARTTIME'] = df['STARTTIME']
    df = pd.concat([df, df_ends])
    df.sort_values(['ICUSTAY_ID', 'CHARTTIME'], inplace=True)
    # for debug purposes
    df_orig = df.copy()
    df['INST_RATE'] = 0
    # now split up per patient (not so efficient but w/e)
    pdf_list = []
    for icustayid in df['ICUSTAY_ID'].unique():
        pdf = df.loc[df['ICUSTAY_ID'] == icustayid, :]
        mid_list = []
        for mid in pdf['mID'].unique():
            inst_rate = pdf.loc[pdf['mID'] == mid, ['CHARTTIME', 'RATE']].groupby('CHARTTIME').sum().cumsum()
            inst_rate['ICUSTAY_ID'] = icustayid
            inst_rate['SUBJECT_ID'] = pdf['SUBJECT_ID'].iloc[0]
            inst_rate['HADM_ID'] = pdf['HADM_ID'].iloc[0]
            inst_rate['ITEMID'] = pdf['ITEMID'].iloc[0]
            inst_rate['mID'] = mid
            if inst_rate['RATE'].iloc[-1] > 1e-10:
                ipdb.set_trace()
            mid_list.append(inst_rate)
        if len(mid_list) == 0:
            # nothing to be seen here
            print('WARNING: no data on patient with icustayid', icustayid)
            continue
        else:
            pdf = pd.concat(mid_list)
            pdf_list.append(pdf)
    df = pd.concat(pdf_list)
    return df

# --- other strange cases ---#
def deal_with_procedures(version):
    """
    Procedures, like drugs, are written as STARTTIME, ENDTIME.
    Instead of computing rates, we just set binary flags when the condition is true.
    """
    df = pd.read_csv(mimic_paths.csvs_dir + version + '/procedureevents_mv_subset.csv')
    df['STARTTIME'] = pd.to_datetime(df['STARTTIME'])
    df['ENDTIME'] = pd.to_datetime(df['ENDTIME'])
    df['VALUENUM'] = 1
#    find_overlapping_drugs(df)
   
    df_ends = df.copy()
    df_ends['CHARTTIME'] = df_ends['ENDTIME']
    df_ends['VALUENUM'] = -1

    df['CHARTTIME'] = df['STARTTIME']
    df = pd.concat([df, df_ends])
    df.sort_values(['ICUSTAY_ID', 'CHARTTIME'], inplace=True)
    # now split up per patient (not so efficient but w/e)
    pdf_list = []
    
    for icustayid in df['ICUSTAY_ID'].unique():
        pdf = df.loc[df['ICUSTAY_ID'] == icustayid, :]
        mid_list = []
        for mid in pdf['mID'].unique():
            inst_value = pdf.loc[pdf['mID'] == mid, ['CHARTTIME', 'VALUENUM']].groupby('CHARTTIME').sum().cumsum()
            inst_value['ICUSTAY_ID'] = icustayid
            inst_value['SUBJECT_ID'] = pdf['SUBJECT_ID'].iloc[0]
            inst_value['HADM_ID'] = pdf['HADM_ID'].iloc[0]
            inst_value['ITEMID'] = pdf['ITEMID'].iloc[0]
            inst_value['mID'] = mid
            mid_list.append(inst_value)
        if len(mid_list) == 0:
            # nothing to be seen here
            print('WARNING: no data on patient with icustayid', icustayid)
            continue
        else:
            pdf = pd.concat(mid_list)
            pdf_list.append(pdf)
    df = pd.concat(pdf_list)
    return df 

# --- static --- #
def build_static_table(version=''):
    """
    extract the static variables, identify where they are, build a table
    """
    # load relevant tables, prep
    admissions_table = pd.read_csv(mimic_paths.admissions_path)
    patients_table = pd.read_csv(mimic_paths.patients_path)
    services_table = pd.read_csv(mimic_paths.services_path)

    patients_table.set_index('SUBJECT_ID', inplace=True)
    # keep just the first service the patient was admitted under (there can be transfers...)
    services_table = services_table.loc[services_table.sort_values('TRANSFERTIME')['HADM_ID'].drop_duplicates(keep='first').index, :]
    services_table.set_index('HADM_ID', inplace=True)
    
    # prep
    table = admissions_table.join(patients_table, on=['SUBJECT_ID'], rsuffix='r', how='left')
    table = table.join(services_table.loc[:, ['CURR_SERVICE']], on=['HADM_ID'], rsuffix='r', how='left')
    # age 
    table['Age'] = (pd.to_datetime(table['ADMITTIME']) - pd.to_datetime(table['DOB']))/np.timedelta64(24*365, 'h')
    # sex
    table.rename(columns={'GENDER': 'Sex'}, inplace=True)
    # emergency status
    table['Emergency'] = (table['ADMISSION_TYPE'] == 'EMERGENCY').astype('int')
    # surgical status
    table['Surgical'] = table['CURR_SERVICE'].str.contains('SURG').astype(bool, errors='ignore') 
    
    # height - just a static variable, but it's found in the chartevents table
    # NOTE: for the purpose of height, we just use this oldish version of chartevents . . .
    height_ID = 226730
    chartevents = pd.read_hdf(mimic_paths.hdf5_dir + '/181002/chartevents_subset.h5', columns=['SUBJECT_ID', 'CHARTTIME', 'VALUE', 'mID', 'ITEMID'])
    height = chartevents.loc[chartevents['ITEMID'] == height_ID, ['SUBJECT_ID', 'VALUE', 'CHARTTIME']]
    # sort by record time
    height.sort_values(by=['SUBJECT_ID', 'CHARTTIME'], inplace=True)
    height.drop('CHARTTIME', axis=1, inplace=True)
    # just keep the first height measurement
    height['VALUE'] = pd.to_numeric(height['VALUE'])
    height = height.loc[height.drop('VALUE', axis=1).drop_duplicates(keep='first').index, :]
    height.set_index('SUBJECT_ID', inplace=True)
    height.rename(columns={'VALUE': 'Height'}, inplace=True)
    table = table.join(height, on=['SUBJECT_ID'], rsuffix='r', how='outer')
    
    # now merge the icustay
    icustay_lookup = get_icustayid_lookup()
    icustay_lookup.set_index('ICUSTAY_ID', inplace=True)
    table = icustay_lookup.loc[:, ['HADM_ID']].join(table.set_index('HADM_ID'), on=['HADM_ID'], rsuffix='r', how='left')
   
    # we can keep the other columns for the craic
    static = table.copy()
    static.reset_index(inplace=True)
    static.rename(columns={'ICUSTAY_ID': 'PatientID'}, inplace=True)

    # deal with errors
    static.loc[static['Age'] <= 0, 'Age'] = np.nan
    static.loc[static['Height'] <= 0, 'Height'] = np.nan

    static_path = mimic_paths.merged_dir + version + '/static.h5'
    print('Saving static table to', static_path, 'shape:', static.shape)
    static.to_hdf(static_path, 'static', append=False, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
    return True

# --- exclusion list --- #
def load_exclusion_list(version=''):
    """
    """
    exclusion_list_path = mimic_paths.validation_dir + 'exclusion_list'
    # --- based on age
    try:
        exclusion_list_age = pd.read_csv(exclusion_list_path + '_age.csv')['SUBJECT_ID'].values
    except FileNotFoundError:
        exclusion_list_age = build_exclusion_list_based_on_age(version)
        exclusion_ser = pd.Series(exclusion_list_age, name='SUBJECT_ID')
        exclusion_ser.to_csv(exclusion_list_path + '_age.csv', index=False, header=True)
    # --- based on variables 
    try:
        exclusion_list_variables = pd.read_csv(exclusion_list_path + '_variables.csv')['SUBJECT_ID'].values
    except FileNotFoundError:
        # build it
        exclusion_list_variables = build_exclusion_list_based_on_variables()
        exclusion_ser = pd.Series(exclusion_list_variables, name='SUBJECT_ID')
        exclusion_ser.to_csv(exclusion_list_path + '_variables.csv', index=False, header=True)
    exclusion_list = np.concatenate([exclusion_list_age, exclusion_list_variables])
    exclusion_list = set(exclusion_list)
    return exclusion_list

def build_exclusion_list_based_on_age(version):
    """
    find patients with excessive age
    """
    static_info = pd.read_hdf(mimic_paths.merged_dir  + version + '/' + 'static.h5')
    bad_stays = static_info.loc[(static_info['Age'] < 16) | (static_info['Age'] > 100), ['HADM_ID', 'SUBJECT_ID']]
    # for now we just exclude based on subject_id
    exclusion_list = bad_stays['SUBJECT_ID'].unique()
    return exclusion_list

def build_exclusion_list_based_on_variables():
    """
    find patients with measurements of exclusion variables,
    make a list of them

    This function is slow! Low-hanging fruit for optimisation
    """
    d_reference = load_reference_tables()
    exclusion_ref = d_reference.loc[d_reference['mID'] == 'exclusion', :]
    tables_to_check = exclusion_ref['table'].unique()
    exclusion_vars = exclusion_ref['ITEMID'].unique()

    exclusion_list = set()
    for table in tables_to_check:
        print(table)
        try:
            var_idx, table_file = prep_table(table)
        except:
            continue
        subj_id = 1
        header = table_file.readline()
        for (i, line) in enumerate(table_file):
            sl = line.strip('\n').split(',')
            subject_id = int(sl[subj_id])
            item_id = int(sl[var_idx])
            if item_id in exclusion_vars:
                exclusion_list.add(subject_id)
            if i % 50000 == 0:
                print(i, len(exclusion_list))
    exclusion_list = list(exclusion_list)
    return exclusion_list
