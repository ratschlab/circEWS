import numpy as np
import pandas as pd

from os.path import join, exists
from time import clock

datapath = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'
data_version = '180822'

def get_datapath():
    return datapath

def get_table_names(tbl_type='all'):
    tbl_list = ['monvals', 'comprvals', 'dervals', 'observrec', 'labres', 'pharmarec']
    if tbl_type == 'all':
        return tbl_list 
    elif tbl_type == 'pharma':
        return ['pharmarec']
    elif tbl_type == 'non-pharma':
        return tbl_list[:-1]

def get_status_str(statusID, tbl_name):
    '''
    Get the meaning of a status
    '''
    status_str = []
    if tbl_name == 'pharmarec':
        statusID_bin_str = '{0:10b}'.format(statusID)[::-1]
        if statusID_bin_str[1] == '1':
            status_str.append('invalidated')
        if statusID_bin_str[2] == '1':
            status_str.append('start')
        if statusID_bin_str[3] == '1':
            status_str.append('caused by event')
        if statusID_bin_str[5] == '1':
            status_str.append('notified')
        if statusID_bin_str[8] == '1':
            status_str.append('stop')
        if statusID_bin_str[9] == '1':
            status_str.append('included in record reports')
    else:
        statusID_bin_str = '{0:11b}'.format(statusID)[::-1]
        if statusID_bin_str[0] == '1':
            status_str.append('out of range')
        if statusID_bin_str[1] == '1':
            status_str.append('invalidated')
        if statusID_bin_str[2] == '1':
            status_str.append('first of connection')
        if statusID_bin_str[3] == '1':
            status_str.append('caused by event')
        if statusID_bin_str[4] == '1':
            status_str.append('compressed')
        if statusID_bin_str[5] == '1':
            status_str.append('notified but not measured')
        if statusID_bin_str[6] == '1':
            status_str.append('bigger than')
        if statusID_bin_str[7] == '1':
            status_str.append('smaller than')
        if statusID_bin_str[10] == '1':
            status_str.append('mandatory')
            
    return ', '.join(status_str)

def voi_id_name_mapping(tbl_name, replace_name=False, include_all=False, use_ref='excel', version='v6'):
    """
    Load the mapping between variableID and variableName
    
    Parameters:
    tbl_name: the name of the table (string)
    replace_name: bool; if True, the variableNames are the medical terms; otherwise, the varialbeName
    are the 'v%s'%variableIDs.
    include_all: bool; if True, all the variables in the table are variables of interest; otherwise,
    only variables can be found in the ref_excel files are the variables of interest.

    Returns:
    voi: dataframe; include the mapping between variableIDs and variableNames of the variables of
    interest
    """
    if use_ref not in ['excel', 'expot']:
        raise Exception('use_ref can only be "excel" or "expot"')
    id_list_path = join(datapath, 'misc_derived', 'id_lists')
    excelref_path = join(datapath, 'misc_derived', 'ref_excel')
    expotref_path = join(datapath, '0_csv_exports')
    vid_set = pd.read_csv(join(id_list_path, 'vID_%s.csv'%tbl_name), sep=',',
                          dtype={'VariableID': int}).VariableID.unique()

    # The head of the variable name is 'v' is the variable is non-pharma variable;
    # otherwise the variable name head is 'p' 
    vname_head = 'p' if tbl_name == 'pharmarec' else 'v' 
    if include_all and not replace_name:
        ref_voi = pd.DataFrame([[x, '%s%d'%(vname_head, x)] for x in vid_set], 
                               columns=['VariableID', 'VariableName'])
    else:
        excelref_filename = 'labref_excel_%s.tsv'%version if tbl_name == 'labres' else 'varref_excel_%s.tsv'%version

        excelref = pd.read_csv(join(excelref_path, excelref_filename), sep='\t', encoding='cp1252')
        
        # Avoid repeat VariableIDs in 'pharmarec' and other tables
        if tbl_name == 'pharmarec':
            excelref = excelref[excelref.Type=='Pharma']
        elif tbl_name != 'labres':
            excelref = excelref[excelref.Type!='Pharma']
        try:
            excelref['VariableID'] = excelref['VariableID'].astype(int)
        except:
            excelref['VariableID'] = excelref.VariableID.apply(lambda x: float('NaN') if x=='???' else float(x))

        if not include_all and not replace_name:
            vid_voi = set(excelref.VariableID) & set(vid_set)
            ref_voi = pd.DataFrame([[x, '%s%d'%(vname_head, x)] for x in vid_voi], 
                                   columns=['VariableID', 'VariableName'])
        else:
            if tbl_name == 'pharmarec':
                expotref = pd.read_csv(join(expotref_path, 'expot-pharmaref.csv'), sep='\t', encoding='cp1252',
                                       usecols=['PharmaID', 'PharmaName']).drop_duplicates('PharmaID', keep='last')
                expotref.rename(columns={'PharmaID': 'VariableID', 'PharmaName': 'VariableName'}, inplace=True)
            else:
                expotref = pd.read_csv(join(expotref_path, 'expot-varref.csv'), sep='\t', encoding='cp1252',
                                       usecols=['VariableID', 'Abbreviation']).drop_duplicates('VariableID', keep='last')
                expotref.rename(columns={'Abbreviation': 'VariableName'}, inplace=True)

            if include_all:
                ref_voi = expotref[expotref.VariableID.isin(vid_set)].copy()
                if use_ref == 'excel':
                    print('Could not use the excel reference table when include_all=True, because the excel reference table is incomplete.')
            else:            
                vid_voi = set(excelref.VariableID) & set(vid_set)
                if use_ref == 'excel':
                    ref_voi = excelref[excelref.VariableID.isin(vid_voi)].copy()
                else:
                    ref_voi = expotref[excelref.VariableID.isin(vid_voi)].copy()
            
    ref_voi.VariableID = ref_voi.VariableID.astype(int)
    ref_voi.VariableName = ref_voi.VariableName.astype(str)
    ref_voi.set_index('VariableID', inplace=True)
    return ref_voi

def get_consent_patient_ids(recompute=False, replace=False):
    """
    Return the list of patientID of patients with consent, i.e. those whose GeneralConsent 
    value smaller than 4 or not recorded in the observrec table, and not in the 
    PID_Exclusion_GeneralConsent (sent by Martin) list
    """
    id_list_path = join(datapath, 'misc_derived', 'id_lists')
    filepath = join(id_list_path, 'PID_WithConsent_not_on_ECMO.csv')
    if recompute:
        df_gc = pd.read_csv(join(datapath, '0_csv_exports', 'expot-observrec.csv'), sep=';',
                            usecols=['PatientID', 'VariableID', 'Value'])
        pid_exclude = set(pd.read_csv(join(id_list_path, 'PID_Exclusion_GeneralConsent.csv')).PatientID) | set(pd.read_csv(join(id_list_path, 'PID_on_ECMO.csv')).PatientID)
        df_gc = df_gc[df_gc.VariableID == 15004651] # extract time series of GeneralConsent
        df_gc.loc[:,'Value'] = df_gc.Value.astype(float)
        pid_noconsent = set(df_gc[df_gc.Value >= 4].PatientID)
        pid_gd = set(pd.read_csv(join(datapath, '0_csv_exports', 'expot-generaldata.csv'), sep=';',
                                 usecols=['PatientID']).PatientID)
        pid_any_but_gd = set()
        for tbl in get_table_names():
            pid_tbl_filepath = join(id_list_path, 'PID_%s.csv'%tbl)
            if not exists(pid_tbl_filepath):
                import ipdb
                ipdb.set_trace()
                if tbl in ['monvals', 'comprvals']:
                    csv_iters = pd.read_csv(join(datapath, '0_csv_exports', 'expot-%s.csv'%tbl), sep=';',
                                            usecols=['PatientID'], chunksize=10**7)
                    pid_tbl = set()
                    for i, chunk in enumerate(csv_iters):
                        pid_tbl = pid_tbl | set(chunk.PatientID)
                else:
                    df_tmp = pd.read_csv(join(datapath, '0_csv_exports', 'expot-%s.csv'%tbl),
                                         usecols=['PatientID'], sep=';')
                    pid_tbl = set(df_tmp.PatientID)
                df_pid_tbl = pd.DataFrame(list(pid_tbl), columns=['PatientID'])
                df_pid_tbl.to_csv(pid_tbl_filepath, index=False)
            else:
                pid_tbl = set(pd.read_csv(pid_tbl_filepath).PatientID)
            print('# patients in %s: %d'%(tbl, len(pid_tbl)))
            pid_any_but_gd = pid_any_but_gd | pid_tbl
        pid_any_and_gd = pid_any_but_gd & pid_gd 
        pid_gd_include = pid_gd - (pid_exclude | pid_noconsent)
        pid_any_but_gd_include = pid_any_but_gd - (pid_exclude | pid_noconsent)
        pid_include = list((pid_any_but_gd | pid_gd) - (pid_gd - pid_any_but_gd) - (pid_exclude | pid_noconsent))

        print('# patients in any measurement tables but generaldata: %d'%(len(pid_any_but_gd)))
        print('# patients in generaldata: %d'%(len(pid_gd)))
        print('# patients in any measurement tables but also in generaldata: %d'%(len(pid_any_and_gd)))
        print('# patients in generaldata but have no measurement: %d'%(len(pid_gd - pid_any_but_gd)))
        print('# patients have measurement but not in generaldata: %d'%(len(pid_any_but_gd - pid_gd)))
        print('# patients in "PID_Exclusion_GeneralConsent.csv": %d'%(len(pid_exclude)))
        print('# patients in "PID_Exclusion_GeneralConsent.csv" but not in general data: %d'%(len(pid_exclude - pid_gd)))
        print('# patients whose GeneralConsent value >= 4: %d'%(len(pid_noconsent)))
        print('# patients whose GeneralConsent >=4 and are also in "PID_Exclusion_GeneralConsent.csv": %d'%len(pid_exclude & pid_noconsent))
        print('# patients in generaldata who give consent: %d'%(len(pid_gd_include)))
        print('# patients in any measurement table but general who give consent: %d'%(len(pid_any_but_gd_include)))
        print('# patients who give consent and have measurement data: %d'%(len(pid_include)))

        df_pid_include = pd.DataFrame(pid_include, columns=['PatientID'])
        if replace:
            df_pid_include.to_csv(filepath, index=False)
    else:
        pid_include = pd.read_csv(filepath, sep=',').PatientID.unique()
    pid_include = np.sort(pid_include)
    return pid_include

def get_filtered_patient_ids(version='v6b'):
    return np.sort(pd.read_csv(join(datapath, 'misc_derived', 'id_lists', version, 'patients_in_datetime_fixed.csv')).PatientID.values)

def get_chunking_info(version='v6b', num_chunks=50, rewrite=False):
    step = 'clean'
    f_chunking = join(datapath, 'misc_derived', 'id_lists', version, 'patients_in_%s_chunking_%d.csv'%(step, num_chunks))
    if exists(f_chunking) and not rewrite:
        df_chunking = pd.read_csv(f_chunking)
    else:
        pid_list = get_filtered_patient_ids(version)
        chunksize = int(np.ceil( len(pid_list) / num_chunks))
        df_chunking = []
        for i in range(num_chunks):
            idx_start = i*chunksize
            idx_stop = min((i+1)*chunksize, len(pid_list))
            pid_list_tmp = pid_list[idx_start:idx_stop]
            df_chunking.extend([[pid, i] for pid in pid_list_tmp])
        df_chunking = pd.DataFrame(np.array(df_chunking), columns=['PatientID', 'ChunkfileIndex'])
        df_chunking.to_csv(f_chunking, index=False)

    df_chunking.set_index('PatientID', inplace=True)
    return df_chunking
        
def get_subset_patient_ids():
    """
    Return the list of patientID of 5% of the patients with consent
    """
    id_list_path = join(datapath, 'misc_derived', 'id_lists')
    filepath = join(id_list_path, 'PID_Subset.csv')
    if not exists(filepath):
        GetValidPidList()
        np.random.seed(0)
        tmp = np.random.rand(len(pID_set))
        pID_subset = pID_set[tmp <= .05] # choose only 5% of the patient
        df_pID_subset = pd.DataFrame(pID_subset, columns=['PatientID'])
        df_pID_subset.to_csv(filepath, index=False)
    else:
        pID_subset = pd.read_csv(filepath).PatientID.unique()
    return pID_subset

def time_difference(t_early, t_later):
    """
    Compute the time difference between t_early and t_later

    Parameters:
    t_early: np.datetime64, list or pandas series.
    t_later: np.datetime64, list or pandas series.
    """
    if type(t_early) == list:
        t1 = np.array(t_early)
    elif type(t_early) == pd.Series:
        t1 = np.array(t_early.tolist())
    else:
        t1 = np.array([t_early])

    if type(t_later) == list:
        t2 = np.array(t_later)
    elif type(t_later) == pd.Series:
        t2 = np.array(t_later.tolist())
    else:
        t2 = np.array([t_later])

    timedelta2float = np.vectorize(lambda x: x / np.timedelta64(3600, 's'))
    t_diff = timedelta2float(t2 - t1)
    return t_diff

def get_all_var_names():
    tbl_list = ['monvals', 'comprvals', 'dervals', 'observrec', 'pharmarec', 'labres']
    vname_list = []
    for tbl in tbl_list:
        ref_voi = voi_id_name_mapping(tbl)
        vname_list.append(ref_voi.VariableName.unique())
    vname_list = np.unique(np.concatenate(tuple(vname_list)))
    return vname_list

def read_single_patient_from_merged(patient_id, vname_list=None, verbose=False):
    readpath = join(datapath, '3_merged', 'fmat_170327')
    t = clock()
    if vname_list is None:
        vname_list = get_all_var_names()
    filename = 'p%s.h5'%patient_id
    filepath = join(readpath, filename)
    df = [pd.read_hdf(filepath, 'Datetime')]
    for var_name in vname_list:
        df.append(pd.read_hdf(filepath, var_name))
    if verbose:
        print('Time to read patient_id=%d: %g sec'%(patient_id, (clock()-t)))
        t = clock()
    df = pd.concat(df, axis=1, join_axes=[df[0].index])
    if verbose:
        print('Time to concate patient_id=%d: %g sec'%(patient_id, (clock()-t)))
    return df
  
def generate_id2string():
    tbl_list = ['monvals', 'comprvals', 'dervals', 'observrec', 'pharmarec', 'labres']
    ref_list = []
    for tbl in tbl_list:
        ref = voi_id_name_mapping(tbl)
        ref_string = voi_id_name_mapping(tbl, replace_name=True)
        ref_string.columns = ['string']
        ref['string'] = ref_string['string']
        ref_list.append(ref)
    ref = pd.concat(ref_list)
    ref.reset_index(drop=True, inplace=True)
    ref.columns = ['id', 'string']
    ref.drop_duplicates(inplace=True)
    ref.to_csv('id2string.csv', index=False)
    return ref

