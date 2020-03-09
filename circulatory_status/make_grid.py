#!/usr/bin/env ipython
# author: stephanie hyland
# operating on data from 3_merged, draw in endpoint-relevant variables and construct 5-min grid for all patients

import pandas as pd
import numpy as np
import ipdb
import os.path
import glob
import sys

from interpolate_lactate import interpolate_patient_lactate
from find_endpoints import find_endpoints


idx = int(sys.argv[1])
reduced = (sys.argv[2] == 'reduced')
if len(sys.argv) < 4:
    mimic = False
else:
    mimic = (sys.argv[3] == 'mimic')
    print('mimic:', mimic)
if mimic:
    sys.path.append('../external_validation/')
    import mimic_paths as paths
    # mimic is only reduced
    assert reduced
    grid_size_in_minutes = 60
    window_size_in_minutes = 60
else:
    import paths
    grid_size_in_minutes = 5
    window_size_in_minutes = 45

print('Grid size:', grid_size_in_minutes)
print('Window size:', window_size_in_minutes)

#in_version = '180214'
#in_version = '180619'
#in_version = '180817'
if mimic:
    #in_version = '180822'
    #in_version = '180926'
    #in_version = '181003'
    in_version = '181023'
    out_version = '181103'
else:
    in_version = 'v6b'
    out_version = in_version
#in_version = '180704'
#out_version = '180426'
#out_version = '180503'
#out_version = '180822'

# build paths

in_dir = paths.merged_dir + in_version + '/' + reduced*'reduced/'
out_dir = paths.endpoints_dir + out_version + '/' + reduced*'reduced/'
#observ_dir = paths.pivoted_dir + in_version + '/observrec/'

bad_patients = open(out_dir + 'bad_patients.csv', 'a')

print(idx)
print(in_dir)
print(out_dir)

if reduced:
    HR_ID = ['vm1']
    MAP_ID = ['vm5']
    #lactate_IDs = ['vm137', 'vm167']       # v5
    lactate_IDs = ['vm136', 'vm146']        # v6
    dobutamine_ID = ['pm41']
    milrinone_ID = ['pm42']
    levosimendan_ID = ['pm43']
    theophyllin_IDs = ['pm44']
    norepinephrine_IDs = ['pm39']
    epinephrine_IDs = ['pm40']
    vasopressin_IDs = ['pm45']
    weight_ID = ['vm131']                   # v6
else:
    HR_ID = ['v200']
    MAP_ID = ['v110']
    lactate_IDs = ["v24000524", "v24000732", "v24000485"]                   #v5, v6
    dobutamine_ID = ['p426']                                                #v5, v6
    milrinone_ID = ['p1000441']                                             #v5, v6
    levosimendan_ID = ['p1000606']                                          # v5, v6
    theophyllin_IDs = ['p1000706', 'p1000707', 'p1000698', '1000267']       #v6
#    theophyllin_IDs = ['p1000706', 'p1000707', 'p1000513']      # v5
    norepinephrine_IDs = ["p1000462", "p1000656", "p1000657", "p1000658"]   #v5, v6
    epinephrine_IDs = ["p71", "p1000750", "p1000649", "p1000650", "p1000655"]   #v5, v6
    vasopressin_IDs = ["p112", "p113"]                                          #v5, v6
    weight_ID = ["v10000400"]
#    bolus_or_tablet_IDs = {"p1000658", "p1000750", "p1000655", "p1000707", "p1000513"}

if mimic:
    # mimic is missing some variables
    print('WARNING: WE ASSUME THERE IS NO VENOUS LACTATE IN MIMIC DATA, CORRECT IF CHANGED')
    lactate_IDs = ['vm136']
    dopamine_ID = 'm_pm_2'
    phenylephrin_ID = 'm_pm_1'

    level1_drug_IDs = dobutamine_ID + milrinone_ID + [dopamine_ID, phenylephrin_ID]
    drug_IDs = level1_drug_IDs + norepinephrine_IDs + epinephrine_IDs + vasopressin_IDs
else:
    level1_drug_IDs = dobutamine_ID + milrinone_ID + levosimendan_ID + theophyllin_IDs
    drug_IDs = level1_drug_IDs + norepinephrine_IDs + epinephrine_IDs + vasopressin_IDs


if mimic:
    chunk_path = paths.chunks_file + '.' + in_version
else:
    chunk_path = paths.chunks_file
df_chunks = pd.read_csv(chunk_path)
files = glob.glob(in_dir + 'fmat*h5')
files = [f.split('/')[-1] for f in files]

def process_chunk(idx, reduced=False):
    """
    """
    print('WARNING: if this chunk already exists, data will be duplicated')
    print(idx)
    pids = df_chunks.loc[df_chunks.ChunkfileIndex == idx, 'PatientID']
    pids = pd.DataFrame(pids)
    idx_start = min(pids['PatientID'])
    idx_stop = max(pids['PatientID'])
    if reduced:
        print('Operating on reduced data!')
        chunkfile = in_dir + 'reduced_fmat_' + str(idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'
        outfile = out_dir + 'reduced_endpoints_' + str(idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'
    else:
        chunkfile = in_dir + 'fmat_' + str(idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'
        outfile = out_dir + 'endpoints_' + str(idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'
    print('Obtaining endpoints from', chunkfile, 'writing to', outfile)

    pid_dfs = []
    for pid in pids['PatientID']:
        gridded_patient = grid_patient(pid, grid_size_in_minutes, window_size_in_minutes, reduced, chunkfile=chunkfile)
        if not gridded_patient is None:
            gridded_patient.reset_index(inplace=True)
            pid_dfs.append(gridded_patient)
        else:
            print('No data for patient', pid)
    df = pd.concat(pid_dfs)
    df.to_hdf(outfile, 'endpoints', append=False, complevel=5,
            complib='blosc:lz4', data_columns=['PatientID'], format='table')
    print('finished processing chunk', chunkfile)
    return True

def merge_lactate(df):
    """
    """
    lactate = df.loc[:, lactate_IDs].mean(axis=1)
    df['lactate'] = lactate
    df.drop(lactate_IDs, inplace=True, axis=1)
    return df

def convert_drugs_to_rates(df, pid):
    """
    DEPRECATED
    Currently, drug info is "dose since last measurement", so we need to turn that into "dose per minute"
    Bolus drugs are assumed to act over 5 minutes (for these drugs).
    """
    print('convert_drugs_to_rates is deprecated')
    raise NotImplementedError
    for drug in drug_IDs:
        df_drug = df[drug].dropna()
        if df_drug.shape[0] == 0:
            # nothing going on here, pass
            df.loc[:, drug + '/min'] = 0
            continue
        # note, we can't say anything about the first time
        if drug in bolus_or_tablet_IDs:
            # bolus/tablet is assumed to operate over a 5 minute period, so we divide by 5 and forward fill for ONE time step
            current_dose_per_minute = df_drug.values/5
        else:
            minutes_elapsed = ((df_drug.index[1:] - df_drug.index[:-1]).astype('timedelta64[s]')/60)
            try:
                assert df_drug.values[0] == 0
            except AssertionError:
                print('WARNING: GivenDose was not 0 at the start. This is PROBABLY a Status 780 problem. DROPPING THIS MEASUREMENT.')
                bad_patients.write(str(int(pid)) + ',start_dose_not_0\n')
                df_drug.values[0] = 0
            current_dose_per_minute = df_drug.values[1:]/minutes_elapsed
            # add a 0 to the start, as it should be in these infusions
            current_dose_per_minute = np.concatenate([[0], current_dose_per_minute])
        df_drug = pd.DataFrame({drug: df_drug, drug + '/min': current_dose_per_minute})
        df = pd.concat([df, df_drug]).sort_index()
        # fill based on bolus or not
        if drug in bolus_or_tablet_IDs:
            # forward fill for ONE stuep
            df.loc[:, drug + '/min'] = df.loc[:, drug + '/min'].fillna(method='ffill', limit=1).fillna(0)
        else:
            # backwards fill, so the rate per minute is true for the whole region, and fill the rest with zeros
            df.loc[:, drug + '/min'] = df.loc[:, drug + '/min'].fillna(method='bfill').fillna(0)
    return df

def merge_process_drugs(df, pid):
    """
    merge identical drugs
    drop original columns
    """
    # already in rates!
#    df = convert_drugs_to_rates(df, pid)
#    df['time'] = (df.index - df.loc[df['v200'].notnull(), :].index[0]).astype('timedelta64[s]')/60
    df['dobutamine'] = df.loc[:, dobutamine_ID].fillna(method='ffill')
    if mimic:
        # these are missing in MIMIC
        df['milrinone'] = 0
        df['levosimendan'] = 0
        df['theophyllin'] = 0
        # these are NOT missing in MIMIC
        df['dopamine'] = df.loc[:, dopamine_ID].fillna(method='ffill')
        df['phenylephrin'] = df.loc[:, phenylephrin_ID].fillna(method='ffill')
    else:
        # these are NOT missing in Bern
        df['milrinone'] = df.loc[:, milrinone_ID].fillna(method='ffill')
        df['levosimendan'] = df.loc[:, levosimendan_ID].fillna(method='ffill')
        df['theophyllin'] = df.loc[:, theophyllin_IDs].fillna(method='ffill').sum(axis=1)
        # these are missing in Bern
        df['dopamine'] = 0
        df['phenylephrin'] = 0

    # epinephrine and norepinephrine must be per minute, per kg, in micrograms (already per-minute in micrograms from pharma table preprocessing)
    df['epinephrine'] = df.loc[:, epinephrine_IDs].fillna(method='ffill').sum(axis=1)
    df['norepinephrine'] = df.loc[:, norepinephrine_IDs].fillna(method='ffill').sum(axis=1)
    df['vasopressin'] = df.loc[:, vasopressin_IDs].fillna(method='ffill').sum(axis=1)
#    df.drop(drug_IDs, inplace=True, axis=1)
#    df.drop(list(map(lambda x: x + '/min', drug_IDs)), inplace=True, axis=1)
    # forward fill all the drugs
    return df

def ffill_drugs(df):
    for drug in ['dobutamine', 'milrinone', 'levosimendan', 'theophyllin', 'dopamine', 'phenylephrin',
            'epinephrine', 'norepinephrine', 'vasopressin']:
        df[drug] = df[drug].fillna(method='ffill')
    return True

def add_thresholds_markers(df):
    """
    Denote where thresholds are met, drugs are present, etc.
    """
    df['lactate_above_threshold'] = (df['lactate'] >= 2)        # note, this will be updated after we interpolate the lactate
    df.loc[df['lactate'].isnull(), 'lactate_above_threshold'] = np.nan
    df['MAP_below_threshold'] = (df[MAP_ID] <= 65)
    df.loc[df[MAP_ID[0]].isnull(), 'MAP_below_threshold'] = np.nan
    # level 1: presence of any of these
    df['level1_drugs_present'] = (df.loc[:, ['dobutamine', 'milrinone', 'levosimendan', 'theophyllin', 'dopamine', 'phenylephrin']] > 0).any(axis=1)
    # level 2: (nor)epinephrine between 0 and 0.1 microg/kg/min
    df['level2_drugs_present'] = ((df.loc[:, 'epinephrine'] > 0) & (df.loc[:, 'epinephrine'] < 0.1*df['weight'])) | ((df['norepinephrine'] > 0) & (df['norepinephrine'] < 0.1*df['weight']))
    #df['level2_drugs_present'] = ((df.loc[:, 'epinephrine'] > 0) & (df.loc[:, 'epinephrine'] < 0.1*df[weight_ID[0]])) | ((df['norepinephrine'] > 0) & (df['norepinephrine'] < 0.1*df[weight_ID[0]]))
    # level 3: (nor)epinephrine  >= 0.1 microg/kg/min, or any vasopressin
    #df['level3_drugs_present'] = (df['norepinephrine'] >= 0.1*df[weight_ID[0]]) | (df['epinephrine'] >= 0.1*df[weight_ID[0]]) | (df['vasopressin'] > 0)
    df['level3_drugs_present'] = (df['norepinephrine'] >= 0.1*df['weight']) | (df['epinephrine'] >= 0.1*df['weight']) | (df['vasopressin'] > 0)
    return True

def add_weight(df, pid):
    """
    Load height and weight from observrec, impute as necessary
    """
    # we previously had to get it from observrec, but it shoudl be in the dataframe now
    # forward and then backwards-fill weight
    df['weight'] = df[weight_ID].fillna(method='ffill').fillna(method='bfill')
    # if anything is missing at this point it means the patient has no weight values
    if df['weight'].isnull().sum() > 0:
        print('Weight is missing on patient', pid, ' - imputing from height if possible')
        typical_weight_dict = np.load(paths.misc_dir + 'typical_weight_dict.npy').item()
        bmi_dict = np.load(paths.misc_dir + 'median_bmi_dict.npy').item()
        # look for height in the static file - this will exist for mimic at some point
        if mimic:
            static_info = pd.read_hdf(paths.merged_dir + in_version + '/static.h5', where='PatientID == ' + str(pid), columns=['Sex', 'Height'])
        else:
            static_info = pd.read_hdf(paths.clean_dir + in_version + '/static.h5', where='PatientID == ' + str(pid), columns=['Sex', 'Height'])
        height = static_info['Height'].iloc[0]
        patient_sex = static_info['Sex'].iloc[0]
        try:
            if np.isnan(height):
                print('Missing height, imputing weight from standard measurement')
                df['weight'] = np.mean([x for x in typical_weight_dict.values()])
            else:
                print('Height is not missing, imputing weight from typical BMI')
                if patient_sex == 'M':
                    BMI = bmi_dict['male']
                elif patient_sex == 'F':
                    BMI = bmi_dict['female']
                elif patient_sex == 'U':
                    BMI = bmi_dict['unknown']
                # load BMI, do stuff here
                weight = BMI*((height/100)**2)
                df['weight'] = weight
        except:
            ipdb.set_trace()
    #print('add_weight is deprecated')
    #raise NotImplementedError
    # we have to get their height/weight from the observrec table, because it wasn't merged in this version

#    pids = df_chunks.loc[df_chunks.ChunkfileIndex == idx, 'PatientID']
#    pids = pd.DataFrame(pids)
#    idx_start = min(pids['PatientID'])
#    idx_stop = max(pids['PatientID'])
#    observ_file = observ_dir + 'observrec__' + str(idx) + '__' + str(idx_start) + '--' + str(idx_stop) + '.h5'
#    height_weight = pd.read_hdf(observ_file, where='PatientID == ' + str(pid), columns = ['Datetime', 'v10000400'])
#    patient_sex = static_info['Sex'].iloc[0]
#    height_weight['Height'] = static_info['Height'].iloc[0]
#    # always sort by time
#    height_weight.set_index('Datetime', inplace=True)
#    height_weight.sort_index(inplace=True)
#    height_weight.dropna(how='all')
#    # case 1, there is no data here
#    if height_weight.empty or height_weight.isnull().mean().mean() == 1:
#        # TODO get real prior values
#        print('missing both height and weight!')
#        typical_weight_dict = np.load(paths.misc_dir + 'typical_weight_dict.npy').item()
#        try:
#            df['weight'] = typical_weight_dict[patient_sex]
#        except KeyError:
#            print('WARNING: patient is nonbinary')
#            df['weight'] = np.mean([x for x in typical_weight_dict.values()])
#        # (don't care about height)
#    else:
#        height_weight = height_weight.fillna(method='ffill').fillna(method='bfill')
#        height_weight.drop_duplicates()      # once we merge it in we have to forward/backward fill it anyway
#        if height_weight['v10000400'].isnull().mean() == 1:
#            # weight is missing! (but height must have been observed)
#            # get BMI information to infer weight
#            bmi_dict = np.load(paths.misc_dir + 'median_bmi_dict.npy').item()
#            height_weight['v10000400'] = weight
#        height_weight.rename(columns={'v10000400': 'weight'}, inplace=True)
#        try:
#            assert set(height_weight.index) < set(df.index)
#            df = df.join(height_weight['weight'], how='left')
#            df['weight'] = df['weight'].fillna(method='ffill').fillna(method='bfill')
#        except AssertionError:
#            print('WARNING: height_weight indices arent contained in df')
#            ipdb.set_trace()
#            original_index = df.index.copy()
#            df = df.join(height_weight['weight'], how='outer')
#            df['weight'] = df['weight'].fillna(method='ffill').fillna(method='bfill')
#            df = df.loc[original_index, :]
    try:
        assert df['weight'].isnull().sum() == 0
    except AssertionError:
        ipdb.set_trace()
    return df

def grid_patient(pid, grid_size_in_minutes, window_size_in_minutes, reduced, chunkfile=None):
    # Note, we load HR so we have the full ICU stay
    if chunkfile is None:
        # get the chunkfile
        idx = df_chunks.loc[df_chunks.PatientID == pid, 'ChunkfileIndex'].values[0]
        pids = df_chunks.loc[df_chunks.ChunkfileIndex == idx, 'PatientID']
        pids = pd.DataFrame(pids)
        idx_start = min(pids['PatientID'])
        idx_stop = max(pids['PatientID'])
        if reduced:
            chunkfile = in_dir + 'reduced_fmat_' + str(idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'
        else:
            chunkfile = in_dir + 'fmat_' + str(idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'
    print('\tprocessing patient ' + str(int(pid)))
    p_df = pd.read_hdf(chunkfile, where='PatientID == ' + str(pid), columns=lactate_IDs + drug_IDs + ['Datetime'] + HR_ID + MAP_ID + weight_ID)
    if p_df.empty:
        print('WARNING: patient has no data')
        bad_patients.write(str(int(pid)) + ',no_data\n')
        return None
    elif p_df.drop(['Datetime'], axis=1).isnull().mean().mean() == 1:
        print('WARNING: patient has no data in these variables')
        bad_patients.write(str(int(pid)) + ',no_endpoint_variables\n')
        return None
    # sort temporally
    p_df.set_index('Datetime', inplace=True)
    p_df.sort_index(inplace=True)
    p_df = add_weight(p_df, pid)     # doesn't modify in place
    # merge identical values, drop the original ones
    p_df = merge_lactate(p_df)
    p_df = merge_process_drugs(p_df, pid)
    # start point is first HR measurement
    if p_df[HR_ID].dropna().empty:
        print('WARNING: patient has no HR')
        bad_patients.write(str(int(pid)) + ',no_HR\n')
        return None
    first_HR = p_df[HR_ID].dropna().index[0]
    last_HR = p_df[HR_ID].dropna().index[-1]
    p_df = p_df.loc[first_HR:last_HR, :]
    # resample to the grid size
    p_df = p_df.resample(str(grid_size_in_minutes) + 'T').median()
    # forward fill drugs again, to fill in gaps
    ffill_drugs(p_df)
    # identify thresholds 
    add_thresholds_markers(p_df)
    # DEBUG
    lactate_test = p_df['lactate_above_threshold'].copy()
    # interpolate lactate
    interpolate_patient_lactate(p_df, pid, grid_size_in_minutes)
    find_endpoints(p_df, MAP_ID, window_size_in_minutes=window_size_in_minutes, grid_size_in_minutes=grid_size_in_minutes)
    p_df['PatientID'] = pid
    # DEBUG
    assert p_df['lactate_above_threshold'].equals(lactate_test)
    return p_df

process_chunk(idx, reduced)
