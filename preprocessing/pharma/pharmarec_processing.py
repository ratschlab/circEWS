#!/usr/bin/env python
import sys
sys.path.append('../../utils')
sys.path.append('../')
import pandas as pd
import numpy as np
from time import time

from unit_normalization import *
from preproc_utils import *
import ipdb

dref = get_reference('DoseUnitChange')
fref = get_reference('FormUnitChange')

pharma_acting_period_dict = np.load('/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/stephanie/pharma_acting_period_v6.npy', allow_pickle=True).item()

def normalize_single_pharma_unit(pharma, reset_index=False):
    """
    Note, since we only look at GivenDose, we only care about the DoseUnitChange reference.
    """
    if len(pharma.VariableID.unique()) > 1:
        raise Exception('More than 1 pharma in the input pharmarec dataframe.')
    pharma_id = pharma.iloc[0].VariableID

    pharma_tmp = pharma.copy()
    if 'Datetime' in pharma_tmp.columns:
        pharma_tmp.set_index('Datetime', inplace=True)
        pharma_tmp.sort_index(inplace=True)
   
    # get the period of interest
    min_time = pharma.index.min()
    max_time = pharma.index.max()

    ref_thisdrug = dref[dref.PharmaID == pharma_id]
    if ref_thisdrug.shape[0] == 0:
        pass
    else:
        # now, for each timepoint we need to check what the unit at that time was
        ref_thisdrug.rename(columns={'ArchTime': 'Datetime'}, inplace=True)
        ref_thisdrug.set_index('Datetime', inplace=True)
        ref_thisdrug.sort_index(inplace=True)
        coefratio = ref_thisdrug.loc[:, ['CoefRatio']]
        # extend it to include all the indices from pharma_tmp, then forward fill
        joined = pd.concat([pharma_tmp, coefratio]).sort_index()
        coefratio = joined['CoefRatio'].fillna(method='ffill')

        coefratio = coefratio[np.logical_not(coefratio.index.duplicated(keep='first'))].copy()

        # subset back to the times included in pharma_tmp
        coefratio = coefratio.loc[pharma_tmp.index]
        pharma_tmp['CoefRatio'] = coefratio

        # now do the scaling
        pharma_tmp['Value'] = pharma_tmp['Value']*pharma_tmp['CoefRatio']
        # for debugging (usually it's 1)
        print('\t\t', pharma_tmp['CoefRatio'].mean())
        pharma_tmp.drop('CoefRatio', axis=1, inplace=True)
  
    # this is for backwards compatibility with xinrui's code
    if reset_index:
        pharma_tmp.reset_index(inplace=True)
    return pharma_tmp
    
def merge_single_pharma_infusions(pharma, lock=None):
    pid = pharma.iloc[0].PatientID
    pharma_tmp = pharma.copy()
    pharma_status = pharma_tmp.Status.unique()
    pharma_status_str = [get_status_str(tmp, 'pharmarec') for tmp in pharma_status]
    
    # status of injection, tablets and etc. which are not pharma items given in a constant rate
    # status_inst = [pharma_status[i] for i in range(len(pharma_status)) if 'start' in pharma_status_str[i] and 'stop' in pharma_status_str[i]]
    status_inst = [780]
    # status of the end of the pharma items given in a constant rate
    status_stop = [pharma_status[i] for i in range(len(pharma_status)) if 'start' not in pharma_status_str[i] and 'stop' in pharma_status_str[i]]

    # calibrate the units of the pharma item from different time periods to the latest unit
    pharma_tmp = normalize_single_pharma_unit(pharma_tmp, reset_index=True)

    # get the name of the pharma item as future column name of the pivoted table 
    pharma_name = pharma_tmp.iloc[0].VariableName

    max_index = pharma_tmp.index.max()
    # deal with each infusion individually
    for infusion_id in pharma_tmp.InfusionID.unique():
        infusion_df = pharma_tmp[pharma_tmp.InfusionID==infusion_id]

        # If the pharma variable with continuous rate does not have a start status,
        # the start time of the infusion is manually computed and add to the table
        if len(infusion_df) == 1 and infusion_df.iloc[0].Status != 780:
            pharma_tmp.loc[infusion_df.index[0], 'Status'] = 780
            infusion_df.loc[infusion_df.index[0], 'Status'] = 780

        # If the drug is given instantaneously in some channel, define the start time to be X min before the timepoint of the record with status 780,
        # and compute the rate accordingly
        try:
            pharma_acting_period_in_minutes = pharma_acting_period_dict[pharma_name]
        except KeyError:
            ipdb.set_trace()
            pharma_acting_period_in_minutes = 5
        if np.sum(infusion_df.Status==780) > 0:
            iloc_780 = np.where(infusion_df.Status==780)[0]
            for i in iloc_780:
                tmp_rate = infusion_df.iloc[i].Value / pharma_acting_period_in_minutes
                new_ind = str( max_index + 1)
                max_index += 1
                pharma_tmp.loc[new_ind] = infusion_df.iloc[i].copy()
                pharma_tmp.loc[new_ind, 'Value'] = 0
                pharma_tmp.loc[new_ind, 'Rate'] = tmp_rate
                pharma_tmp.loc[new_ind, 'Status'] = 524
                pharma_tmp.loc[new_ind, 'Datetime'] = infusion_df.iloc[i].Datetime - np.timedelta64(pharma_acting_period_in_minutes, 'm')
                pharma_tmp.loc[infusion_df.index[i], 'Rate'] = 0
                pharma_tmp.loc[infusion_df.index[i], 'Status'] = 776

         # if the last status of the infusion channel is not stop status, set it to stop status
        if infusion_df.iloc[-1].Status == 520:
            pharma_tmp.loc[infusion_df.index[-1], 'Status'] = 776
            pharma_tmp.loc[infusion_df.index[-1], 'Rate'] = 0

        # if initial status doesn't include start...
        if infusion_df.iloc[0].Status in [520, 8]:
            if infusion_df.iloc[0].Value == 0:
                # but the first Value is a zero, assume this is the start of the infusion
                pharma_tmp.loc[infusion_df.index[0], 'Status'] = 524
            else:
                rate = infusion_df.iloc[1].Value / ((infusion_df.iloc[1].Datetime - infusion_df.iloc[0].Datetime) / np.timedelta64(1, 'h'))
                if rate == 0:
                    ind = 2
                    while rate == 0 and ind < len(infusion_df):
                        # if the rate equals to 0, and the number of records are large enough, compute the
                        # the rate from next record
                        rate = infusion_df.iloc[ind].Value / ((infusion_df.iloc[ind].Datetime - infusion_df.iloc[ind-1].Datetime) / np.timedelta64(1, 'h'))
                        ind += 1

                    if rate==0:
                        # if the rate still equals to zero using all records in the current infusion channel, 
                        # delete the last record whose value is 0, and set the status of first record to 
                        # be 780 (instantaneous injection)
                        pharma_tmp.loc[infusion_df.index[0], 'Status'] = 780
                        pharma_tmp.drop(infusion_df.index[1:], inplace=True)
                        continue

                time_diff = int(infusion_df.iloc[0].Value/rate*3600)
                if time_diff > 0:
                    new_ind = str(max_index + 1)
                    max_index += 1
                    pharma_tmp.loc[new_ind] = infusion_df.iloc[0].copy()
                    pharma_tmp.loc[new_ind, 'Value'] = 0.0
                    pharma_tmp.loc[new_ind, 'Status'] = 524 if infusion_df.iloc[0].Status == 520 else 4
                    pharma_tmp.loc[new_ind, 'Rate'] = rate
                    pharma_tmp.loc[new_ind, 'Datetime'] = np.datetime64(infusion_df.iloc[0].Datetime) + np.timedelta64(-time_diff, 's')
                else:
                    pharma_tmp.loc[infusion_df.index[1], 'Value'] = infusion_df.iloc[0].Value + infusion_df.iloc[1].Value
                    pharma_tmp.loc[infusion_df.index[0], 'Value'] = 0.0
                    pharma_tmp.loc[infusion_df.index[0], 'Status'] = 524

    pharma_tmp.loc[pharma_tmp[pharma_tmp.Status==780].index, 'Rate'] = float('NaN')
    
    pharma_tmp.sort_values(['InfusionID', 'Datetime'], inplace=True)

    # pivoted the pharma table to a new table where the column is the values of the pharma item
    col_pivoted = pd.pivot_table(pharma_tmp, values='Value', aggfunc=np.sum,
                                 index=['PatientID', 'Datetime'], columns=['VariableName'])

    # pivot the table to a table where each column is an infusion channel and whose values are rates
    original_index = pharma_tmp.index
	
    rate_pivoted = pd.pivot_table(pharma_tmp, values='Rate', dropna=False,
            index=['PatientID', 'Datetime'], columns=['InfusionID'])

    # pivot the table to a table where each column is an infusion channel and whose values are statuses
    status_pivoted = pd.pivot_table(pharma_tmp, values='Status', dropna=False, 
            index=['PatientID', 'Datetime'], columns=['InfusionID'])
    # pivot the table to a table where each column is an infusion channel and whose values are given doses
    value_pivoted = pd.pivot_table(pharma_tmp, values='Value', aggfunc=np.sum, dropna=False, 
            index=['PatientID', 'Datetime'], columns=['InfusionID'])

    # comptue the rate of the pharma item in different channels
    # initialize the rate the pharma item using the rate_pivoted
    rate_computed = rate_pivoted.copy()
    rate_computed.reset_index(inplace=True)
    # here we look at the infusion channels one by one
    for infusion_id in rate_pivoted.columns:
        # here we look at the non-nan statuses of the current infusion channel
        tmp_status = status_pivoted[infusion_id].dropna().to_frame()
        tmp_status.reset_index(inplace=True)

        # If the pharma variable with continuous rate has no stop status, then 
        # enforce the last rate to be 0 (as stop ) 
        if tmp_status.iloc[-1][infusion_id] not in status_stop:
            rate_computed.loc[rate_computed[rate_computed.Datetime==tmp_status.iloc[-1].Datetime].index,infusion_id] = 0.0                    
        # we look at the non-nan given doses of the current infusion channel
        tmp_value = value_pivoted[infusion_id].dropna().to_frame()
        tmp_value.reset_index(inplace=True)

        ## If the infusion status is instantaneous statuses like 780 and 782, whose
        ## timestamps are interleaved with continuous infusion, don't use them to
        ## compute the infusion rate because their rate is 'nan' (set value to nan)
        idx_inst = tmp_status[tmp_status[infusion_id]==status_inst].index
        if len(idx_inst) > 0:
            tmp_value.loc[idx_inst, infusion_id] = np.nan


        tmp_value.dropna(inplace=True)
        tmp_datetime = tmp_value.Datetime.tolist()
        # if there are more than 1 record in the current infusion channel, then there is at least one
        # real-value rate; otherwise, the rate of the pharma item in the current channel is 'nan'
        if len(tmp_datetime) > 1:
            tdiff = time_difference(tmp_datetime[:-1], tmp_datetime[1:])*60     # the function gives the time difference in hours, we need it in minutes
            givendose = np.array(tmp_value.iloc[1:][infusion_id].tolist())
            rate_tmp = givendose / tdiff
            for i in range(len(tmp_datetime)-1):
                # assign constant rate to all timepoints within the interval of current recorded timepoint
                # and the next recorded time in the current infusion channel
                within_interv = (rate_computed.Datetime>=tmp_datetime[i]) & (rate_computed.Datetime<tmp_datetime[i+1])
                rate_computed.loc[within_interv , infusion_id] = rate_tmp[i]
        else:
            rate_computed.loc[:,infusion_id] = np.nan

    rate_computed.set_index(['PatientID', 'Datetime'], inplace=True)
    # initialize the computed value matrix with the pivoted value table
    value_mat = value_pivoted.copy().as_matrix()

    # assign 0 to nans
    value_mat[np.isnan(value_mat)] = 0

    # assign 0 to timepoint-and-channels where the statuses are not instantaneous statuses
    # which means that we keep the values of the timepoint-and-channels where the statuses are instantaneous
    value_mat[status_pivoted.as_matrix()!=status_inst] = 0

    # the given dose at a timepoint is the sum of values over different infusion channels at that timepoint
    # the column representing the pharma table in a bigger table is initialized by only summing all the instantaneous values
    col_pivoted = pd.DataFrame(np.sum(value_mat, axis=1), index=value_pivoted.index, columns=[pharma_name])
    if np.sum(status_pivoted.as_matrix()==status_inst) > 0:
        col_pivoted_inst = col_pivoted[np.sum(status_pivoted.as_matrix()==status_inst, axis=1) > 0].copy()
        col_pivoted_inst.rename(columns={pharma_name:pharma_name+'_inst'}, inplace=True)
        col_pivoted.loc[:,:] = 0
    else:
        col_pivoted_inst = None
    col_pivoted.reset_index(inplace=True)

    for i in np.arange(1, len(col_pivoted)):
        tdiff = time_difference(col_pivoted.iloc[i-1].Datetime, col_pivoted.iloc[i].Datetime)*60
        # compute the given dose at a current time point by adding up the given dose from all the constant-flow infusion channels,
        # and also the instantaneous give dose values from other infusion channels if they exist.
        col_pivoted.loc[col_pivoted.index[i], pharma_name] += tdiff * np.nansum(rate_computed.iloc[i-1])

    col_pivoted.set_index(['PatientID', 'Datetime'], inplace=True)
    col_pivoted.rename(columns={pharma_name:pharma_name+'_flow'}, inplace=True)
    if col_pivoted_inst is not None:
        col_pivoted = col_pivoted.join(col_pivoted_inst, how='outer')

    try:
        assert(np.abs(np.nansum(col_pivoted.as_matrix()) - np.nansum(pharma_tmp.Value)) < 1e-3)
    except AssertionError:
        print('Assertion Error!')
        ipdb.set_trace()

    output_rate = True
    if output_rate:
        rate_computed_merged = pd.DataFrame(np.nansum(rate_computed.as_matrix(), axis=1), columns=[pharma_name], index=col_pivoted.index)
        return rate_computed_merged
    else:
        col_pivoted_merged = pd.DataFrame(np.nansum(col_pivoted.as_matrix(), axis=1), columns=[pharma_name], index=col_pivoted.index)
        return col_pivoted_merged

def steph_process_status780(df, infusionID, pharma_acting_period_in_minutes, counter):
    """
    If the drug is given instantaneously in some channel, 
    define the end time to be bolus_duration_in_minute *after* the timepoint of the record with status 780,
    and convert it to a "virtual" normal infusion during that time.

    Each virtual infusion gets a new infusion ID (in case they can be overlapping).
    """
    start_code = 524
    stop_code = 776

    times_780 = df.index[df.Status == 780]
    for time_780 in times_780:
        given_dose = df.loc[time_780, 'Value']
        if given_dose == 0:
            # drop this since it's useless
            df.drop(time_780, inplace=True)
        else:
            df.loc[time_780, 'Value'] = 0
            df.loc[time_780, 'Status'] = start_code
            df.loc[time_780, 'InfusionID'] = str(infusionID) + '_' + str(counter)

            end_time = time_780 + pd.Timedelta(pharma_acting_period_in_minutes, unit='m')
            # EDGE case: if the time already exists, you just overwrite another infusion
            while end_time in df.index:
                end_time = end_time - pd.Timedelta(1, unit='s')

            df.loc[end_time] = df.loc[time_780].copy()

            df.loc[end_time, 'Value'] = given_dose
            df.loc[end_time, 'Status'] = stop_code
            df.loc[end_time, 'InfusionID'] = str(infusionID) + '_' + str(counter)

            counter +=1 
    df.sort_index(inplace=True)

def steph_process_subinfusion(df, subinfusion_id):
    """
    NOTE: the logic is that the value of the Rate at t is the rate until the next measurement.
    That is, we must forward fill.
    """
    assert df.Value.iloc[0] == 0
    minutes_elapsed = ((df.index[1:] - df.index[:-1]).astype('timedelta64[s]')).values/60
    dose_received = df['Value'].values[1:]
    rate = dose_received/minutes_elapsed
    # add a 0 at the end
    rate = np.concatenate([rate, [0]])
    # set the final rate to be 0
    df['Rate'] = rate

    df.drop(['InfusionID', 'Value'], axis=1, inplace=True)
    df.rename(columns={'Rate': 'Infusion_' + str(subinfusion_id)}, inplace=True)
    return df

def steph_process_missing_start(df):
    print('starting value is not 0')
    # copying the logic xinrui used, mostly
    # if there are subsequent records (there must be), estimate the rate
    # assume the rate is constant and back-project to figure out when this infusion must have started
    dose_difference = df.iloc[1].Value = df.iloc[0].Value
    time_elapsed = (df.index[1] - df.index[0])/np.timedelta64(1, 'm')
    estimated_rate = dose_difference/time_elapsed
    if estimated_rate == 0:
        ipdb.set_trace()
    else:
        if df.iloc[1].Status == 776:
            print('status 776 observed')
        if df.iloc[1].Status == 776 and df.iloc[1].Value == 0:
            # the next time has a stop status, but no dose was given during that period
            # treat the first timepoint like status780, and drop this one
            if not df.shape[0] == 2:
                ipdb.set_trace()
            df.drop(df.index[1], inplace=True)
            print('RESETTING STATUS 776 SITUATION')
            df.Status = 780
        else:
            time_elapsed = df.Value.values[0]/estimated_rate
            start_time = df.index[0] - pd.Timedelta(time_elapsed, unit='m')
            df.loc[start_time, 'Value'] = 0
            df.loc[start_time, 'Status'] = 524
            df.loc[start_time, 'InfusionID'] = df.iloc[0].InfusionID
            df.sort_index(inplace=True)

def steph_rates_from_single_infusion(df, pharma_acting_period_in_minutes):
    """
    """
    InfusionID = int(df['InfusionID'].iloc[0])
    # first up, we drop the Rate column from the database (we don't trust it at time of writing)
    df['Rate'] = np.nan

    # DEBUG CHECK
    if (df.Status == 524).sum() > 1:
        print('multiple starts on this infusion?')
        ipdb.set_trace()

    # if there was no dose given on this infusion, forget about it
    if df.Value.sum() == 0:
        return []

    # if there is only a single value, we have to treat it as 780
    if df.shape[0] == 1:
        df.Status = 780

    counter_780 = 0
    steph_process_status780(df, InfusionID, pharma_acting_period_in_minutes, counter_780)
    assert (df.Status == 780).sum() == 0

    assert df.shape[0] >= 2

    if not df.Value.values[0] == 0:
        steph_process_missing_start(df)

    # processing missing start can reintroduce status 780
    if (df.Status == 780).sum() > 0:
        steph_process_status780(df, InfusionID, pharma_acting_period_in_minutes, counter_780)
        assert (df.Status == 780).sum() == 0

    # asserts, etc.
    # now, convert all Values into Rates (lose first row)
    # rename, drop, simplify (all we need is rate at a given time)
    df.drop(['Status', 'PatientID', 'VariableID', 'VariableName'], axis=1, inplace=True)
    # we have to split into SUB infusions due to status_780
    individual_subinfusions = []
    for subinfusion_id in df.InfusionID.unique():
        subinfusion = df[df.InfusionID == subinfusion_id].copy()
        if subinfusion.Value.sum() == 0:
            # just don't add it
            pass
        if subinfusion.shape[0] > 0:
            try:
                subinfusion = steph_process_subinfusion(subinfusion, subinfusion_id)
            except AssertionError:
                print('issue processing subinfusion')
                ipdb.set_trace()
            individual_subinfusions.append(subinfusion)
    return individual_subinfusions

def steph_rates_from_all_infusions(pharma):
    """

    """
    pharma.set_index('Datetime', inplace=True)
    pharma.sort_index(inplace=True)
    # normalise records 
    pharma = normalize_single_pharma_unit(pharma)
    #
    pharma_id = pharma.VariableName.iloc[0]
    pid = pharma.PatientID.iloc[0]
    # TODO: we need this for each variable
    try:
        pharma_acting_period_in_minutes = pharma_acting_period_dict[pharma_id]
    except KeyError:
        print('WARNING: no acting period for drug', pharma_id, '- setting to 5 for now')
        pharma_acting_period_in_minutes = 5
    #
    # for each infusion, convert it to rates
    infusion_df_list = []
    for infusion in pharma.InfusionID.unique():
        infusion_df = pharma.loc[pharma.InfusionID == infusion, :].copy()
        try:
            infusion_rates = steph_rates_from_single_infusion(infusion_df, pharma_acting_period_in_minutes)
        except ValueError:
            print('seem to be missing data in an infusion')
            ipdb.set_trace()
        infusion_df_list.extend(infusion_rates)

    if len(infusion_df_list) == 0:
        rate_over_all_infusions = pd.DataFrame({'PatientID': [pid]*pharma.shape[0], 'Datetime': pharma.index, pharma_id: [0]*pharma.shape[0]})
        rate_over_all_infusions.set_index(['PatientID', 'Datetime'], inplace=True)
    else:
        # try:
        infusions_wide = pd.concat(infusion_df_list, axis=1)
        # except:
        #     import ipdb
        #     ipdb.set_trace()

        infusions_wide.sort_index(inplace=True)
        infusions_wide.fillna(method='ffill', inplace=True)
        # summing over NaNs contributes 0, we don't need to replace them
        rate_over_all_infusions = pd.DataFrame({'PatientID': [pid]*infusions_wide.shape[0], 'Datetime': infusions_wide.index, pharma_id: infusions_wide.sum(axis=1)})
        rate_over_all_infusions.set_index(['PatientID', 'Datetime'], inplace=True)

    # convert it to floats
    rate_over_all_infusions = rate_over_all_infusions.astype('float64')
    return rate_over_all_infusions

def steph_pivot_single_patient_pharma_table(pharma):
    pharma_ids = pharma.VariableID.unique()
    pharma_id_cols = []
    for pharma_id in pharma_ids:
        pharma_df = pharma[pharma.VariableID==pharma_id].copy()
        pharma_id_rate = steph_rates_from_all_infusions(pharma_df)
        # check if the individual pharma ids have duplicated indices
        pharma_id_cols.append(pharma_id_rate)
    # now merge them all
    df_pivoted = pharma_id_cols[0]
    for col in pharma_id_cols[1:]:
        df_pivoted = df_pivoted.join(col, how='outer')
#    df_pivoted = pd.concat(pharma_id_cols)
    df_pivoted.sort_index(inplace=True)
    # now forward fill, fill zeros
    df_pivoted.fillna(method='ffill', inplace=True)
    df_pivoted.fillna(0, inplace=True)
    if df_pivoted.index.duplicated().sum() > 0:
        if df_pivoted.loc[df_pivoted.index.value_counts().index[0]].duplicated().sum() > 0:
            df_pivoted = df_pivoted[np.logical_not(df_pivoted.index.duplicated())].copy()
        else:
            ipdb.set_trace()
    return df_pivoted

def pivot_single_patient_pharma_table(pharma, lock=None):
    cols_pivoted = []
    pharma_ids = pharma.VariableID.unique()
    for pharma_id in pharma_ids:
        pharma_df = pharma[pharma.VariableID==pharma_id].copy()
        cols_pivoted.append(merge_single_pharma_infusions(pharma_df, lock))
    df_pivoted = cols_pivoted[0]
    if len(cols_pivoted) > 1:
        for i in range(1, len(cols_pivoted)):
            df_pivoted = df_pivoted.join(cols_pivoted[i], how='outer')
    return df_pivoted

def pivot_pharma_table(pharma, switch='xinrui'):
    # this function gets called by raw2pivoted.py in table2matrix
    # currently it seems like pharma is for a single patient already, so the loop here is unnecessary
    if switch == 'xinrui':
        df_pivoted = []
        pharma.sort_values(['PatientID', 'VariableID', 'InfusionID', 'Datetime'], inplace=True)
        for patient_id in pharma.PatientID.unique():
            patient_pharma = pharma[pharma.PatientID==patient_id]
            df_pivoted.append(pivot_single_patient_pharma_table(patient_pharma))
        df_pivoted = pd.concat(df_pivoted)
    elif switch == 'steph':
        df_pivoted = steph_pivot_single_patient_pharma_table(pharma)
    return df_pivoted
