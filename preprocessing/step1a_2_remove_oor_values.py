#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import gc
import os

import sys
sys.path.append('../utils')
import preproc_utils
import matplotlib.pyplot as plt


def cumul_val_to_rate(df, variable_ids):
    """
    Convert cumulative values to discrete values 
    """ 
    old_len = len(df)
    df.sort_values(['VariableID', 'Datetime', 'Value'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    pid = df.iloc[0].PatientID
    time_unit = np.timedelta64(3600, 's')


    df.loc[:,'Rate'] = 0
    for vid in variable_ids:
        if np.sum( df.VariableID==vid ) == 0:
            continue
            
        if np.sum( df.VariableID==vid ) == 1:
            df.drop(df.index[df.VariableID==vid], inplace=True)
            continue
            
        df_vid = df[df.VariableID==vid].copy()
        if df_vid.Datetime.duplicated().sum() > 0:
            df.drop(df_vid.index[df_vid.Datetime.duplicated(keep='last')], axis=0, inplace=True)
            df_vid.drop_duplicates('Datetime', keep='last', inplace=True)
            
        valdiff = np.diff(df_vid.Value.values)
        valdiff[valdiff<0] = 0
        tdiff =  np.diff(df_vid.Datetime.values) / time_unit
        rate = valdiff / tdiff
        df.loc[df_vid.index[:-1], 'Rate'] = rate            
        if len(df_vid[df_vid.Value>40000]) > 0:
            if len(np.where(np.diff(np.where(df_vid.Value.values>40000)[0]) > 1)[0]) > 0:
                idx_outliers = np.where(df_vid.Value.values>40000)[0]
                idx_change_points = np.where(np.diff(idx_outliers)>1)[0]
                for i in range(len(idx_change_points)):
                    if i==0:
                        tmp_idx_outliers = idx_outliers[idx_outliers<idx_outliers[idx_change_points[i]]]
                    elif i<len(idx_change_points)-1:
                        tmp_idx_outliers = idx_outliers[np.logical_and(idx_outliers>=idx_outliers[idx_change_points[i]],
                                                                       idx_outliers<idx_outliers[idx_change_points[i+1]])]
                    else:
                        tmp_idx_outliers = idx_outliers[idx_outliers>=idx_outliers[idx_change_points[i]]]
                    tmp_idx_outliers = tmp_idx_outliers-1
                    # display(df.loc[df_vid.index[tmp_idx_outliers], 'Rate'])
                    df.loc[df_vid.index[tmp_idx_outliers], 'Rate'] = df.loc[df_vid.index[tmp_idx_outliers[0]-1], 'Rate']       
                    # display(df.loc[df_vid.index[tmp_idx_outliers], 'Rate'])
            else:
                idx_outliers = np.where(df_vid.Value.values>40000)[0]-1
                # display(df.loc[df_vid.index[idx_outliers]])
                df.loc[df_vid.index[idx_outliers], 'Rate'] = df.loc[df_vid.index[idx_outliers[0]-1], 'Rate']
                # display(df.loc[df_vid.index[idx_outliers]])
    df.loc[df.index[df.VariableID.isin(variable_ids)], 'Value'] = df.loc[df.index[df.VariableID.isin(variable_ids)], 'Rate']
    df.drop('Rate', axis=1, inplace=True)   
    return df


def remove_records_with_invalid_status(df, tbl_name):
    status_set = df.Status.unique()
    if tbl_name == 'pharmarec':
        status_binary = ['{0:10b}'.format(s)[::-1] for s in status_set]
    else:
        status_binary = ['{0:11b}'.format(s)[::-1] for s in status_set]
    invalid_status_set = status_set[np.where( [x[1]=='1' for x in status_binary])]
    if len(invalid_status_set) > 0:
        df.drop(df.index[df.Status.isin(invalid_status_set)], inplace=True)
    return df


def change_arterial_to_venous(labres, voi, tbl_name, index_chunk, monvals_svo2_path):
    vid_arterial_to_venous = {24000529: 24000740, 20001300: 24000739, 24000526: 24000837, 24000548: 24000836, 
                              20004200: 24000731, 24000867: 24000833, 24000524: 24000732, 24000866: 24000835, 
                              24000525: 24000735, 24000513: 24000733, 24000514: 24000736, 24000512: 24000734, 
                              20000800: 24000737, 24000530: 24000738}
    arterial_vids_with_venous = [key for key in vid_arterial_to_venous.keys()]
    arterial_vids_wo_venous = [20000200, 20000300, 20001200, 24000426, 24000521, 24000522, 24000549]
    
    patient_ids = labres.PatientID.unique()
    if 20000800 not in labres.VariableID.unique():
        return labres
    svo2 = pd.read_hdf(monvals_svo2_path, where='PatientID=%d'%labres.iloc[0].PatientID, mode='r')
    if len(svo2) == 0:
        return labres

    
    sao2_id = 20000800
    sao2_dt = labres[labres.VariableID==sao2_id].Datetime.values.reshape((-1, 1))
    sao2_dt_tiled = np.tile(sao2_dt, (1, len(svo2)))
    svo2_dt = svo2.Datetime.values.reshape((1, -1))
    tdiff = np.abs(sao2_dt_tiled - svo2_dt) / np.timedelta64(1, 'm')
    tdiff_min = np.min(tdiff, axis=1)
    tdiff_argmin = np.argmin(tdiff, axis=1)

    for i in np.where(tdiff_min <= 4)[0]:
        svo2_dt_tmp = svo2_dt[0,tdiff_argmin[i]]
        svo2_value = svo2[svo2.Datetime==svo2_dt_tmp].iloc[0].Value

        sao2_dt_tmp = sao2_dt[i,0]
        sao2_et_tmp = labres[np.logical_and(labres.VariableID==sao2_id, labres.Datetime==sao2_dt_tmp)].EnterTime
        for et in sao2_et_tmp:
            sao2_value = labres[np.logical_and(labres.VariableID==sao2_id, labres.EnterTime==et)].iloc[0].Value

            if (sao2_value - svo2_value) / 8.454 < 0.1:
                # look for all variables that is within 2 min window centered at the current datetime
                possible_venous = labres[labres.Datetime==sao2_dt_tmp]

                possible_venous = possible_venous[np.logical_and(possible_venous.EnterTime >= et-np.timedelta64(5, 's'), 
                                                                 possible_venous.EnterTime <= et+np.timedelta64(5, 's'))]
                idx_no_venous = possible_venous.index[possible_venous.VariableID.isin(arterial_vids_wo_venous)]
                if len(idx_no_venous) > 0:
                    labres.drop(idx_no_venous, inplace=True)
                possible_venous = possible_venous[possible_venous.VariableID.isin(arterial_vids_with_venous)]
                assert(np.sum(possible_venous.VariableID.value_counts()>1)==0)
                if len(possible_venous) > 0:
                    labres.loc[possible_venous.index, 'VariableID'] = possible_venous.VariableID.apply(lambda x: vid_arterial_to_venous[x])

    return labres


def remove_out_of_range(df, val_range, tbl_name, index_chunk):
    vid_list = df.VariableID.unique()
    if 120 in vid_list or 170 in vid_list:
        for vid in [120, 170]:
            tmp = df[df.VariableID==vid]
            if len(tmp) == 0:
                continue
            et_oor = tmp[np.logical_or(tmp.Value > val_range.loc[vid].UpperBound, 
                                       tmp.Value < val_range.loc[vid].LowerBound)].Datetime
            for et in et_oor:
                df_dt_oor = df[np.logical_and(df.Datetime >= et - np.timedelta64(30,'s'),
                                              df.Datetime <  et + np.timedelta64(30,'s'))]
                
                df_spike = df.loc[df_dt_oor.index[df_dt_oor.VariableID.isin([vid-20, vid-10, vid])]].copy()
                df.drop(df_dt_oor.index[df_dt_oor.VariableID.isin([vid-20, vid-10, vid])], inplace=True)

    for vid in df.VariableID.unique():
        if vid in val_range.index[val_range.LowerBound.notnull()|val_range.UpperBound.notnull()]:
            tmp = df[df.VariableID==vid]
            index_oor = tmp.index[np.logical_or(tmp.Value > val_range.loc[vid].UpperBound, 
                                                tmp.Value < val_range.loc[vid].LowerBound)]
            if len(index_oor) > 0:
                df.drop(index_oor, inplace=True) 
        else:
            pass
    gc.collect()
    return df

def increase_categorical_counter_to_merge(df):
    """
    The urine cluture variable will be merged with the (categorical) blood cluture location variable.
    The blood cluture location varibale conatins values 0-4. Urine contains values 1-2
    The urine values will be incremented by 4
    """
    df.loc[df.VariableID == 15002175,'Value'] = df.loc[df.VariableID == 15002175,'Value'].astype(float)+4
    return df


def correct_weight_height(df):
    # Height:10000450; Weight:10000400
    cols = {'Height': 10000450, 'Weight': 10000400}

    # delete height record if the height is higher than 220 cm
    df.loc[df.index[np.logical_and(df.VariableID==10000450, df.Value>10000)], 'Value'] /= 100
    df.drop(df.index[np.logical_and(df.VariableID==10000450, df.Value>240)], inplace=True)
    # delete weight record if the weight is heavier than 500 kg
    df.drop(df.index[np.logical_and(df.VariableID==10000400, df.Value>500)], inplace=True)

    df_tmp = df[df.VariableID.isin([10000450, 10000400])]
    if len(df_tmp)== 0:
        return df

    height_weight = pd.pivot_table(df_tmp.copy(), values='Value', index='EnterTime', columns='VariableID')
    if height_weight.shape[1] == 1:
        if height_weight.columns[0] == cols['Height']:
            df.drop(df.index[np.logical_and(df.VariableID==10000450, df.Value<130)], inplace=True)
        else:
            pass
        return df
            
    imputed = height_weight.copy().fillna(method='ffill')
    imputed.fillna(method='bfill', inplace=True)
    for i in range(len(imputed)):
        height = imputed.iloc[i][cols['Height']]
        weight = imputed.iloc[i][cols['Weight']]

        if height <= 120 and weight >= 120:
            imputed.loc[imputed.index[i], cols['Weight']] = height
            imputed.loc[imputed.index[i], cols['Height']] = weight
            height = weight
            weight = imputed.loc[imputed.index[i], cols['Weight']]

        elif height < 100 and height >= 30:
            height += 100
            imputed.loc[imputed.index[i], cols['Height']] = height


        if height < 130:
            imputed.loc[imputed.index[i], cols['Height']] = float('NaN')
        else:
            bmi = weight / ((height/100)**2)
            if bmi > 60 or bmi < 10:
                imputed.loc[imputed.index[i], cols['Weight']] = float('NaN')
                imputed.loc[imputed.index[i], cols['Height']] = float('NaN')
            
            
    for i in range(len(df_tmp)):
        vid = df_tmp.iloc[i].VariableID
        df.loc[df_tmp.index[i], 'Value'] = imputed[imputed.index==df_tmp.iloc[i].EnterTime].iloc[0][vid]

    df.dropna(how='any', inplace=True)
    return df   


def main():
    data_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', version)
    input_path = os.path.join(data_path, 'datetime_fixed', tbl_name)
    output_path = os.path.join(data_path, 'oor_removed', tbl_name)
    if not os.path.exists(output_path) and output_to_disk:
        os.makedirs(output_path)

    pid_chunkfile_index = preproc_utils.get_chunking_info(version=version)
    pid_list = np.array( pid_chunkfile_index.index[pid_chunkfile_index.ChunkfileIndex==index_chunk] )
    output_path = os.path.join(output_path, '%s_%d_%d--%d.h5'%(tbl_name, index_chunk, np.min(pid_list), np.max(pid_list)))

    # Load the global std values for all variables
    voi = preproc_utils.voi_id_name_mapping(tbl_name, replace_name=True, version=version)
    if tbl_name == 'labres':
        voi['VariableName'] = voi.VariableName.apply(lambda x: x.replace("'", ""))
        voi.loc[24000737, 'VariableName'] = 'v-SO2'
        vid_with_reasonable_0 = voi.index[voi.LowerBound < 0]
    else:
        vid_with_reasonable_0 = voi.index[np.logical_or(np.logical_or(voi.LowerBound < 0, voi.NormalValue==0),
                                                        voi.MetaVariableUnit.apply(lambda x: ('ordinal' in x.lower() or 'categorical' in x.lower() or 'yes/no' in x.lower()) if type(x)==str else False))]
    vid_with_nonsense_0 = list(set(voi.index) - set(vid_with_reasonable_0))

    df_idx_start = 0
    num_pid = len(pid_list)
    df_height = []
    cnt_pid_urine = 0
    for i, pid in enumerate(pid_list):
        filename = [f for f in os.listdir(input_path) if '%s_%d_'%(tbl_name, index_chunk) in f][0]
        df = pd.read_hdf(os.path.join(input_path, filename), where='PatientID=%d'%pid, mode='r')
        
        if len(df) == 0:
            print('Patient', pid, 'have no data in %s'%tbl_name)
            continue
            
        # rename columns of the pharmarec table
        df.rename(columns={'PharmaID': 'VariableID', 'GiveDose': 'Value', 'DateTime': 'Datetime', 
                           'SampleTime': 'Datetime'}, inplace=True)
        
        # select only variables of interest
        vid_intersect = set(df.VariableID) & set(voi.index)

        # add height (10000450) to the variables of interests if the table is 'observrec'
        if tbl_name == 'observrec':
            vid_intersect |= {10000450}
        elif tbl_name == 'dervals':
            vid_intersect |= {830005420, 30015110, 30015010, 30015075, 30015080}

        df.drop(df.index[~df.VariableID.isin(vid_intersect)], inplace=True)
        gc.collect()

        if len(df) == 0:
            print('Patient', pid, 'have no data of interest in %s'%tbl_name)
            continue

        # Only remove value 0 for variables for which 0 doesn't have any clinical meaning
        if tbl_name not in ['dervals', 'pharmarec']:
            index_nonsense_0 = df.index[np.logical_and(df.Value==0, df.VariableID.isin(vid_with_nonsense_0))]
            # if len(index_nonsense_0) > 0:
            #     print('Patient', pid, 'has non-sense 0 records.')
            #     print(df.loc[index_nonsense_0])
            #     df.drop(index_nonsense_0, inplace=True)

        # remove records with status containing 2 (invalidated)
        df = remove_records_with_invalid_status(df, tbl_name)

        df.sort_values(by=['Datetime', 'VariableID', 'EnterTime'], inplace=True)

        # remove identical records
        df.drop_duplicates(['Datetime', 'VariableID', 'Value', 'Status'], inplace=True)

        if tbl_name == 'labres':
            monvals_svo2_path = os.path.join(data_path, 'datetime_fixed', 'monvals_svo2', 
                                             'monvals_svo2_%d_%d--%d.h5'%(index_chunk, np.min(pid_list), np.max(pid_list))) 
            df = change_arterial_to_venous(df, voi, tbl_name, index_chunk, monvals_svo2_path)
            df.drop_duplicates(['Datetime', 'VariableID', 'Value'], inplace=True)
            # fixed troponin conversion
            if 24000538 in df.VariableID.unique():
                df.loc[df.index[df.VariableID==24000538], 'Value'] = df[df.VariableID==24000538].Value.values * 1000
            if 24000806 in df.VariableID.unique():
                if df[df.VariableID==24000806].Datetime.min() <= np.datetime64('2016-05-01'):
                    idx_to_convert = df.index[np.logical_and(df.VariableID==24000806, df.Datetime <= np.datetime64('2016-05-01'))]
                    df.loc[idx_to_convert,'Value'] = df.loc[idx_to_convert, 'Value'] * 1000

        elif tbl_name == 'dervals':
            cumulative_variable_ids = set(voi.index[voi.VariableName.apply(lambda x: 'cumul' in x or '/c' in x)])
            cumulative_variable_ids &= set(df.VariableID.tolist())
            if len(cumulative_variable_ids) == 0:
                pass
                # print('Patient', pid, 'does not have cumulative dervals variables.')
            else:
                cumulative_variable_ids = np.sort(list(cumulative_variable_ids))
                df = cumul_val_to_rate(df, cumulative_variable_ids)

        elif tbl_name == 'observrec':
            cumulative_variable_ids = set(voi.index[voi.VariableName.apply(lambda x: 'cumul' in x or '/c' in x)])
            cumulative_variable_ids &= set(df.VariableID.tolist())
            if len(cumulative_variable_ids) == 0:
                pass
                # print('Patient', pid, 'have cumulative observrec variables.')
            else:
                cnt_pid_urine += 1
                cumulative_variable_ids = np.sort(list(cumulative_variable_ids))
                df_old = df.copy()
                df = cumul_val_to_rate(df, cumulative_variable_ids)
                assert(df.Value.max()!=float('Inf'))
                df = correct_weight_height(df)
                df_height.append(df[df.VariableID==10000450].copy())
                df.drop(df.index[df.VariableID==10000450], inplace=True)
                df = increase_categorical_counter_to_merge(df)

        df = remove_out_of_range(df, voi, tbl_name, index_chunk)
        df.drop(df.index[~df.VariableID.isin(vid_intersect)], inplace=True)

        df.set_index(np.arange(df_idx_start, df_idx_start+len(df)), drop=True, inplace=True)

        df_idx_start += len(df)
        if output_to_disk:
            df.to_hdf(output_path, 'data', append=True, format='table', data_columns=['PatientID', 'VariableID'], complevel=5, complib='blosc:lz4')

        if (i+1)%50 == 0:
            print('%d / %d'%(i+1, num_pid))
               
        gc.collect()
    print(cnt_pid_urine)
        
    if tbl_name == 'observrec' and output_to_disk:
        height_info_path = os.path.join(data_path, 'oor_removed', 'height')
        if not os.path.exists(height_info_path):
            os.mkdir(height_info_path)

        df_height = pd.concat(df_height, axis=0)
        df_height.reset_index(inplace=True, drop=True)

        df_height.to_hdf(os.path.join(height_info_path, 'height_%d_%d--%d.h5'%(index_chunk, np.min(pid_list), np.max(pid_list))), 
                         'data', data_columns=['PatientID'], complevel=5, complib='blosc:lz4')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-version')
    parser.add_argument('-tbl_name')
    parser.add_argument('--index_chunk', type=int, default=None)
    parser.add_argument('--output_to_disk', action='store_true')
    args = parser.parse_args()
    version = args.version
    tbl_name = args.tbl_name
    index_chunk = args.index_chunk
    output_to_disk = args.output_to_disk
    main()

