import numpy as np
import pandas as pd
from os.path import join
from os import listdir
import h5py

bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital'
data_version = 'v6b'
t_unit = np.timedelta64(1,'s')

def get_mimic_test_pids(result_date='181108'):
    """
    Get MIMIC test set pids from the result files produced by Matthias. (Not sure where the original dataset splitting information is saved)
    
    Params:
        result_date (string): date of the latest result directory
    """
    for i in range(5):
        pids = []
        mimic_res = join(bern_path, '8_predictions', result_date, 'reduced', 'held_out', 
                         'WorseStateFromZero_0.0_8.0_shap_top20_variables_MIMIConly_random_%d_lightgbm_full'%i)
        for f in listdir(mimic_res):
            if '.h5' not in f:
                continue
            with h5py.File(join(mimic_res, f)) as store:
                pids.extend([key for key in store.keys()])
        np.save(join(bern_path, 'external_validation', 'misc_derived', 'mimic_retrain_random_%d_test_pids.npy'%i), np.array(pids))

def read_mimic_test_pids(data_split):
    """
    Read the test set pids of the MIMIC data split
    """
    return np.load(join(bern_path, 'external_validation', 'misc_derived', 'mimic_retrain_%s_test_pids.npy'%data_split))

def plot_patient_ts(df, tau, mode='alarm', dt_ref=None, cnt_frame=None):
    if mode not in ['alarm', 'event']:
        raise Exception('Wrong mode.')

    status_plt_setting = dict(InEvent=dict(color='C3',alpha=0.5,label='In Event'),
                              Maybe=dict(color='C3',alpha=0.25,label='Maybe'),
                              ProbNot=dict(color='C9',alpha=0.25,label='Probably Not'),
                              Unknown=dict(color='C7',alpha=0.5,label='Unknown'))
                          
    fig = plt.figure(figsize=(20,4))
    
    hours = df.index/3600
    barw = 5/60
    plt.hlines(tau, 0, hours[-1], color='k', alpha=0.7, linestyle='--', label='$\\tau=%g$'%tau)
    
    for key, val in status_plt_setting.items():
        if df[key].sum() > 0:
            plt.bar(hours[df[key]], 1, barw, 0, 
                    align='edge', color=val['color'], alpha=val['alpha'], label=val['label'])
    if (df.TrueLabel==1).sum() > 0:
        plt.bar(hours[df.TrueLabel==1], 0.1, barw, -.1, 
                align='edge', color='C1', alpha=0.75, label='Label (Deteriorate in [0,8] hrs)')
    if df.OnDrug.sum() > 0:
        plt.bar(hours[df.OnDrug], 0.1, barw, -.1, 
                align='edge', color='C6', alpha=0.75, label='On Drug')
        
    if df.GeqTau.sum() > 0:
        plt.bar(hours[df.GeqTau], .1, barw, 1, 
                align='edge', color='C4', alpha=0.75, label='Score$\\geq\\tau$')
    
    plt.scatter(hours, df.PredScore, alpha=0.7, label='Score', color='C4', marker='.')
    
    if df.Onset.sum() > 0:
        dt_onset = hours[df.Onset]
        plt.scatter(dt_onset, 1.125*np.ones((len(dt_onset),)), label='True onset', color='C3' , marker='x')

    if mode == 'alarm':
        dt_true_alarm = hours[df.TrueAlarm]
        dt_false_alarm = hours[np.logical_and(df.Alarm, ~df.TrueAlarm)]

        if dt_ref is not None:
            dt_true_alarm = dt_true_alarm[dt_true_alarm<=dt_ref]
            dt_false_alarm = dt_false_alarm[dt_false_alarm<=dt_ref]

        plt.scatter(dt_true_alarm, 1.125*np.ones((len(dt_true_alarm),)), label='True Alarm', color='C2' , marker='v')
        plt.scatter(dt_false_alarm, 1.125*np.ones((len(dt_false_alarm),)), label='False Alarm', color='C3' , marker='v')

        if dt_ref is not None:
            plt.axvspan(dt_ref+delta_t, min(dt_ref+(delta_t+window_size), hours[-1]), 
                        color='C7', alpha=0.2, ymin=0, ymax=1, 
                        label='[%g,%g] hrs after alarm'%(delta_t,(delta_t+window_size)))
    else:
        dt_alarm = hours[df.Alarm]
        plt.scatter(dt_alarm, 1.125*np.ones((len(dt_alarm),)), label='Alarm', color='k' , marker='v')
        if dt_ref is not None:
            win_pre_event = df.Alarm[np.logical_and(df.index>=dt_ref*3600-max_sec,df.index<dt_ref*3600-min_sec)]
            plt.axvspan(max(0, dt_ref-(delta_t+window_size)), dt_ref-delta_t, 
                        color='C3' if win_pre_event.sum()>0 else 'C7', alpha=0.2, ymin=0, ymax=1, 
                        label='[%g,%g] hrs before event'%(delta_t,(delta_t+window_size)))

    pid = df.iloc[0].PatientID
    plt.xlabel('Time since admission [hr]')
    plt.title('Patient %d'%pid)
    plt.ylim([-0.15, 1.2])
    plt.legend(bbox_to_anchor=(0., 1.1, 1., .15), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    

def align_time(df_left, df_right, use_mimic=False):
    # adjust the absolute Datetime of the endpoint dataframe to be aligned with those of the prediction dataframe
    idx_ref = 1
    if np.unique(np.diff(df_right.index)/t_unit)[0]>5*60:
        df_right = df_right.resample('5T').pad()
    if df_left.index[0] < df_right.index[0]:
        right_ref=True
    else:
        right_ref=False
    if right_ref:  
        if use_mimic:
            idx_aligned = np.where(((df_left.index - df_right.index[idx_ref])/t_unit)<=0)[0][-1]
        else:
            idx_aligned = np.argmin(np.abs((df_left.index - df_right.index[idx_ref])/t_unit))
        reldt_ref = df_left.iloc[idx_aligned].RelDatetime
        assert(np.abs((df_left.index[idx_aligned]-df_right.index[idx_ref])/t_unit) < 5*60)
        df_right['RelDatetime'] = (df_right.index - df_right.index[idx_ref]) / t_unit + reldt_ref
        df_right['DatetimeAlign'] = [df_left.index[idx_aligned] + (x-reldt_ref)*t_unit for x in df_right.RelDatetime]
    else:
        if use_mimic:
            idx_aligned = np.where(((df_right.index - df_left.index[idx_ref])/t_unit)<=0)[0][-1]
        else:
            idx_aligned = np.argmin(np.abs((df_right.index - df_left.index[idx_ref])/t_unit))
        reldt_ref = df_left.iloc[idx_ref].RelDatetime
        assert(np.abs((df_left.index[idx_ref]-df_right.index[idx_aligned])/t_unit) < 5*60)
        df_right['RelDatetime'] = (df_right.index - df_right.index[idx_aligned]) / t_unit + reldt_ref
        df_right['DatetimeAlign'] = [df_left.index[idx_ref] + (x-reldt_ref)*t_unit for x in df_right.RelDatetime]
    df_right.drop('RelDatetime', axis=1, inplace=True)
    df_right.set_index('DatetimeAlign', inplace=True, drop=True)
    df_left = df_left[df_left.index<=df_right.index[-1]].copy()

    # merge both prediction and endpoint dataframe based on the Datetime
    df = df_right.merge(df_left, how='outer', left_index=True, right_index=True)
    if right_ref:
        df['RelDatetime'] = ((df.index - df_left.index[idx_aligned]) / t_unit).astype(int)
    else:
        df['RelDatetime'] = ((df.index - df_left.index[idx_ref]) / t_unit + reldt_ref).astype(int)
        
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: 'AbsDatetime'}, inplace=True)
    df.set_index('RelDatetime', inplace=True)
    for col in ['Stable', 'InEvent', 'Maybe', 'ProbNot', 'OnDrug']:
        df[col] = df[col].fillna(False)
    df['Unknown'] = df.Unknown.fillna(True)
    df = df[df.index>=0]
    return df

def merge_event_gaps(df, max_gap2merge):
    if df.InEvent.sum() == 0:
        return df

    dt_event = df.index[df.InEvent]
    beg_event = np.concatenate((dt_event[[0]], 
                                dt_event[np.where(np.diff(dt_event)>300)[0]+1]))
    end_event = np.concatenate((dt_event[np.where(np.diff(dt_event)>300)[0]], 
                                dt_event[[-1]])) + 300
    if len(beg_event) == 1:
        return df

    len_gaps = beg_event[1:] - end_event[:-1]

    if np.min(len_gaps) > max_gap2merge * 60:
        return df
    
    for i in np.where(len_gaps<=max_gap2merge * 60)[0]:
        print(i)
        dt_gaps = df.index[np.logical_and(df.index<beg_event[i+1], df.index>=end_event[i])]
        df.loc[dt_gaps, 'InEvent'] = True
        df.loc[dt_gaps, 'PredScore'] = float('NaN')
    return df



