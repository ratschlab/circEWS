import matplotlib
textsize = 12
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['font.size'] = textsize
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.labelsize'] = textsize
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = textsize
matplotlib.rcParams['xtick.labelsize'] = textsize
matplotlib.rcParams['ytick.labelsize'] = textsize
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['legend.fontsize'] = textsize
import matplotlib.pyplot as plt
import seaborn as sns
current_palette = sns.color_palette()

import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import join, exists
import gc
import pickle
import h5py
import utils 

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--result_date', default='181108')
parser.add_argument('--result_dir', default='shap_top500_features')
parser.add_argument('--result_split', default='held_out')

parser.add_argument('--model', choices=['lightgbm', 'lstm'], default='lightgbm')
parser.add_argument('--delta_t', type=int, default=0)
parser.add_argument('--window_size', type=int, default=480)
parser.add_argument('--t_silence', type=int, default=30)
parser.add_argument('--max_gap2merge', type=int, default=0)
parser.add_argument('--reset_time', type=int, default=25)

parser.add_argument('--random', action='store_true')
parser.add_argument('--kars_baseline', action='store_true')
parser.add_argument('--use_subsample', action='store_true')
parser.add_argument('--use_test_set', type=int, default=None)
parser.add_argument('--idx_threshold', type=int, default=0)

args = parser.parse_args()

model = args.model
result_date = args.result_date
result_dir = args.result_dir
result_split = args.result_split

delta_t = args.delta_t
window_size = args.window_size
t_silence = args.t_silence
max_gap2merge = args.max_gap2merge
reset_time = args.reset_time

random = args.random
kars_baseline = args.kars_baseline
use_subsample = args.use_subsample
use_test_set = args.use_test_set
idx_threshold = args.idx_threshold

# model = 'lightgbm'
# result_date = '181108'
# result_dir = 'Baseline_Clinical'
# result_split = 'held_out'

# delta_t = 0
# window_size = 480
# t_silence = 30
# max_gap2merge = 0
# reset_time = 25

# random = False
# kars_baseline = True
# use_subsample = False

ts = t_silence * 60
max_sec = (delta_t+window_size) * 60
min_sec = delta_t * 60
t_unit = np.timedelta64(1,'s')

bern_path = utils.bern_path
data_version = utils.data_version

use_mimic = True if 'mimic' in result_dir.lower() else False
if 'MIMIC_BERN' in result_dir:
    use_mimic = False


if model == 'lstm':
    res_path = join(bern_path, '8_predictions', 'lstm', data_version, 'reduced', result_split)
else:
    res_path = join(bern_path, '8_predictions', result_date, 'reduced', result_split)

if kars_baseline:
    res_path = join(res_path, 'WorseStateFromZero_0.0_8.0_shap_top500_features_lightgbm_full')
else:
    res_path = join(res_path, result_dir if model=='lstm' else 'WorseStateFromZero_0.0_8.0_%s_lightgbm_full'%result_dir)
                

if use_mimic:
    ep_path = join(bern_path, 'external_validation', 'endpoints', '181103', 'reduced') 
else:
    ep_path = join(bern_path, '3a_endpoints', data_version, 'reduced')


prob_files = [f for f in listdir(res_path) if '.h5' in f]
need_calib = 'baseline' in result_dir.lower()
if need_calib:
    print('Compute maxval and minval for calibration')
    if kars_baseline:
        
        bl_res_path = join(bern_path, 'Baseline', data_version, 'predictions') # baseline result path
        bl_res_file = (('' if result_dir=='Baseline' else '%s_'%result_dir) + 'Predictions' + 
                       ('' if result_split == 'held_out' else '_%s'%result_split.split('_')[1]) + '.h5')

        bl_res_path = join(bl_res_path, bl_res_file)
            
        with h5py.File(bl_res_path, 'r') as pstore:
            pids = [key for key in pstore.keys()]

        min_pred_score = float('Inf')
        max_pred_score = 0

        for i, pid in enumerate(pids):
            predscore = pd.read_hdf(bl_res_path, pid, mode='r')['PredScore']
            min_pred_score = min(min_pred_score, predscore.min())
            max_pred_score = max(max_pred_score, predscore.max())
            if (i+1)%100 == 0:
                print('%d patients have been processed.'%(i+1))

        print('Minval: %g, Maxval: %g'%(min_pred_score, max_pred_score))
    else:

        min_pred_score = float('Inf')
        max_pred_score = 0

        i = 0
        for f in prob_files:
            with h5py.File(join(res_path, f), 'r') as pstore:
                pids = [key for key in pstore.keys()]

            for pid in pids:
                predscore = pd.read_hdf(join(res_path, f), pid, mode='r')['PredScore']
                min_pred_score = min(min_pred_score, predscore.min())
                max_pred_score = max(max_pred_score, predscore.max())
                if (i+1)%100 == 0:
                    print('%d patients have been processed.'%(i+1))
                i += 1

    # thresholds = np.linspace(0,1,num=201)
    thresholds = np.log(np.linspace(1, 1001, num=501).tolist()) / np.log(1000)
    # thresholds = np.log(np.linspace(1, 1003, num=335)) / np.log(1000)
    thresholds = min_pred_score + thresholds*(max_pred_score-min_pred_score)
    # thresholds = np.concatenate((thresholds, [thresholds[-1]+np.diff(thresholds)[0]]))
else:  
    # thresholds = np.log(np.linspace(1, 1001, num=501).tolist()) / np.log(1000)
    thresholds = np.log(np.linspace(1, 1003, num=335)) / np.log(1000)
    # thresholds = thresholds[[idx_threshold]]


    
FA = {'tau_%g'%tau: [] for tau in thresholds}
TA = {'tau_%g'%tau: [] for tau in thresholds}
CE = {'tau_%g'%tau: [] for tau in thresholds}
ME = {'tau_%g'%tau: [] for tau in thresholds}

if random:
    np.random.seed(2018)

if use_mimic:
    if 'MIMICOnly' in result_dir:
        subsample_split_path = join(bern_path, 'external_validation', 'misc_derived', 
                                    'MIMIC_prevalence_subsample_%s_rseed2017.pickle'%('_'.join(result_dir.split('_')[4:])))
    else:
        subsample_split_path = join(bern_path, 'external_validation', 'misc_derived', 
                                    'MIMIC_prevalence_subsample_held_out_rseed2017.pickle')
        
    with open(subsample_split_path, 'rb') as f:
        subsample_pids = pickle.load(f)


num_event = 0
cnt_LOS = 0
cnt_LOE = 0
cnt_LOStable = 0

if use_mimic:
    drugid = ['pm41', 'pm42', 'm_pm_2', 'm_pm_1', 'pm39', 'pm40', 'pm45']
else:
    drugid = ['pm%d'%pmid for pmid in [41, 42, 43, 44, 39, 40, 45]] 

for f in prob_files:

    # read prediction dataframes
    with h5py.File(join(res_path, f), 'r') as pstore:
        pids = [key for key in pstore.keys()]
        if use_mimic and use_subsample:
            pids = [pid for pid in pids if int(pid[1:]) in subsample_pids]


        if use_mimic and use_test_set is not None:
            subsample_pids = utils.read_mimic_test_pids('random_%s'%use_test_set)
            pids = [pid for pid in pids if pid in subsample_pids]
    
    # read endpoint dataframes
    ep_f = [ff for ff in listdir(ep_path) if '_%s_'%f.split('_')[-1][:-3] in ff][0]
    df_ep = pd.read_hdf(join(ep_path, ep_f), mode='r')
    df_ep = df_ep[['PatientID', 'Datetime', 'endpoint_status'] + drugid]
    
    df_ep['Stable'] = df_ep.endpoint_status=='event 0'
    df_ep['InEvent'] = df_ep.endpoint_status.isin(['event 1', 'event 2', 'event 3'])
    df_ep['Maybe'] = df_ep.endpoint_status.isin(['maybe 1', 'maybe 2', 'maybe 3'])
    df_ep['ProbNot'] = df_ep.endpoint_status.isin(['probably not 1', 'probably not 2', 'probably not 3'])
    df_ep['Unknown'] = df_ep.endpoint_status=='unknown'
    df_ep['OnDrug'] = (df_ep[drugid]>0).sum(axis=1) > 0
    df_ep.drop(['endpoint_status']+drugid, axis=1, inplace=True)

    for pid in pids:
        # get the endpoint dataframe of the current patient and set absolute Datetime as index (for alignment)
        pid_num = int(pid[1:]) if pid[0]=='p' else int(pid)
        
        df_ep_tmp = df_ep[df_ep.PatientID==pid_num].copy()    
        df_ep_tmp.set_index('Datetime', inplace=True)
        
        # get the prediction datafram of the current patient and set absolute Datetime as index (for alignment)
        if kars_baseline:
            df_prob_tmp  = pd.read_hdf(bl_res_path, pid[1:], mode='r')                
        else:
            df_prob_tmp = pd.read_hdf(join(res_path, f), pid, mode='r')
        

        if model == 'lightgbm':
            df_prob_tmp = df_prob_tmp[['AbsDatetime', 'PredScore','TrueLabel', 'RelDatetime']]
            df_prob_tmp.set_index('AbsDatetime', inplace=True)
        else:
            df_prob_tmp['RelDatetime'] = (df_prob_tmp.index - df_prob_tmp.index[0]) / t_unit
            
        if random:
            idx_notnull = df_prob_tmp.index[df_prob_tmp.PredScore.notnull()]
            df_prob_tmp.loc[idx_notnull, 'PredScore'] = np.random.rand(len(idx_notnull))
            
        df = utils.align_time(df_prob_tmp, df_ep_tmp, use_mimic=use_mimic)
        del df_ep_tmp, df_prob_tmp
        gc.collect() 

        if max_gap2merge > 0:
            df = utils.merge_event_gaps(df, max_gap2merge)
            
        df['Onset'] = False
        if df.InEvent.sum() > 0:
            dt_event = df.index[df.InEvent]
            beg_event = np.concatenate((dt_event[[0]], 
                                        dt_event[np.where(np.diff(dt_event)>300)[0]+1]))
            beg_event = beg_event[beg_event>min_sec]
            df.loc[beg_event, 'Onset'] = True
            for dt_onset in beg_event:
                win_pre_event = df[np.logical_and(df.index>=dt_onset-max_sec,df.index<dt_onset-min_sec)]
                if len(win_pre_event) in [0, win_pre_event.PredScore.isnull().sum()]:
                    # if prior to the event, the status are unknown and there is no prediction score,
                    # this event is listed unpredictable at all, hence delete it from the Onset list
                    df.loc[dt_onset, 'Onset'] = False
                    
        num_event += df.Onset.sum()
        cnt_LOS += len(df)
        cnt_LOE += df.InEvent.sum()
        cnt_LOStable += df.Stable.sum()

        for i, tau in enumerate(thresholds):

            # Compute true alarms and false alarms
            df['GeqTau'] = (df.PredScore >= tau)
            df['Alarm'] = False
            df['TrueAlarm'] = False
            if df.GeqTau.sum() > 0:
                dt_geq_tau = df.index[df.GeqTau]
                if t_silence == 5 and reset_time == 0:
                    df.loc[dt_geq_tau,'Alarm'] = True                    
                else:
                    beg_geq_tau = np.concatenate((dt_geq_tau[[0]], 
                                                  dt_geq_tau[np.where(np.diff(dt_geq_tau)>300)[0]+1]))
                    end_geq_tau = np.concatenate((dt_geq_tau[np.where(np.diff(dt_geq_tau)>300)[0]], 
                                                  dt_geq_tau[[-1]])) + 300

                    dt = 0
                    reset = False

                    while dt < beg_geq_tau[-1]: 
                        dt = beg_geq_tau[beg_geq_tau>=dt][0]
                        reset = False

                        while not reset:

                            if dt in df.index and df.loc[dt, 'GeqTau']:

                                dist2event = dt - df.index[df.InEvent].values
                                if np.sum( dist2event >= 0 ) == 0:
                                    df.loc[dt, 'Alarm'] = True
                                    dt += ts
                                else:
                                    min_dist2event = np.min(dist2event[dist2event >= 0])
                                    if min_dist2event >= reset_time*60:
                                        df.loc[dt, 'Alarm'] = True
                                        dt += ts
                                    else:
                                        dt = dt + reset_time*60 - min_dist2event
                            else:
                                reset = True

                for dt_alarm in df.index[df['Alarm']]:
                    win_post_alarm = df[np.logical_and(df.index<=dt_alarm+max_sec,df.index>dt_alarm+min_sec)]
                    if len(win_post_alarm) in [0, win_post_alarm.Unknown.sum()]:
                        # if there is no more time after the alarm or the status are completely unknown, we 
                        # consider the alarm neither true or false, hence disable it. 
                        df.loc[dt_alarm, 'Alarm'] = False
                    else:
                        df.loc[dt_alarm, 'TrueAlarm'] = win_post_alarm.InEvent.sum()>0
                #     utils.plot_patient_ts(df, tau, mode='alarm', dt_ref=dt_alarm/3600)
                # utils.plot_patient_ts(df, tau, mode='alarm')
                    

            # Compute captured events and missed events
            df['CatchedOnset'] = False
            if df.Onset.sum() > 0:
                for dt_onset in df.index[df.Onset]:
                    win_pre_event = df[np.logical_and(df.index>=dt_onset-max_sec,df.index<dt_onset-min_sec)]
                    df.loc[dt_onset, 'CatchedOnset'] = win_pre_event.Alarm.sum() > 0
                #     utils.plot_patient_ts(df, tau, mode='event', dt_ref=dt_onset/3600)
                # utils.plot_patient_ts(df, tau, mode='event')

            TA['tau_%g'%tau].append([pid, df.TrueAlarm.sum()])
            FA['tau_%g'%tau].append([pid, df.Alarm.sum() - df.TrueAlarm.sum()])
            CE['tau_%g'%tau].append([pid, df.CatchedOnset.sum()])
            ME['tau_%g'%tau].append([pid, df.Onset.sum() - df.CatchedOnset.sum()])
        gc.collect()
    gc.collect()
    
cnts = []
for tau in thresholds:
    key = 'tau_%g'%tau
    
    total_TA = np.sum([x[1] for x in TA[key]])
    total_FA = np.sum([x[1] for x in FA[key]])
    total_CE = np.sum([x[1] for x in CE[key]])
    total_ME = np.sum([x[1] for x in ME[key]])

    cnts.append([tau, total_TA, total_FA, total_CE, total_ME])
    
cnts = pd.DataFrame(cnts, columns=['tau','TA', 'FA', 'CE', 'ME'])

if use_mimic and not use_subsample:
    result_dir += '_no_subsample'

out_path = join(bern_path, 'circews_analysis', 'alarm_event_cnts', 
                data_version, 'merged_%g_reset_%g'%(max_gap2merge, reset_time))

if use_test_set is None:
    out_path = join(out_path, result_dir, result_split)
else:
    out_path = join(out_path, result_dir+'_use_retrain_set_%d'%use_test_set, result_split)

if not exists(out_path):
    makedirs(out_path)

cnts.to_csv(join(out_path, ('rand_' if random else '')+'dt-%g_ws-%g_ts-%g_cnts.csv'%(delta_t, window_size, t_silence)), index=False)
np.savez(join(out_path, 'for_event_prevalence.npz'), 
         num_event=num_event, cnt_LOS=cnt_LOS, 
         cnt_LOE=cnt_LOE, cnt_LOStable=cnt_LOStable)
