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
parser.add_argument('--recall', type=float, default=0.9)
parser.add_argument('--use_subsample', action='store_true')
parser.add_argument('--use_test_set', type=int, default=None)

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
recall = args.recall

use_subsample = args.use_subsample
use_test_set = args.use_test_set

# model = 'lightgbm'
# result_date = '181108'
# result_dir = 'shap_top20_variables_MIMIC'
# result_split = 'held_out'

# delta_t = 0
# window_size = 480
# t_silence = 30
# max_gap2merge = 0
# reset_time = 25
# recall = 0.9
# use_subsample = False
# use_test_set = 0

ts = t_silence * 60
max_sec = (delta_t+window_size) * 60
min_sec = delta_t * 60

bern_path = utils.bern_path
data_version = utils.data_version

use_mimic = True if 'mimic' in result_dir.lower() else False
print(use_mimic)

if result_dir == 'shap_top500_features':
    threshold_dict = dict(temporal_1={0.8: 0.857236, 0.9: 0.679142, 0.95: 0.401373 },
                          temporal_2={0.8: 0.838625, 0.9: 0.626938, 0.95: 0.281699},
                          temporal_3={0.8: 0.856067, 0.9: 0.648161, 0.95: 0.281699},
                          temporal_4={0.8: 0.831848, 0.9: 0.621108, 0.95: 0.200687}, 
                          temporal_5={0.8: 0.846359, 0.9: 0.587809, 0.95: 0.200687}, 
                          held_out={0.8: 0.861820, 0.9: 0.720456, 0.95: 0.371314})

elif result_dir == 'shap_top20_variables':
    threshold_dict = {'temporal_1': {0.8: 0.8808215067473738,
                                     0.9: 0.7174294481276855,
                                     0.95: 0.44747422694073546},
                      'temporal_2': {0.8: 0.8606877876372363,
                                     0.9: 0.6431396419047642,
                                     0.95: 0.2816993466714189},
                      'temporal_3': {0.8: 0.8695086778590647,
                                     0.9: 0.6791421659802078,
                                     0.95: 0.2816993466714189},
                      'temporal_4': {0.8: 0.8572362772695626,
                                     0.9: 0.6978072283874117,
                                     0.95: 0.20068666377598746},
                      'temporal_5': {0.8: 0.8684350153803698,
                                     0.9: 0.6530137974403645,
                                     0.95: 0.20068666377598746},
                      'held_out': {0.8: 0.8525024006352193,
                                   0.9: 0.7426289015378912,
                                   0.95: 0.44747422694073546}}

elif 'shap_top20_variables_MIMIConly' in result_dir:
    threshold_dict = {'temporal_1': {0.8: 0.8438263056807518,
                                     0.9: 0.7079505469890286,
                                     0.95: 0.5227339080223317},
                      'temporal_2': {0.8: 0.7957966087795766,
                                     0.9: 0.4262512003176097,
                                     0.95: 0.20068666377598746},
                      'temporal_3': {0.8: 0.8290461251590622,
                                     0.9: 0.648160890716723,
                                     0.95: 0.4659800028906792},
                      'temporal_4': {0.8: 0.7291735736121544,
                                     0.9: 0.4013733275519749,
                                     0.95: 0.20068666377598746},
                      'temporal_5': {0.8: 0.7788199112828432,
                                     0.9: 0.5720011145449331,
                                     0.95: 0.2816993466714189},
                      'held_out': {0.8: 0.6906273357687085,
                                   0.9: 0.4262512003176097,
                                   0.95: 0.2816993466714189}}
elif result_dir == 'shap_top20_variables_MIMIC':
    if use_test_set == 0:
        threshold_dict = {'temporal_1': {0.8: 0.8141599230214829,
                                         0.9: 0.7347066608853082,
                                         0.95: 0.6431396419047642},
                          'temporal_2': {0.8: 0.8044332546937506,
                                         0.9: 0.7012679069853189,
                                         0.95: 0.5104929723474184},
                          'temporal_3': {0.8: 0.7940056808582895,
                                         0.9: 0.6622572447554149,
                                         0.95: 0.5227339080223317},
                          'temporal_4': {0.8: 0.7921923190188374,
                                         0.9: 0.6577092845332329,
                                         0.95: 0.4262512003176097},
                          'temporal_5': {0.8: 0.7426289015378912,
                                         0.9: 0.5633986933428379,
                                         0.95: 0.33333333333333337},
                          'held_out': {0.8: 0.8386246145705598,
                                       0.9: 0.7079505469890286,
                                       0.95: 0.5542526105605248}}

    if use_test_set == 1:
        threshold_dict = {'temporal_1': {0.8: 0.7957966087795766,
                                         0.9: 0.6868992801178706,
                                         0.95: 0.5227339080223317},
                          'temporal_2': {0.8: 0.7847028130491337,
                                         0.9: 0.6431396419047642,
                                         0.95: 0.4971205646114242},
                          'temporal_3': {0.8: 0.7975656510865553,
                                         0.9: 0.6622572447554149,
                                         0.95: 0.5227339080223317},
                          'temporal_4': {0.8: 0.759584533650943,
                                         0.9: 0.6086916009002754,
                                         0.95: 0.4659800028906792},
                          'temporal_5': {0.8: 0.7319665508030779,
                                         0.9: 0.5633986933428379,
                                         0.95: 0.4013733275519749},
                          'held_out': {0.8: 0.8318481125154829,
                                       0.9: 0.7012679069853189,
                                       0.95: 0.5801208964980813}}
    if use_test_set == 2:
        threshold_dict = {'temporal_1': {0.8: 0.8372944536596248,
                                         0.9: 0.7347066608853082,
                                         0.95: 0.595109945003589},
                          'temporal_2': {0.8: 0.7975656510865553,
                                         0.9: 0.6868992801178706,
                                         0.95: 0.5878093311876458},
                          'temporal_3': {0.8: 0.7940056808582895,
                                         0.9: 0.6906273357687085,
                                         0.95: 0.5720011145449331},
                          'temporal_4': {0.8: 0.7662843588032355,
                                         0.9: 0.6086916009002754,
                                         0.95: 0.5227339080223317},
                          'temporal_5': {0.8: 0.7319665508030779,
                                         0.9: 0.5801208964980813,
                                         0.95: 0.3713144507689456},
                          'held_out': {0.8: 0.8438263056807518,
                                       0.9: 0.7234205717983191,
                                       0.95: 0.6379379507945723}}
    if use_test_set == 3:
        threshold_dict = {'temporal_1': {0.8: 0.7768045911163969,
                                         0.9: 0.6431396419047642,
                                         0.95: 0.44747422694073546},
                          'temporal_2': {0.8: 0.7451761489691829,
                                         0.9: 0.5801208964980813,
                                         0.95: 0.3713144507689456},
                          'temporal_3': {0.8: 0.7549392743365122,
                                         0.9: 0.595109945003589,
                                         0.95: 0.4262512003176097},
                          'temporal_4': {0.8: 0.7204560007449916,
                                         0.9: 0.5444894851931955,
                                         0.95: 0.4013733275519749},
                          'temporal_5': {0.8: 0.6751019550882568,
                                         0.9: 0.4659800028906792,
                                         0.95: 0.33333333333333337},
                          'held_out': {0.8: 0.8027466551039498,
                                       0.9: 0.6530137974403645,
                                       0.95: 0.5340199971093208}}

    if use_test_set == 4:
        threshold_dict = {'temporal_1': {0.8: 0.8093782646762628,
                                         0.9: 0.6868992801178706,
                                         0.95: 0.5104929723474184},
                          'temporal_2': {0.8: 0.7684504564822081,
                                         0.9: 0.6150326800047523,
                                         0.95: 0.4659800028906792},
                          'temporal_3': {0.8: 0.7768045911163969,
                                         0.9: 0.6086916009002754,
                                         0.95: 0.4262512003176097},
                          'temporal_4': {0.8: 0.7808075602740688,
                                         0.9: 0.5633986933428379,
                                         0.95: 0.4013733275519749},
                          'temporal_5': {0.8: 0.7400360293466851,
                                         0.9: 0.5104929723474184,
                                         0.95: 0.2816993466714189},
                          'held_out': {0.8: 0.8202992809188493,
                                       0.9: 0.6830726742233938,
                                       0.95: 0.5444894851931955}}

if model == 'lstm':
    res_path = join(bern_path, '8_predictions', 'lstm', data_version, 'reduced', result_split)
else:
    res_path = join(bern_path, '8_predictions', result_date, 'reduced', result_split)
res_path = join(res_path, result_dir if model=='lstm' else 'WorseStateFromZero_0.0_8.0_%s_lightgbm_full'%result_dir)

if use_mimic:
    ep_path = join(bern_path, 'external_validation', 'endpoints', '181103', 'reduced') 
else:
    ep_path = join(bern_path, '3a_endpoints', data_version, 'reduced')


prob_files = [f for f in listdir(res_path) if '.h5' in f]
tau = threshold_dict[result_split][recall]

num_event = 0

if use_mimic:
    if 'MIMICOnly' in result_dir:
        subsample_split_path = join(bern_path, 'external_validation', 'misc_derived', 
                                    'MIMIC_prevalence_subsample_%s_rseed2017.pickle'%('_'.join(result_dir.split('_')[4:])))
    else:
        subsample_split_path = join(bern_path, 'external_validation', 'misc_derived', 
                                    'MIMIC_prevalence_subsample_held_out_rseed2017.pickle')
        
    with open(subsample_split_path, 'rb') as f:
        subsample_pids = pickle.load(f)

if use_mimic:
    drugid = ['pm41', 'pm42', 'm_pm_2', 'm_pm_1', 'pm39', 'pm40', 'pm45']
else:
    drugid = ['pm%d'%pmid for pmid in [41, 42, 43, 44, 39, 40, 45]] 

if use_mimic and not use_subsample:
    result_dir += '_no_subsample'

out_path = join(bern_path, 'circews_analysis', 'alarm_score_for_calibration', 
                data_version)

if use_test_set is None:
    out_path = join(out_path, result_dir, result_split)
else:
    out_path = join(out_path, result_dir+'_use_retrain_set_%d'%use_test_set, result_split)


if not exists(out_path):
    makedirs(out_path)


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
    df_ep = df_ep[['PatientID', 'Datetime', 'endpoint_status']+drugid]
    
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
        df_prob_tmp = pd.read_hdf(join(res_path, f), pid, mode='r')
        

        if model == 'lightgbm':
            df_prob_tmp = df_prob_tmp[['AbsDatetime', 'PredScore','TrueLabel', 'RelDatetime']]
            df_prob_tmp.set_index('AbsDatetime', inplace=True)
        else:
            df_prob_tmp['RelDatetime'] = (df_prob_tmp.index - df_prob_tmp.index[0]) / t_unit
            
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


        # Compute true alarms and false alarms
        df['GeqTau'] = (df.PredScore >= tau)
        df['Alarm'] = False
        df['TrueAlarm'] = False
        if df.GeqTau.sum() > 0:
            dt_geq_tau = df.index[df.GeqTau]

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
        if df.Alarm.sum() == 0:
            continue
        df_status = df[np.logical_or(df.Alarm, df.Onset)].copy()
        df_status['Status'] = df_status.Alarm.apply(lambda x: 'Alarm' if x else 'CF_Onset')
        df_status['IsAlarmTrue'] = df_status.TrueAlarm.apply(lambda x: 'Yes' if x else 'No')
        df_status.loc[df_status.index[df_status.PredScore.isnull()], 'IsAlarmTrue'] = np.nan
        df_status = df_status[[ 'AbsDatetime', 'Status', 'IsAlarmTrue', 'PredScore']]
        df_status.reset_index(inplace=True)
        df_status.to_hdf(join(out_path, f), pid, complib='blosc:lz4', complevel=5, format='table')
    gc.collect()

