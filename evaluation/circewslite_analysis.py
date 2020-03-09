import pandas as pd
import numpy as np
from os.path import join, exists, split
from os import mkdir, makedirs, listdir
import gc
import matplotlib.pyplot as plt
import seaborn
from copy import deepcopy
from time import time
import pickle


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('split_name')
parser.add_argument('f')
parser.add_argument('--t_reset',type=int, default=25)
args = parser.parse_args()
split_name = args.split_name
f = args.f
t_reset = args.t_reset


# split_name = 'temporal_4'                                                                                                                 
# f = 'batch_38.h5'                                                                                                                         
# t_reset = 25

fixed_rec = 0.9
model_name = 'shap_top500_features'

delta_t = 0
window_size = 480
data_version = 'v6b'
result_version = '181108'
t_postevent = np.timedelta64(2, 'h')
wsize_upper_h = (window_size+delta_t) * np.timedelta64(1, 'm')
wsize_lower_h = delta_t * np.timedelta64(1, 'm')

bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'
alarm_path = join(bern_path, 'circews_analysis', 'alarm_score_for_calibration_new',
                  data_version, 'merged_0_reset_%d'%t_reset, model_name, split_name)

ep_path = join(bern_path, '3a_endpoints', data_version,'reduced')
res_dir = lambda s: 'WorseStateFromZero_0.0_8.0_%s_lightgbm_full'%s
pred_path =  join(bern_path, '8_predictions', result_version, 'reduced', 
                  split_name, res_dir(model_name))


with pd.HDFStore(join(pred_path, f), mode='r') as tmp:
    pids = [int(key[2:]) for key in tmp.keys()]
    gc.collect()


df_store = pd.HDFStore(join(pred_path, f), mode='r')
pids = [int(key[2:]) for key in df_store.keys()]
df_store.close()
gc.collect()

stats = dict()
lst_period_type = ['critical_window', 'maintenance_window', 'uncritical_window', 'patients_wo_events']
for period_type in lst_period_type:
    stats.update({period_type: dict(valid_los=[], cnt_alarm=[], los=[])})
    stats.update(cnt_catched_event=0, cnt_missed_event=0)

is_critical_win = lambda t, ts: np.logical_and(ts< t-wsize_lower_h, 
                                               ts>=t-wsize_upper_h)

is_maintenance_win = lambda t, ts: np.logical_and(ts>t, 
                                                  ts<=t+t_postevent) 

is_uncritical_win = lambda t, ts, mode: ts<t-wsize_upper_h if mode=='before' else ts>t+t_postevent

t_start = time()
for n, pid in enumerate(pids):
    df =  pd.read_hdf(join(alarm_path, f), 'p%d'%pid).reset_index()
    # df =  pd.read_hdf(join(alarm_path, 'rec_%g'%fixed_rec+f), 'p%d'%pid).reset_index()
    df.set_index('AbsDatetime', inplace=True)
    for col in ['InEvent', 'Stable']:
        df.loc[:,col] = df[col].astype(int)
        
    if df.InEvent.sum()==0:
        # assert('Yes' not in df.IsAlarmTrue.unique())
        stats['patients_wo_events']['valid_los'].append( df.Stable.sum() / 12)
        stats['patients_wo_events']['cnt_alarm'].append( df.Alarm.sum() )
        stats['patients_wo_events']['los'].append( len(df)/12 )
    else:
        stable_sum = 0
        beg_onset = df.index[np.where(np.array([0]+np.diff(df.InEvent.values).tolist())==1)[0]]
        end_onset = df.index[np.where(np.diff(df.InEvent.values)==-1)[0]]
        if df.iloc[0].InEvent==1:
            beg_onset = np.concatenate([[df.index[0]], beg_onset])
        if df.iloc[-1].InEvent==1:
            end_onset = np.concatenate([end_onset, [df.index[-1]]])
        assert(len(beg_onset)==len(end_onset))

        ### Critical window
        for i, dt in enumerate(beg_onset):
            dt = np.datetime64(dt)
            win_pre_event = df[is_critical_win(dt, df.index.values)]
            if len(win_pre_event)==0:
                continue

            if len(win_pre_event)==0 or win_pre_event.Stable.sum()==0:
                continue
                
            if ~ df.loc[dt,'Onset']:
                pass
            elif win_pre_event.Alarm.sum()>0 and df.loc[dt,'Onset'] and df.loc[dt,'CatchedOnset']:
                stats['cnt_catched_event'] += 1
            elif win_pre_event.Alarm.sum()==0 and df.loc[dt,'Onset'] and ~df.loc[dt,'CatchedOnset']:
                stats['cnt_missed_event'] += 1
            else:
                print('Alarm number', win_pre_event.Alarm.sum(),'; Onset status', df.loc[dt, 'CatchedOnset'])
                print(dt)
                raise Exception('Problem!!!!')
                
            if i > 0:
                win_pre_event = win_pre_event[win_pre_event.index>end_onset[i-1]]

            stable_sum += win_pre_event.Stable.sum() / 12
            stats['critical_window']['valid_los'].append( win_pre_event.Stable.sum() / 12 )
            stats['critical_window']['los'].append( len(df)/12 )
            stats['critical_window']['cnt_alarm'].append( win_pre_event.Alarm.sum() )

        ### Uncritical window
        for i, dt in enumerate(beg_onset):
            dt = np.datetime64(dt)
            win_pre_event = df[is_uncritical_win(dt, df.index.values, 'before')]
            if len(win_pre_event)==0:
                continue
            if i > 0:
                win_pre_event = win_pre_event[win_pre_event.index>end_onset[i-1]+t_postevent]
        
            if len(win_pre_event)==0 or win_pre_event.Stable.sum()==0:
                continue

            stable_sum += win_pre_event.Stable.sum() / 12
            stats['uncritical_window']['valid_los'].append(win_pre_event.Stable.sum() / 12)
            stats['uncritical_window']['los'].append( len(df)/12 )
            stats['uncritical_window']['cnt_alarm'].append(win_pre_event.Alarm.sum())
            
        
        win_post_last_event = df[is_uncritical_win(np.datetime64(end_onset[-1]),df.index.values,'after')]
        stable_sum += win_post_last_event.Stable.sum() / 12
        stats['uncritical_window']['valid_los'].append(win_post_last_event.Stable.sum() / 12)
        stats['uncritical_window']['los'].append( len(df)/12 )
        stats['uncritical_window']['cnt_alarm'].append(win_post_last_event.Alarm.sum())

        ### Maintenance window
        for i, dt in enumerate(end_onset):
            dt = np.datetime64(dt)
            win_post_event = df[is_maintenance_win(dt, df.index.values)]
            if len(win_post_event)==0:
                continue
            if i < len(beg_onset) - 1:
                win_post_event = win_post_event[win_post_event.index<beg_onset[i+1]-wsize_upper_h]
            if len(win_post_event)==0 or win_post_event.Stable.sum()==0:
                continue
            stable_sum += win_post_event.Stable.sum() / 12
            stats['maintenance_window']['valid_los'].append(win_post_event.Stable.sum() / 12)
            stats['maintenance_window']['los'].append( len(df)/12 )
            stats['maintenance_window']['cnt_alarm'].append(win_post_event.Alarm.sum())

        assert(np.abs(df.Stable.sum()/12-stable_sum)<1e-10)
    if (n+1)%10==0:
        print('Process %d patients, time: %4.4g sec'%(n+1, time()-t_start))
        gc.collect()
        

# with open(join(alarm_path, 'rec_%g'%fixed_rec+f.replace('.h5', '.pkl')), 'wb') as tmp:
#     pickle.dump(stats, tmp)
with open(join(alarm_path, f.replace('.h5', '.pkl')), 'wb') as tmp:
    pickle.dump(stats, tmp)




