import pandas as pd
import numpy as np
from os.path import join, exists, split
from os import mkdir, makedirs, listdir
import gc
import matplotlib.pyplot as plt
import seaborn
from time import time
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('split_name')
parser.add_argument('f')
args = parser.parse_args()
split_name = args.split_name
f = args.f

# split_name = 'temporal_5'
# f = 'batch_42.h5'

metaid_oor_mapping = dict(vm136=[2.2, float('inf')],
                                vm146=[2.2, float('inf')],
                                vm5=[150, 65],
                                vm1=[110, 60],
                                pm41='Dobutamine',
                                pm42='Milrinone',
                                vm13=[-float('inf'), 4],
                                vm28=[2,-2],
                                vm172=[1.2, float('inf')],
                                vm174=[7.8, 4],
                                vm176=[10, float('inf')],
                                vm4=[140, 40],
                                vm62=[30, float('inf')],
                                vm3=[200, 90],
                                vm20=[-float('inf'), 90])


metaid_name_mapping = dict(vm136='a-Lactate',
                           vm146='v-Lactate',
                           vm5='ABP mean (invasive)',
                           vm1='Heart rate',
                           pm41='Dobutamine',
                           pm42='Milrinone',
                           vm13='Cardiac output',
                           vm28='RASS',
                           vm172='INR',
                           vm174='Blood glucose',
                           vm176='C-reactive protein',
                           vm4='ABP diastolic (invasive)',
                           vm62='Peak inspiratory pressure (ventilator)',
                           vm3='ABP systolic (invasive)',
                           vm20='SpO2')

del metaid_oor_mapping['pm41'], metaid_oor_mapping['pm42']
del metaid_name_mapping['pm41'], metaid_name_mapping['pm42']

delta_t = 0
window_size = 480
data_version = 'v6b'
result_version = '181108'
t_postevent = np.timedelta64(2,'h')
wsize_upper_h = (window_size+delta_t) * np.timedelta64(1,'m')
wsize_lower_h = delta_t * np.timedelta64(1,'m')



bern_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'
data_path  = join(bern_path,'3_merged', data_version,'reduced')
ep_path = join(bern_path,'3a_endpoints', data_version,'reduced')
res_dir = lambda s: 'WorseStateFromZero_0.0_8.0_%s_lightgbm_full'%s
pred_path =  join(bern_path,'8_predictions', result_version,'reduced', 
                  split_name, res_dir('shap_top20_variables_MIMIC_BERN'))

out_path = join(bern_path,'circews_analysis','simple_alarm', split_name)
if not exists(out_path):
    mkdir(out_path)


with pd.HDFStore(join(pred_path, f), mode='r') as tmp:
    pids = [int(key[2:]) for key in tmp.keys()]
gc.collect()

lst_vmid = [key for key in metaid_name_mapping.keys()]
lst_period_type = ['critical_window', 'maintenance_window', 'uncritical_window', 'patients_wo_events']
stats = dict()
for vmid in lst_vmid + ['any']:
    tmp = dict()
    for period_type in lst_period_type:
        tmp.update({period_type: dict(valid_los=[], cnt_alarm=[], los=[])})
    tmp.update(cnt_catched_event=0, cnt_missed_event=0)
    stats.update({vmid: tmp})
    
is_critical_win = lambda t, ts: np.logical_and(ts< t-wsize_lower_h, 
                                               ts>=t-wsize_upper_h)

is_maintenance_win = lambda t, ts: np.logical_and(ts>t, 
                                                  ts<=t+t_postevent) 

is_uncritical_win = lambda t, ts, mode: ts<t-wsize_upper_h if mode=='before' else ts>t+t_postevent

is_win_pos_alarm = lambda t, ts: np.logical_and(ts> t+wsize_lower_h, 
                                                ts<=t+wsize_upper_h)

t_start = time()
for n, pid in enumerate(pids):    
    ff = [x for x in listdir(data_path) if 'fmat_%d_'%int(f[:-3].split('_')[1]) in x][0]
    df = pd.read_hdf(join(data_path, ff),'reduced', where='PatientID=%d'%pid)[['Datetime']+lst_vmid+['pm41','pm42','pm43','pm44','pm87']]

    ff = [x for x in listdir(ep_path) if 'endpoints_%d_'%int(f[:-3].split('_')[1]) in x][0]
    df_ep = pd.read_hdf(join(ep_path, ff), where='PatientID=%d'%pid)[['Datetime','endpoint_status']]

    df_lbl =  pd.read_hdf(join(pred_path, f),'p%d'%pid)[['AbsDatetime','TrueLabel']]
    # df.loc[:,'Datetime'] = pd.DatetimeIndex(df.Datetime).round('min').values
    # df_ep.loc[:,'Datetime'] = pd.DatetimeIndex(df_ep.Datetime).round('min').values
    # df_lbl.loc[:,'AbsDatetime'] = pd.DatetimeIndex(df_lbl.AbsDatetime).round('min').values
    total_los = 0
    df = df.groupby('Datetime').mean()
    df_ep.set_index('Datetime', inplace=True)
    df_lbl.set_index('AbsDatetime', inplace=True)
    
    df = df.merge(df_ep, how='outer', left_index=True, right_index=True)
    df = df.merge(df_lbl, how='outer', left_index=True, right_index=True)
    df.sort_index(inplace=True)

    df_ep.loc[:,'Stable'] = (df_ep.endpoint_status=='event 0').astype(int)
    df_ep.loc[:,'InEvent'] = df_ep.endpoint_status.isin(['event 1','event 2','event 3']).astype(int)

    beg_stable = df_ep.index[np.where(np.array([0]+np.diff(df_ep.Stable.values).tolist())==1)]
    end_stable = df_ep.index[np.where(np.array(np.diff(df_ep.Stable.values).tolist())==-1)]
    if df_ep.iloc[0].Stable==1:
        beg_stable = np.concatenate([[df_ep.index[0]], beg_stable])
    if df_ep.iloc[-1].Stable==1:
        end_stable = np.concatenate([end_stable, [df_ep.index[-1]]])
    assert(len(beg_stable)==len(end_stable))
    
    df.loc[:,'Stable'] = False
    for i in range(len(beg_stable)):
        df.loc[df.index[np.logical_and(df.index>=beg_stable[i],
                                       df.index<=end_stable[i])],'Stable'] = True

    beg_onset = df_ep.index[np.where(np.array([0]+np.diff(df_ep.InEvent).tolist())==1)]
    end_onset = df_ep.index[np.where(np.array(np.diff(df_ep.InEvent).tolist())==-1)]
    if df_ep.iloc[0].InEvent==1:
        beg_onset = np.concatenate([[df_ep.index[0]], beg_onset])
    if df_ep.iloc[-1].InEvent==1:
        end_onset = np.concatenate([end_onset, [df_ep.index[-1]]])
    assert(len(beg_onset)==len(end_onset))
    
    df.loc[:,'InEvent'] = False
    for i in range(len(beg_onset)):
        df.loc[df.index[np.logical_and(df.index>=beg_onset[i],
                                       df.index<=end_onset[i])],'InEvent'] = True

    for col in ['Stable', 'InEvent']:
        df.loc[:,col] = df[col].astype(int)
        
    df.loc[:,'Uncertain'] = ((df.Stable+df.InEvent)==0).astype(int)
    for pmid in ['pm41','pm42','pm43','pm44','pm87']:
        df.loc[:,pmid] = df[pmid].fillna(method='ffill').fillna(0)
    df['OnDrug'] = (df[['pm41','pm42','pm43','pm44','pm87']].sum(axis=1)>0).astype(int)
    
    df.loc[:,'Onset'] = False
    for i, dt in enumerate(beg_onset):
        dt = np.datetime64(dt)
        win_pre_event = df[is_critical_win(dt, df.index.values)]
        if len(win_pre_event)==0 or win_pre_event.Stable.sum()==0:
            continue
        df.loc[dt,'Onset'] = True
        
    del df_ep, df_lbl
    gc.collect()

    dt_unstable = df.index[df.Stable==0]
    
    for col in metaid_oor_mapping.keys():
        if metaid_oor_mapping[col][0] > metaid_oor_mapping[col][1]:
            if col == 'vm28':
                df.loc[:,col+'_Alarm'] = np.logical_or(df[col].values >= metaid_oor_mapping[col][0],
                                                       df[col].values <  metaid_oor_mapping[col][1])
            else:
                df.loc[:,col+'_Alarm'] = np.logical_or(df[col].values > metaid_oor_mapping[col][0],
                                                       df[col].values < metaid_oor_mapping[col][1])
        else:
            df.loc[:,col+'_Alarm'] = np.logical_and(df[col].values > metaid_oor_mapping[col][0],
                                                    df[col].values < metaid_oor_mapping[col][1])
        if len(dt_unstable) > 0:
            df.loc[dt_unstable, col+'_Alarm'] = np.nan

        for dt in df.index[np.abs(df[col+'_Alarm'])==1]:
            dt = np.datetime64(dt)
            win_pos_alarm  = df[is_win_pos_alarm(dt, df.index.values)]
            
            if win_pos_alarm.InEvent.sum() > 0:
                df.loc[dt, col+'_Alarm'] = +1
            elif win_pos_alarm.Uncertain.sum() == len(win_pos_alarm):
                df.loc[dt, col+'_Alarm'] = 0
            else:
                df.loc[dt, col+'_Alarm'] = -1

    df['any_Alarm'] = np.abs(df[[col for col in df.columns if 'Alarm' in col]]).sum(axis=1)>0
    if len(dt_unstable) > 0:
        df.loc[dt_unstable,'any_Alarm'] = np.nan
    for dt in df.index[np.abs(df.any_Alarm)==1]:
        dt = np.datetime64(dt)
        win_pos_alarm  = df[is_win_pos_alarm(dt, df.index.values)]        
        if win_pos_alarm.InEvent.sum() > 0:
            df.loc[dt,'any_Alarm'] = 1
        elif win_pos_alarm.Uncertain.sum() == len(win_pos_alarm):
            df.loc[dt,'any_Alarm'] = 0
        else:
            df.loc[dt,'any_Alarm'] = -1

    for vmid in lst_vmid+['any']:
        df.loc[:,vmid+'_CatchedOnset'] = False
    for i, dt in enumerate(df.index[df.Onset]):
        dt = np.datetime64(dt)
        win_pre_event = df[is_critical_win(dt, df.index.values)]
        for vmid in lst_vmid+['any']:
            df.loc[dt,vmid+'_CatchedOnset'] = win_pre_event[vmid+'_Alarm'].abs().sum()>0

    if df.InEvent.sum()==0:
        # assert('Yes' not in df.IsAlarmTrue.unique())
        tdiff = np.array([0]+(np.diff(df.index.values)/np.timedelta64(1,'h')).tolist())
        tdiff_stable = tdiff[df.Stable==1]
        los_h = np.sum(tdiff)
        los_stable_h = np.sum(tdiff_stable)
        for vmid in lst_vmid+['any']:
            stats[vmid]['patients_wo_events']['valid_los'].append( los_stable_h )
            stats[vmid]['patients_wo_events']['los'].append( los_h )            
            stats[vmid]['patients_wo_events']['cnt_alarm'].append( df[vmid+'_Alarm'].abs().sum() )
    else:
        stable_sum = 0
        beg_onset = df.index[np.where(np.array([0]+np.diff(df.InEvent.values).tolist())==1)[0]]
        end_onset = df.index[np.where(np.diff(df.InEvent.values)==-1)[0]]
        if df.iloc[0].InEvent:
            beg_onset = np.concatenate([[df.index[0]], beg_onset])
        if df.iloc[-1].InEvent:
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
                
            for vmid in lst_vmid+['any']:
                if ~ df.loc[dt,'Onset']:
                    pass
                elif win_pre_event[vmid+'_Alarm'].abs().sum()>0 and df.loc[dt,'Onset'] and df.loc[dt,vmid+'_CatchedOnset']:
                    stats[vmid]['cnt_catched_event'] += 1
                elif win_pre_event[vmid+'_Alarm'].abs().sum()==0 and df.loc[dt,'Onset'] and ~df.loc[dt,vmid+'_CatchedOnset']:
                    stats[vmid]['cnt_missed_event'] += 1
                else:
                    print('Alarm number', win_pre_event[vmid+'_Alarm'].abs().sum(),'; Onset status', df.loc[dt,vmid+'_CatchedOnset'])
                    print(dt)
                    raise Exception('Problem!!!!')
                
            if i > 0:
                win_pre_event = win_pre_event[win_pre_event.index>end_onset[i-1]]

            tdiff = np.array([0]+(np.diff(win_pre_event.index.values)/np.timedelta64(1,'h')).tolist())
            tdiff_stable = tdiff[win_pre_event.Stable==1]
            los_stable_h = np.sum(tdiff_stable)
            los_h = np.sum(tdiff)
            stable_sum += los_stable_h
            for vmid in lst_vmid+['any']:
                stats[vmid]['critical_window']['valid_los'].append( los_stable_h )
                stats[vmid]['critical_window']['los'].append( los_h )            
                stats[vmid]['critical_window']['cnt_alarm'].append( win_pre_event[vmid+'_Alarm'].abs().sum() )

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

            tdiff = np.array([0]+(np.diff(win_pre_event.index.values)/np.timedelta64(1,'h')).tolist())
            tdiff_stable = tdiff[win_pre_event.Stable==1]
            los_stable_h = np.sum(tdiff_stable)
            los_h = np.sum(tdiff)
            stable_sum += los_stable_h
            for vmid in lst_vmid+['any']:
                stats[vmid]['uncritical_window']['valid_los'].append(los_stable_h)
                stats[vmid]['uncritical_window']['los'].append( los_h )            
                stats[vmid]['uncritical_window']['cnt_alarm'].append(win_pre_event[vmid+'_Alarm'].abs().sum())
            
        
        win_post_last_event = df[is_uncritical_win(np.datetime64(end_onset[-1]),df.index.values,'after')]
        tdiff = np.array([0]+(np.diff(win_post_last_event.index.values)/np.timedelta64(1,'h')).tolist())
        tdiff_stable = tdiff[win_post_last_event.Stable==1]
        los_stable_h = np.sum(tdiff_stable)
        los_h = np.sum(tdiff)
        stable_sum += los_stable_h
        for vmid in lst_vmid+['any']:
            stats[vmid]['uncritical_window']['los'].append( los_h )            
            stats[vmid]['uncritical_window']['valid_los'].append(los_stable_h)
            stats[vmid]['uncritical_window']['cnt_alarm'].append(win_post_last_event[vmid+'_Alarm'].abs().sum())

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
            tdiff = np.array([0]+(np.diff(win_post_event.index.values)/np.timedelta64(1,'h')).tolist())
            tdiff_stable = tdiff[win_post_event.Stable==1]
            los_stable_h = np.sum(tdiff_stable)
            stable_sum += los_stable_h
            los_h = np.sum(tdiff)
            for vmid in lst_vmid+['any']:
                stats[vmid]['maintenance_window']['valid_los'].append(los_stable_h)
                stats[vmid]['maintenance_window']['los'].append(los_h)
                stats[vmid]['maintenance_window']['cnt_alarm'].append(win_post_event[vmid+'_Alarm'].abs().sum())

        tdiff = np.array([0]+(np.diff(df.index.values)/np.timedelta64(1,'h')).tolist())
        tdiff_stable = tdiff[df.Stable==1]
        all_los_stable_h = np.sum(tdiff_stable)

        assert(np.abs(all_los_stable_h-stable_sum)<1)
        
    if (n+1)%10==0:
        print('Process %d patients, time: %4.4g sec'%(n+1, time()-t_start))
        gc.collect()
        
with open(join(out_path, f.replace('.h5','.pkl')),'wb') as ff:
    pickle.dump(stats, ff)




