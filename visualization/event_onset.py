#!/usr/bin/env ipython
#
# author: Stephanie Hyland
# purpose: Visualise variables before endpoint onset.

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import pdb
import paths

imputed_dir = paths.root + '/5_imputed/imputed_180221/exploration/'
merged_dir  = paths.root + '/3_merged/180214/'
endpoints_dir = paths.root + '3a_endpoints/180214/'

id2string = np.load(paths.root + '/misc_derived/stephanie/id2string.npy').item()
id2string['v24000524'] = 'alac'
id2string['v24000732'] = 'vlac1'
id2string['v24000485'] = 'vlac2'

df_chunks = pd.read_csv(paths.root + '/misc_derived/id_lists/PID_50chunkfile_index.csv')
pid_list = df_chunks['PatientID'].values

def load_patients(vIDs, pids, imputed):
    """
    Given a list of pids, load them
    (this is nontrivial because they're all from different chunks!)
    """
    df_chunks_subset = df_chunks.loc[df_chunks['PatientID'].isin(pids), :]
    chunks = df_chunks_subset['ChunkfileIndex'].unique()
    # only look at chunks up to 45 (due to held out)
    chunks = chunks[chunks < 45]
    df_list = []
    endpoint_list = []
    for chunk_idx in chunks:
        pid_start = str(df_chunks.loc[df_chunks['ChunkfileIndex'] == chunk_idx, 'PatientID'].min())
        pid_stop = str(df_chunks.loc[df_chunks['ChunkfileIndex'] == chunk_idx, 'PatientID'].max())
        
        chunk_pids = df_chunks_subset.loc[df_chunks_subset['ChunkfileIndex'] == chunk_idx, 'PatientID'].values
        pids_string = ','.join(list(map(str, chunk_pids)))
        
        # get endpoint info (we only need patients with endpoints here)
        endpoints = pd.read_hdf(endpoints_dir + 'endpoints_' + str(chunk_idx) + '_' + pid_start + '--' + pid_stop + '.h5', where='PatientID in [' + pids_string + ']', columns=['event1', 'event2', 'event3', 'PatientID', 'Datetime'])
        endpoint_stats = endpoints.groupby('PatientID').apply(lambda x: x.loc[:, ['event1', 'event2', 'event3']].sum().sum())
        pids_with_endpoints = endpoint_stats[endpoint_stats > 0].index.values
        print('Only', len(pids_with_endpoints), 'patients have endpoints from this chunk (total', len(chunk_pids), ')')
        if len(pids_with_endpoints) == 0:
            continue
        endpoints = endpoints.loc[endpoints['PatientID'].isin(pids_with_endpoints), :]
        pids_string = ','.join(list(map(str, pids_with_endpoints)))
        if imputed:
            chunk_df = pd.read_hdf(imputed_dir + 'batch_' + str(chunk_idx) + '.h5', where='PatientID in [' + pids_string + ']', columns=['PatientID', 'AbsDatetime'] + vIDs)
            chunk_df.rename(columns={'AbsDatetime': 'Datetime'}, inplace=True)
        else:
            chunk_df = pd.read_hdf(merged_dir + 'fmat_' + str(chunk_idx) + '_' + pid_start + '--' + pid_stop + '.h5', where='PatientID in [' + pids_string + ']', columns=['PatientID', 'Datetime'] + vIDs)
        # merge 
        df_list.append(chunk_df)
        endpoint_list.append(endpoints)
    df = pd.concat(df_list)
    df.reset_index(inplace=True, drop=True)
    df.set_index(['PatientID', 'Datetime'], inplace=True)
    df.sort_index(inplace=True)
    endpoints = pd.concat(endpoint_list)
    endpoints.reset_index(inplace=True, drop=True)
    endpoints.set_index(['PatientID', 'Datetime'], inplace=True)
    endpoints.sort_index(inplace=True)
    return df, endpoints

def extract_traces(df, endpoints, vIDs, hours, level, exclude_regions_with_endpoints):
    # now we need to get segments of "hours" length before endpoints start
    event_starts = endpoints.loc[(endpoints['event' + str(level)] - endpoints['event' + str(level)].shift()) == 1, :].index.values
    # now we take the regions up to them
    onsets = [(x[0], x[1] - pd.Timedelta(str(hours) + ' hours')) for x in event_starts]
    print('Found', len(event_starts), 'event onsets')
    traces = [None]*len(vIDs)
    for start, stop in zip(onsets, event_starts):
        df_subset = df.loc[start:stop, :]
        if df_subset.empty:
            # we have lots of patients missing because they're not in the imputed file
            print('no data in this region, dropping')
            continue
        endpoint_subset = endpoints.loc[start:stop, :]
        if exclude_regions_with_endpoints and (1 in endpoint_subset.loc[:, ['event1', 'event2', 'event3']].values[:-1, :]):
            print('excluding region containing endpoint')
            continue
        else:
            # keep this segment
            df_subset.reset_index(inplace=True)
            minutes_before = (stop[1] - df_subset['Datetime'])
            df_subset.index = minutes_before
            trace = df_subset.loc[:, vIDs]
#            try:
#                assert df_subset.index.max() <= hours*60
#            except AssertionError:
#                pdb.set_trace()
            for (i, vID) in enumerate(vIDs):
                if traces[i] is None:
                    traces[i] = trace.loc[:, [vID]]
                else:
                    traces[i] = traces[i].join(trace.loc[:, [vID]], how='outer', rsuffix='x')
    # now resample to get on comparable grid
    traces = [t.resample('5T').median() for t in traces]
    print('Finished with', len(traces), 'traces.')
    return traces

def view_onset(vIDs=['v200', 'v300', 'v110'], n_patients=500, hours=1, imputed=True, level=1, limits=[(30, 150), (10, 30), (50, 150)], exclude_regions_with_endpoints=True):
    """
    """
    # get patients
    pids = np.random.choice(pid_list, n_patients, replace=False)
    df, endpoints = load_patients(vIDs, pids, imputed)
    traces = extract_traces(df, endpoints, vIDs, hours, level, exclude_regions_with_endpoints)
    # we need to merge the traces somehow
    # now plot it
    fig, axarr = plt.subplots(nrows=len(vIDs))
    # do each variable independently
    for (i, ax) in enumerate(axarr):
        vID = vIDs[i]
        vID_traces = traces[i]
        means = vID_traces.mean(axis=1)
        medians = vID_traces.median(axis=1)
        mins = vID_traces.min(axis=1)
        maxs = vID_traces.max(axis=1)
        sems = vID_traces.sem(axis=1)
        iqr = vID_traces.quantile(0.75, axis=1) - vID_traces.quantile(0.25, axis=1)
        times = -means.index.astype('timedelta64[m]').values
        # now plot it
        vID_color = ['red', 'green', 'blue'][i]
        robust = True
        if robust:
            line, = ax.plot(times, medians.values, color=vID_color)
            ax.fill_between(times, medians.values - iqr.values, medians.values + iqr.values, alpha=0.5, color=line.get_color())
        else:
            line, = ax.plot(times, means.values, color=vID_color)
            ax.fill_between(times, means.values - sems.values, means.values + sems.values, alpha=0.5, color=vID_color)
        ax.fill_between(times, medians.values - mins.values, medians.values + maxs.values, alpha=0.2, color=vID_color)
    
    for (i, ax) in enumerate(axarr):
        if limits[i] is not None:
            ax.set_ylim(limits[i])
        ax.set_ylabel(id2string[vIDs[i]])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(bottom='off', left='off')
        ax.grid(linestyle='--', alpha=0.5)
        ax.set_facecolor((0.95, 0.95, 0.95)) 
   
    plt.title('number of samples: ' + str(traces[0].shape[1]))
    axarr[-1].set_xlabel('minutes before endpoint onset')
    plt.savefig('onset_level' + str(level) + '_imputed'*imputed + '_endpoints.excluded'*exclude_regions_with_endpoints + '_' + str(n_patients) + '_patients.png')
    return True

def view_onsets_all_variables(n_patients=1000, hours=4, imputed=True, level=1, exclude_regions_with_endpoints=True):
    for variable in vIDs:
        print('visualising variable', vID)
        view_onset(['v200', vID], n_patients, hours, imputed, level, limits=[None, None], exclude_regions_with_endpoints=exclude_regions_with_endpoints)
