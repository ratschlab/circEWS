''' Visualization script for data from all pipeline stages

    Authors: SH, with additions from MF/MH
'''

import os
import pdb
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42

import paths

ID2STRING = paths.root + '/misc_derived/visualisation/id2string_v6.npy'
MID2STRING = paths.root + 'misc_derived/visualisation/mid2string_v6.npy'
id2string = np.load(ID2STRING).item()
mid2string = np.load(MID2STRING).item()

PRINT_FRIENDLY = True

# ---------------------------- DATA FETCHERS ------------------------------------------------------------------------------------------------------------------------------------


def get_patient_data(pid, variables=None, hours=None, configs=None, horizons=None,  events=['event1', 'event2', 'event3', 'maybe_event1', 'maybe_event2', 'maybe_event3']):
    """
    Load patient's data.
    """
    df_chunks = pd.read_csv(configs["chunk_file_path"])
    if not pid in df_chunks['PatientID'].values:
        print('Patient', pid, 'does not exist in chunk file.')
        print('Maybe try', np.random.choice(df_chunks['PatientID'], 1))
        raise ValueError(pid)
    chunk_idx = df_chunks.loc[df_chunks['PatientID'] == pid, 'ChunkfileIndex'].values[0]

    idx_start = min(df_chunks.loc[df_chunks.ChunkfileIndex == chunk_idx, 'PatientID'])
    idx_stop = max(df_chunks.loc[df_chunks.ChunkfileIndex == chunk_idx, 'PatientID'])
    chunkname = str(chunk_idx) + '_' + str(idx_start) + '--' + str(idx_stop) + '.h5'

    split = configs["data_split"]
    task = configs["label_task"]
    show_reduced = configs['show_reduced']

    HR_variable = "vm1" if show_reduced else "v200"

    if not HR_variable in variables:
        variables = variables + [HR_variable]

    if configs["show_imputed"]:

        if configs["verbose"]:
            print('NOTICE: using split', split)

        # there is a chance this file won't exist in this split...
        imputed_path =  configs["imputed_path"]+ show_reduced*"reduced/"+ split + '/' + 'batch_' + str(chunk_idx) + '.h5'
        df = pd.read_hdf(imputed_path, columns=variables + ['PatientID', 'AbsDatetime'], mode='r')

        df = df.loc[df['PatientID'] == pid, :]
        df.drop('PatientID', axis=1, inplace=True)
        df.rename(columns={'AbsDatetime': 'Datetime'}, inplace=True)
    else:
        df = pd.read_hdf(configs["merged_path"] + show_reduced*"reduced/reduced_" + 'fmat_' + chunkname, where='PatientID ==' + str(pid), columns=variables + ['Datetime'], mode='r')
        df.drop_duplicates(inplace=True)

    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    if len(df[np.logical_not(df[HR_variable].isnull())]) == 0:
        print('No HR data available for patient', pid)
        return pd.DataFrame(columns=[''])

    if show_reduced:
        endpoints = pd.read_hdf(configs["endpoints_path"]+"reduced/reduced_" + 'endpoints_' + chunkname, where='PatientID == ' + str(pid), mode='r')
    else:
        endpoints = pd.read_hdf(configs["endpoints_path"]+'endpoints_' + chunkname, where='PatientID == ' + str(pid), mode='r')

    endpoints = endpoints.loc[:, ['Datetime'] + events]
    endpoints.set_index('Datetime', inplace=True)
    df = df.join(endpoints, how='outer')

    for start, stop in horizons:

        # predictions_path = configs["predictions_path"] + show_reduced*'reduced/' + split + '/' + task + '_' + str(start) + '_' + str(stop) + '_fixed_history_hist_4_hours_5pointsummary_{}/batch_'.format(configs["ml_model"]) + str(chunk_idx) + '.h5'
        predictions_path = configs["predictions_path"] + show_reduced*'reduced/' + split + '/' + task + '_' + str(start) + '_' + str(stop) + '_' + configs['ml_model'] + '/batch_' + str(chunk_idx) + '.h5'
        
        try:
            predictions = pd.read_hdf(predictions_path, key='p' + str(pid), mode='r')
#            if configs["show_reduced"]:
#                predictions = pd.read_hdf(configs["predictions_path"] + 'reduced/' + split + '/' + task + '_' + str(start) + '_' + str(stop) + '_fixed_history_hist_4_hours_5pointsummary_{}/batch_'.format(configs["ml_model"]) + str(chunk_idx) + '.h5', 'p' + str(pid), mode='r')
#            else:
#                predictions = pd.read_hdf(configs["predictions_path"] + split + '/' + task + '_' + str(start) + '_' + str(stop) + '_fixed_history_hist_4_hours_5pointsummary_{}/batch_'.format(configs["ml_model"]) + str(chunk_idx) + '.h5', 'p' + str(pid), mode='r')

            predictions = predictions.loc[:, ['AbsDatetime', 'PredScore', 'TrueLabel']]
            predictions.rename(columns={'AbsDatetime': 'Datetime', 'PredScore': 'Score_' + str(start) + '-' + str(stop), 'TrueLabel': 'Label_' + str(start) + '-' + str(stop)}, inplace=True)
            predictions.set_index('Datetime', inplace=True)
            df = df.join(predictions, how='outer')

        except:
            print('No predictions available for patient', pid)
            # we can't make it nan because it'll get deleted later
            df['Score_' + str(start) + '-' + str(stop)] = -1
            df['Label_' + str(start) + '-' + str(stop)] = -1


    df['RelDatetime'] = (df.index - df[HR_variable].dropna().index[0]).astype('timedelta64[s]')
    df['Datetime'] = df.index
    df.set_index('RelDatetime', inplace=True)
    df.sort_index(inplace=True)


    df['not_good_segment'] = 1
    goodsegments = pd.read_hdf(configs["good_segments_path"], where='PatientID==' + str(pid), mode='r')

    for index, segment in goodsegments.iterrows():
        df.loc[(df.Datetime >= segment['LeftTs']) & (df.Datetime <= segment['RightTs']), 'not_good_segment'] = 0
    
    df.dropna(how='all', axis=1, inplace=True)
    return df


# ------------- VISUALIZATION HELPERS --------------------------------------------------------------------------------------------

def overlay_endpoints_on_axis(ax, df):
    """
    Show endpoints on an axis using vertical grey areas.
    """
    ymin, ymax = ax.get_ylim()
    df_temp = df.loc[:, ['event1', 'event2', 'event3', 'maybe_event1', 'maybe_event2', 'maybe_event3']]
    df_temp[df_temp == 0] = np.nan

    if df_temp.empty:
        print('No endpoints to overlay, skipping')

    event1 = df['event1'].dropna()
    event1[event1 == 0] = np.nan
    ax.fill_between(event1.index/3600, ymin*event1, ymax*event1,
        label='event1',
        color='#d83636',
        alpha=0.5,
        linewidth=0,
        zorder=0)
    
    maybe_event1 = df['maybe_event1'].dropna()
    maybe_event1[maybe_event1 == 0] = np.nan
    ax.fill_between(maybe_event1.index/3600, ymin*maybe_event1, ymax*maybe_event1,
        label='maybe_event1',
        color='#d83636',
        hatch='X',
        edgecolor='black',
        alpha=0.25,
        linewidth=0,
        zorder=0)

    event2 = df['event2'].dropna()
    event2[event2 == 0] = np.nan
    ax.fill_between(event2.index/3600, ymin*event2, ymax*event2,
        label='event2',
        color='#d83636',
        alpha=0.5,
        linewidth=0,
        zorder=0)

    maybe_event2 = df['maybe_event2'].dropna()
    maybe_event2[maybe_event2 == 0] = np.nan
    ax.fill_between(maybe_event2.index/3600, ymin*maybe_event2, ymax*maybe_event2,
        label='maybe_event2',
        color='#d83636',
        hatch='X',
        edgecolor='black',
        alpha=0.25,
        linewidth=0,
        zorder=0)

    event3 = df['event3'].dropna()
    event3[event3 == 0] = np.nan
    ax.fill_between(event3.index/3600, ymin*event3, ymax*event3,
        label='event3',
        color='#d83636',
        alpha=0.5,
        linewidth=0,
        zorder=0)
   
    maybe_event3 = df['maybe_event3'].dropna()
    maybe_event3[maybe_event3 == 0] = np.nan
    ax.fill_between(maybe_event3.index/3600, ymin*maybe_event3, ymax*maybe_event3,
        label='maybe_event3',
        color='#d83636',
        hatch='X',
        edgecolor='black',
        alpha=0.25,
        linewidth=0,
        zorder=0)



def overlay_label_on_axis(ax, df, label):
    """
    Show labels on an axis using vertical grey areas.
    """
    ymin, ymax = ax.get_ylim()
    prediction_task = "Label_" + label.split('_')[1]
    
    df_temp = df.loc[:, [prediction_task]]

    if df_temp.empty:
        print('No labels to overlay, skipping')

    trueLabel = df[prediction_task].dropna()
    trueLabel[trueLabel == 0] = np.nan
    ax.fill_between(trueLabel.index/3600, ymin*trueLabel, ymax*trueLabel,
        color='#989898',
        alpha=0.5,
        linewidth=1.1,
        zorder=0)


def overlay_prediction_results_on_axis(ax, df, treshold):
    """
    Show labels on an axis using vertical grey areas.
    """
    ymin, ymax = ax.get_ylim()
    prediction_label = "Label_" + ax.get_ylabel().split('_')[1]
    prediction_score = "Score_" + ax.get_ylabel().split('_')[1]
    
    df['truepositiv'] = (df[prediction_label] == 1) & (df[prediction_score] >= treshold)
    truepositiv = df['truepositiv'].dropna()
    truepositiv[truepositiv == 0] = np.nan
    ax.fill_between(truepositiv.index/3600, ymin*truepositiv, ymax*truepositiv,
        color='#32CD32',
        alpha=0.5,
        linewidth=1.1,
        zorder=0)

    df['falsepositiv'] = (df[prediction_label] == 0) & (df[prediction_score] >= treshold)
    falsepositiv = df['falsepositiv'].dropna()
    falsepositiv[falsepositiv == 0] = np.nan
    ax.fill_between(falsepositiv.index/3600, ymin*falsepositiv, ymax*falsepositiv,
        color='#FD6A02',
        alpha=0.5,
        linewidth=1.1,
        zorder=0)

    df['falsenegative'] = (df[prediction_label] == 1) & (df[prediction_score] < treshold)
    falsenegative = df['falsenegative'].dropna()
    falsenegative[falsenegative == 0] = np.nan
    ax.fill_between(falsenegative.index/3600, ymin*falsenegative, ymax*falsenegative,
        color='#F9A602',
        alpha=0.5,
        linewidth=1.1,
        zorder=0)



def overlay_goodsegment_on_axis(ax, df):
    """
    Show "bad segments" (not a good segment) as area with white background on the axis.
    """
    ymin, ymax = ax.get_ylim()
    df_temp = df.loc[:, 'not_good_segment']

    not_good_segment = df['not_good_segment'].dropna()
    not_good_segment[not_good_segment == 0] = np.nan
    ax.fill_between(not_good_segment.index/3600, ymin*not_good_segment, (ymin-(ymax-ymin)*0.05)*not_good_segment,
        color='#FF0000',
        alpha=0.5,
        linewidth=1.1)



# --------------------- MAIN VISUALIZATION FUNCTION -------------------------------------------------------------------

def vis_patient(pid, hours=None, ylims=None, variables=None, file_prefix='p', configs=None, horizons=None):
    """
    Plot a given patient at given hours for given variables.

    Keyword arguments:
        pid -- patient id
        variables -- list of variable IDs (should be like vXXX, pXXXXX, X digits)
        hours -- tuple [start_hour, end_hour] to restrict x-axis
        overlay_endpoints -- flag denoting if endpoint status should be overlaid
        imputed -- flag denoting if imputed or pre-imputed ("raw") data should be used
        ylims -- list of tuples [[y0_min, y0_max, ..., yn_min, yn_max]] for n variables. A tuple can be replaced with 'None' in the list to use default limits for that variable, or [999, 999] to let the script choose smart limits
        output_path -- folder to which images are saved, default is folder of script
        file_prefix -- pass a prefix with which the filename will starts
    """

    variable_names = []
    if variables is None:
        #full
        variables=["vm5", "vm1", "vm13", "vm136", "vm28", "vm172", "vm174", "vm176", "vm20", "vm62","vm23", "pm87", "pm39"]
        variable_names=["ABP mean [mmHg]", "Heart rate [/min]", "Cardiac output [l/min]", "Arterial lactate [mmol/l]", "RASS Score", "INR", "Blood glucose [mmol/l]", "C-reactive protein [mg/l]", "SpO2 [%]", "Ventilator peak pressure [cmH2O]","Supplemental oxygen [l/min]", "Non-opioid analgetics [binary]", "Norepinephrine [μg/l]"]
        #reduced
        variables=["vm5", "vm3", "vm4", "vm1", "vm136", "vm28", "vm174", "vm176", "vm20","vm23", "pm87", "pm39"]
        variable_names=["MAP [mmHg]", "ABPsystolic [mmHg]", "ABPdiastolic [mmHg]", "Heart rate [/min]", "Arterial lactate [mmol/l]", "RASS Score", "Blood glucose [mmol/l]", "C-reactive protein [mg/l]", "SpO2 [%]","Supplemental oxygen [l/min]", "Non-opioid analgetics [binary]", "Norepinephrine [μg/l]"]

    df = get_patient_data(pid, variables=variables, hours=hours, configs=configs, horizons=horizons)

    if df.empty:
        raise Exception("No data available for this patient!")

    variables = [v for v in variables if v in df.columns]
    n_vars = len(variables) 
    fig = plt.figure(figsize=(10, n_vars + 1))
    gs = mpl.gridspec.GridSpec(n_vars + len(horizons), 1, height_ratios = [1]*n_vars + [0.4]*len(horizons))
    axarr = [plt.subplot(x) for x in gs]
    colours = plt.cm.Dark2(np.linspace(0, 1, n_vars))
    # for aligning the y-axis labels
    labelx = -0.15
 
    if ylims is None:
        ylims = [None]*n_vars
    else:
        assert len(ylims) == n_vars

    for i in range(len(variables)):
        var = variables[i]
        if configs['show_reduced']:
            varname = mid2string[var]
        else:
            varname = id2string[var]

        if len(variable_names) == len(variables):
            varname = variable_names[i]

        ax = axarr[i]
        c = colours[i]
        relevant_slice = df.loc[~df[var].isnull()]

        if configs["show_imputed"]:
            ax.plot(relevant_slice.index/3600, relevant_slice[var], color=c, label='_nolegend_')
        else:
            ax.plot(relevant_slice.index/3600, relevant_slice[var], color=c, label='_nolegend_', alpha=0.3)
            ax.scatter(relevant_slice.index/3600, relevant_slice[var], color=c, label='_nolegend_', s=2.2)

        ax.set_ylabel(varname, rotation='horizontal', ha='right')
        ax.yaxis.set_label_coords(labelx, 0.5)
        y_limit = ylims[i]

        if not y_limit is None:
            if (y_limit[0] == 999 & y_limit[1] == 999):
                y_limit = [relevant_slice[var].quantile(0.05),relevant_slice[var].quantile(0.95)]
            else:
                ax.set_ylim(y_limit)

        ax.set_xlim(hours[0], hours[1])
        
#    if hours is None:
#        # do no filtering
#        pass
#    else:
#        assert len(hours) == 2
#        start, end = hours
#        # we can have an open start/end, e.g. hours can be [None, 5]
#        if not start is None:
#            df = df.loc[start*3600:, :]
#        if not end is None:
#            df = df.loc[:end*3600, :]

        if configs["with_endpoints"]:
            overlay_endpoints_on_axis(ax, df)

        if configs["with_goodsegments"]:
            overlay_goodsegment_on_axis(ax, df)

    for (hi, (start, stop)) in enumerate(horizons):
        var = 'Score_' + str(start) + '-' + str(stop)
        if PRINT_FRIENDLY:
            varname = 'Score'
        else:
            varname = var
        relevant_slice = df.loc[:]
        axarr[n_vars + hi].scatter(relevant_slice.index/3600, relevant_slice[var], label='_nolegend_', s=2.2)
        axarr[n_vars + hi].plot(relevant_slice.index/3600, relevant_slice[var].fillna(0), label='_nolegend_', alpha=0.3)
        axarr[n_vars + hi].set_ylabel(varname, rotation='horizontal', ha='right')
        axarr[n_vars + hi].yaxis.set_label_coords(labelx, 0.5)
        axarr[n_vars + hi].set_ylim([0, 1])
        axarr[n_vars + hi].set_yticklabels([])
        axarr[n_vars + hi].set_xlim(hours[0], hours[1])

        if configs["with_labels"]:
            overlay_label_on_axis(axarr[n_vars + hi], df, var)

        if configs["with_performance"]:
            overlay_prediction_results_on_axis(axarr[n_vars + hi], df, configs["treshold"])

    for ax in axarr:
        ax.set_facecolor((0.95, 0.95, 0.95)) 
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(bottom='off', left='off')
        ax.grid(linestyle='--', alpha=0.5)
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.setp(axarr[-1].get_xticklabels(), visible=True)

    axarr[-1].set_xlabel('hours after admission')
#    axarr[-1].legend()

    if not PRINT_FRIENDLY:
        # don't print patient IDs
        title = 'patient ' + str(pid) + configs['show_imputed']*' (imputed)'
        plt.suptitle(title)
    fig.savefig(os.path.join(configs["output_folder"], file_prefix + str(pid) + configs['show_imputed']*'.imputed' + '.png'), bbox_inches='tight')
    if PRINT_FRIENDLY:
        # also save a pdf
        fig.savefig(os.path.join(configs["output_folder"], file_prefix + str(pid) + configs['show_imputed']*'.imputed' + '.pdf'), bbox_inches='tight')

    plt.clf()
    plt.close()





if __name__=="__main__":

    version = 'v6b'
    MERGED_DIR  = paths.root + '/3_merged/' + version + '/'
    ENDPOINTS_DIR = paths.root + '/3a_endpoints/' + version + '/'
    IMPUTED_DIR = paths.root + '/5_imputed/imputed_180918/'
    PREDICTIONS_DIR = paths.root + '/8_predictions/181108/'
    GOOD_SEGMENTS_PATH = paths.root + '/misc_derived/good_segments_180108.h5'

    OUTPUT_DIR = paths.root + '/visualisation/patients/martin_for_paper'

    CHUNKS = paths.root + '/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.csv'

    #HORIZONS = [[1,3],[2,4],[5,7]]
    if PRINT_FRIENDLY:
        HORIZONS = [[0.0, 8.0]]
    else:
        HORIZONS = [[0.0, 4.0], [0.0, 8.0]]
    
    parser=argparse.ArgumentParser()

    parser.add_argument("--pid", type=int, default=None, help="Patient ID of stay to display")
    parser.add_argument("--lhours",type=int,default=None, help="Left boundary of the segment to display")
    parser.add_argument("--rhours", type=int, default=None, help="Right boundary of the segment to display")

    parser.add_argument("--verbose", type=bool, default=True, help="Verbosity mode with info messages")

    parser.add_argument("--show_imputed", type=bool, default=False, help="Should the imputed data be displayed?")
    parser.add_argument("--show_reduced", type=bool, default=True, help="Should the dim-reduced version of the data be displayed?")
    parser.add_argument("--data_split", default="held_out", help="Data of which split should be analyzed?")
    parser.add_argument("--label_task", default="WorseStateFromZero", help="Which base task should be displayed for labels/predictions")

    parser.add_argument("--output_folder", default=OUTPUT_DIR, help="Folder where plots are generated")
    parser.add_argument("--chunk_file_path", default=CHUNKS, help="Chunk file to be used to find PID/Batch mapping")
    parser.add_argument("--merged_path", default=MERGED_DIR, help="Directory from where mertged data should be loaded")
    parser.add_argument("--endpoints_path", default=ENDPOINTS_DIR, help="Base directory for endpoints, reduced will be auto-attached")
    parser.add_argument("--imputed_path", default=IMPUTED_DIR, help="Base directory of the imputed data, reduced will be auto-attached")
    parser.add_argument("--predictions_path", default=PREDICTIONS_DIR, help="Base directory for the predictions to use")
    parser.add_argument("--good_segments_path", default=GOOD_SEGMENTS_PATH, help="Which good segments file should be loaded and overlaid?")

    parser.add_argument("--with_endpoints", type=bool, default=True, help="Should endpoints be overlaid on the plot?")
    parser.add_argument("--with_goodsegments", type=bool, default=False, help="Should good segments be overlaid on the plot?")
    parser.add_argument("--with_labels", type=bool, default=True, help="Should the true label be overlaid on the plot?")
    parser.add_argument("--with_performance", type=bool, default=False, help="Should the true positive, false positive and false negative be overlaid on the plot?")
    parser.add_argument("--treshold", type=float, default=0.4, help="Should the true positive, false positive and false negative be overlaid on the plot?")

    parser.add_argument("--ml_model", default="shap_top500_features_lightgbm_full", help="For which ML model should the predictions be displayed?")

    args=parser.parse_args()
    configs=vars(args)

    PIDS_OF_TESTSET=os.path.join(paths.root, "misc_derived/temporal_split_180918.tsv")
    #load patientid file
    df_ids = pd.read_csv(PIDS_OF_TESTSET, '\t')
    uniquePatientIDs = df_ids[df_ids['held_out'] == 'test'].pid.values

    vis_patient(configs["pid"], hours=(configs["lhours"], configs["rhours"]), configs=configs, horizons=HORIZONS)
#    for pid in uniquePatientIDs:
        #vis_patient(configs["pid"], hours=(configs["lhours"], configs["rhours"]), configs=configs, horizons=HORIZONS)
 #       vis_patient(pid, hours=(configs["lhours"], configs["rhours"]), configs=configs, horizons=HORIZONS)
    



   


