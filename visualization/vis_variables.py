#!/usr/bin/env ipython
# author: Stephanie
#

import numpy as np
import pandas as pd
import h5py
import re
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import paths

# --- globals --- #
id2string = np.load(paths.root + '/misc_derived/stephanie/id2string.npy').item()

def vis_merged():
    """
    Visualise variables pre-imputation.
    """
    out_dir = paths.root + '/visualisation/variables/3_merged/'
    in_dir = paths.root + '/3_merged/fmat_170302/'
    # the following is a hack
    variable_ids = pd.read_hdf(paths.root + '/5_imputed/fmat_endpoints_imputed_170308.h5', mode='r', stop=1).columns
    # and more
    pdict = np.load(paths.root + '/misc_derived/split_170301.npy').item()
    patients = pdict['train'] + pdict['validation']
    for varid in variable_ids:
        if varid in {'PatientID', 'RelDatetime', 'AbsDatetime', 'event1', 'event2', 'event3', 'maybe_event1', 'maybe_event2', 'maybe_event3', 'probably_not_event1', 'probably_not_event2', 'probably_not_event3'}:
            # skip
            continue
        df_list = []
        for pid in np.random.choice(patients, size=250):
            if int(pid) in [762, 767]:
                print('this patient has an error, skipping')
                continue
            try:
                temp_df = pd.read_hdf(in_dir + 'p' + pid + '.h5', key=varid, mode='r')
            except KeyError:
                print('missing key', varid, 'in patient', pid)
                continue
            temp_df['PatientID'] = pid
            df_list.append(temp_df)
        var_df = pd.concat(df_list)
        plot_var(varid, var_df, out_dir)
    return True

def vis_imputed():
    """
    Visualise variables post-imputation
    """
    out_dir = paths.root + '/visualisation/variables/5_imputed/'
    fname = paths.root + '/5_imputed/fmat_endpoints_imputed_170308.h5'
    df = pd.read_hdf(fname, mode='r')
    variable_ids = df.columns
    for varid in variable_ids:
        if varid in {'PatientID', 'RelDatetime', 'AbsDatetime', 'event1', 'event2', 'event3', 'maybe_event1', 'maybe_event2', 'maybe_event3', 'probably_not_event1', 'probably_not_event2', 'probably_not_event3'}:
            #skip
            continue
        var_df = df[['PatientID', 'RelDatetime', varid]]
        plot_var(varid, var_df, out_dir)
    return True
    
def plot_var(varid, var_df, out_dir):
    """
    Basically, just load data for this variable and do various visualisations.
    """
    assert 'PatientID' in var_df.columns
    assert varid in var_df.columns
    try:
        varname = id2string[varid]
        print('Plotting', varname, '(' + varid + ')')
    except KeyError:
        print("WARNING: no string recorded for id", varid)
        varname = varid
    # check if it's bad
    if np.sum(np.isfinite(var_df[[varid]].values)) < 2:
        print('Variable', varname, '(' + varid + ') is (almost) entirely nan. Skipping.')
        return False
    # drop NAs
    var_df = var_df.dropna()
    # the plotting
    identifier = out_dir + varid + '_' + re.sub('/', '', re.sub(' ', '_', varname))
    var_hist(var_df[[varid]].values, varname, varid, identifier)
    var_hist_bypatient(var_df[['PatientID', varid]], varname, varid, identifier)
    return True

def var_hist(values, varname, varid, identifier):
    """
    Create histogram of values for this variable.
    """
    print('\thistogram...')
    if 'merged' in identifier:
        colour='green'
    elif 'imputed' in identifier:
        colour='purple'
    plt.hist(values, normed=1, bins=75, facecolor=colour)
    plt.xlabel('value')
    plt.title(varname + '(' + varid + ')')
    plt.grid(True)
    plt.savefig(identifier + '.hist.png')
    plt.clf()
    plt.close()
    return True 

def var_hist_bypatient(df, varname, varid, identifier):
    """
    Create histogram of mean and variance per patient.
    """
    print('\thistogram of per-patient mean and standard deviation...')
    patient_split = df.groupby('PatientID')
    stds = patient_split.std().dropna().values
    means = patient_split.mean().dropna().values
    # create plots side by side
    fig, axarr = plt.subplots(1, 2)
    axarr[0].hist(means, normed=1, bins=75, facecolor='orange')
    axarr[0].grid(True)
    axarr[0].set_title('Means (per patient)')
    axarr[1].hist(stds, normed=1, bins=75, facecolor='blue')
    axarr[1].grid(True)
    axarr[1].set_title('Standard Deviations (per patient)')
    fig.suptitle(varname + '(' + varid + ')')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(identifier + '.patient_hist.png')
    plt.clf()
    plt.close()
    return True
