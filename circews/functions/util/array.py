
import ipdb
import os

import numpy as np
import scipy.stats as sp_stats

import circews.functions.statistics as mlhc_statistics

def nan_ratio(in_arr):
    ''' Returns NAN ratio of input array'''
    return np.sum(in_arr)/in_arr.size


def print_nan_stats(in_arr):
    ''' Print NAN statistics/ value statistics about an array'''
    print("[0.0: {:.5f}, 1.0: {:.5f}, NAN: {:.5f}]".format(np.sum(in_arr==0.0)/in_arr.size,np.sum(in_arr==1.0)/in_arr.size,
                                                           np.sum(np.isnan(in_arr))/in_arr.size))

def pos_ratio(in_arr):
    ''' Returns positive ratio of input array'''
    return np.sum(in_arr==1.0/in_arr.size)

def column_statistics_median_std(in_arr):
    ''' Prints columns statistics along the columns for an array'''
    median_arr=np.median(in_arr,axis=0)
    std_arr=np.std(in_arr,axis=0)
    
    for i in range(median_arr.size):
        print("Col {}: MED: {:.3f}, STD: {:.3f}".format(i,median_arr[i],std_arr[i]))

def first_last_diff(in_arr):
    ''' An approximation to a trend over a time series by just taking the difference
        of the first and last elements'''
    return in_arr[-1]-in_arr[0]

def first_last_diff_NAN(in_arr):
    ''' An approximation to a trend over a time series by just taking the difference
        of the first and last elements. Dealing with missing values. '''
    if np.isnan(in_arr[-1]) or np.isnan(in_arr[0]):
        return np.nan
    else:
        return in_arr[-1]-in_arr[0]

def print_pos_stats(in_arr,desc_str):
    if np.sum(in_arr==1.0)>0:
        print(desc_str)


def empty_nan(sz):
    ''' Returns an empty NAN vector of specified size'''
    arr=np.empty(sz)
    arr[:]=np.nan
    return arr

def value_empty(size, default_val, dtype=None):
    ''' Returns a vector filled with elements of a specific value'''

    if dtype is not None:
        tmp_arr=np.empty(size, dtype=dtype)
    else:
        tmp_arr=np.empty(size)

    tmp_arr[:]=default_val
    return tmp_arr


def clip_to_val(mat, threshold):
    ''' Clips the absolute value of array entries to a certain threshold, maintaining the sign'''
    mat[mat>threshold] = threshold
    mat[mat<-threshold] = -threshold
    return mat

def time_diff(t1, t2=None):
    ''' Returns the time difference between two arrays of the same shape, and if one array is passed
        return the differenced time-steps'''
    if t2 is None:
        tdiff = np.diff(t1)
    else:
        if type(t1) == list:
            t1 = np.array(t1)
        if type(t2) == list:
            t2 = np.array(t2)
        tdiff = t2 - t1
    return tdiff


def array_mode(in_arr):
    ''' Estimates the mode of an array by using a histogram and finding the bar with the
        most values in it'''
    if in_arr.size==1:
        return in_arr[0]
    counts,edges = np.histogram(in_arr,bins=50)
    max_idx=counts.argmax()
    midpoint=(edges[max_idx]+edges[max_ids+1])/2
    return midpoint

def time_window_mean(in_arr):
    ''' Summarizes a temporal vector by computing window means over non-overlapping segments of length 3 on it
    '''
    assert(in_arr.shape[0]%3==0)
    out_arr_nrows=in_arr.shape[0]//3
    out_arr=np.zeros((out_arr_nrows,in_arr.shape[1]))
    for idx in np.arange(0,in_arr.shape[0],3):
        out_arr[idx//3,:]=np.mean(in_arr[idx:idx+3,:],axis=0)
    return out_arr.flatten()

def time_window_mean_std_trend(in_arr):
    ''' Summarizes a temporal vector by computing mean, std and trend features'''
    mean_vect=np.mean(in_arr,axis=0).flatten()
    std_vect=np.std(in_arr,axis=0).flatten()
    trend_vect=np.apply_along_axis(lambda slice_arr: mlhc_statistics.lr_slope(slice_arr),0,in_arr).flatten()
    return np.concatenate([mean_vect,std_vect,trend_vect])


def time_window_5point_summary(in_arr):
    ''' Construct a 5 point summary of a time window using the 5 statistical functions {MIN, MAX, MEDIAN, IQR, SLOPE}'''
    if in_arr.size==0:
        trend_vect=np.zeros((in_arr.shape[1]))
        median_vect=np.zeros((in_arr.shape[1]))
        iqr_vect=np.zeros((in_arr.shape[1]))
        min_vect=np.zeros((in_arr.shape[1]))
        max_vect=np.zeros((in_arr.shape[1]))
    else:
        median_vect=np.median(in_arr, axis=0).flatten()
        iqr_vect=sp_stats.iqr(in_arr,axis=0).flatten()
        trend_vect=np.apply_along_axis(lambda slice_arr: first_last_diff(slice_arr),0,in_arr).flatten()
        min_vect=np.min(in_arr, axis=0).flatten()
        max_vect=np.max(in_arr,axis=0).flatten()

    out_arr=np.concatenate([median_vect, iqr_vect, min_vect, max_vect, trend_vect])
    return out_arr

def time_window_5point_summary_NAN(in_arr):
    ''' Construct a 5 point summary of a time window using the 5 statistical functions {MIN, MAX, MEDIAN, IQR, SLOPE}'''
    median_vect=np.nanmedian(in_arr, axis=0).flatten()
    iqr_vect=sp_stats.iqr(in_arr,axis=0,nan_policy="omit").flatten()
    if in_arr.size==0:
        trend_vect=np.zeros(0)
    else:
        trend_vect=np.apply_along_axis(lambda slice_arr: first_last_diff_NAN(slice_arr),0,in_arr).flatten()
    min_vect=np.nanmin(in_arr, axis=0).flatten()
    max_vect=np.nanmax(in_arr,axis=0).flatten()
    return np.concatenate([median_vect, iqr_vect, min_vect, max_vect, trend_vect])

def time_window_5point_summary_non_robust(in_arr):
    ''' Construct a 5 point summary of a time window using the 5 statistical functions {MIN, MAX, MEAN, STD, SLOPE}'''
    if in_arr.size==0:
        mean_vect=np.zeros((in_arr.shape[1]))
        std_vect=np.zeros((in_arr.shape[1]))
        trend_vect=np.zeros((in_arr.shape[1]))
        min_vect=np.zeros((in_arr.shape[1]))
        max_vect=np.zeros((in_arr.shape[1]))
    else:
        mean_vect=np.mean(in_arr, axis=0).flatten()
        std_vect=np.std(in_arr,axis=0).flatten()
        trend_vect=np.apply_along_axis(lambda slice_arr: first_last_diff(slice_arr),0,in_arr).flatten()
        min_vect=np.min(in_arr, axis=0).flatten()
        max_vect=np.max(in_arr,axis=0).flatten()
    
    out_arr=np.concatenate([mean_vect, std_vect, min_vect, max_vect, trend_vect])
    return out_arr

def time_window_5point_summary_non_robust_NAN(in_arr):
    ''' Construct a 5 point summary of a time window using the 5 statistical functions {MIN, MAX, MEAN, STD, SLOPE}'''
    mean_vect=np.nanmean(in_arr, axis=0).flatten()
    std_vect=np.nanstd(in_arr,axis=0).flatten()
    if in_arr.size==0:
        trend_vect=np.zeros(0)
    else:
        trend_vect=np.apply_along_axis(lambda slice_arr: first_last_diff_NAN(slice_arr),0,in_arr).flatten()
    min_vect=np.nanmin(in_arr, axis=0).flatten()
    max_vect=np.nanmax(in_arr,axis=0).flatten()
    return np.concatenate([mean_vect, std_vect, min_vect, max_vect, trend_vect])

def test_welford_alg():
    ''' Small test case to verify correctness of Welford's algorithm'''
    X=nprand.rand(20,10)
    mean_arr=np.mean(X,axis=0)
    std_arr=np.std(X,axis=0)
    n_online=0
    online_mean=np.zeros(X.shape[1])
    online_mqsr=np.zeros(X.shape[1])

    for row_idx in np.arange(X.shape[0]):
        n_online+=1
        delta=X[row_idx,:]-online_mean
        online_mean+=delta/n_online
        delta2=X[row_idx,:]-online_mean
        online_mqsr+=np.multiply(delta,delta2)

    train_mean=online_mean
    train_std=np.sqrt(online_mqsr/n_online)
    machine_eps=1e-15
    assert((np.abs(train_mean-mean_arr)<=machine_eps).all())
    assert((np.abs(train_std-std_arr)<=machine_eps).all());
