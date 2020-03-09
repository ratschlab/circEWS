
''' Basic statistical functions'''

import functools
import scipy.stats as sp_stats
import numpy.linalg as np_la
import numba
import numpy as np

import circews.functions.algorithm as mlhc_algorithm

@numba.jit(nogil=True)
def energy(x):
    """Returns the energy contained in a signal

    :param x: Input signal

    """
    return np.mean(np.square(x))

def coeff_variation(x):
    return sp_stats.variation()

def pearson_corr(x,y):
    common_finite_arr=np.isfinite(x) & np.isfinite(y)
    if np.sum(common_finite_arr)==0:
        return np.nan
    else:
        x=x[common_finite_arr]
        y=y[common_finite_arr]
        return sp_stats.pearsonr(x,y)[0]

def energy_std(x):
    n=x.shape[0]
    energies = np.empty(shape=int(np.ceil(n / 6.)), dtype=np.float32)

    for j, i in enumerate(range(0, n, 6)):
        part_sig = x[i:i + 6]
        energies[j] = algorithm.energy(part_sig)
        
    energy_std = np.std(energies)
    return energy_std

def first_last_diff(x):
    return x[-1] - x[0]

def geometric_mean(x):
    if not (x <= 0).sum() % 2 == 0:
        return np.nan

    return sp_stats.gmean(x)

def ratio(x,y,eps=1e-05):
    ''' Returns ratio of two numbers '''
    if np.isnan(x) or np.isnan(y) or abs(x)<eps or abs(y)<eps:
        return np.nan
    return x/y

def iqr(x):
    ''' Returns the interquartile range of a sample, a statistic of the dispersion in the sample, robust to outliers'''
    return np.percentile(x,75) - np.percentile(x,25)

def nan_iqr(x):
    ''' Returns the interquartile range of a sample, while ignoring any NAN samples'''
    return sp_stats.iqr(x,nan_policy="omit")

def kurtosis(x):
    return sp_stats.kurtosis(x)

def line_length(x):
    return np.mean(np.abs(np.diff(x)))

def lr_slope(x):
    ''' Accepts a 1D array x and computes the linear regression line slope assuming unit spacing on the array '''
    finite_arr=np.isfinite(x)
    
    if np.sum(np.isfinite(x))<2:
        return np.nan

    t = np.arange(x.size)
    t = t[finite_arr]
    x = x[finite_arr]
    slope, _, _  = mlhc_algorithm.linregress(t, x)
    return slope

def maximum(x):
    return np.amax(x)

def minimum(x):
    return np.amin(x)

def mean(x):
    return np.mean(x)

def median(x):
    return np.median(x)

def minus(x,y):
    return x.toarray()-y.toarray()

def norm(x):
    return np_la.norm()

def signal_noise_ratio(x):
    a = np.asanyarray(x)
    m = a.mean(0)
    sd = a.std(axis=0, ddof=0)
    return np.where(sd == 0, 0, m / sd)

def skewness(x):
    return sp_stats.skew(x)

def std(x):
    return np.std(x)

def three_means(x):
    a, b, c = np.array_split(x, 3)
    return [np.mean(a), np.mean(b), np.mean(c)]

def variance(x):
    return np.var(x)

@numba.jit(nopython=False)
def sum_squared(X):
    """Utility function to find norm of array
    :param X: Input array
    """
    old_shape = X.shape
    X.shape = (-1,)
    ret = np.dot(X, X)
    X.shape = old_shape
    return ret
