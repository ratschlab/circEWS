
import itertools
import fractions
import multiprocessing
import contextlib
import math

import numba
import numpy as np
import scipy as sp
import scipy.interpolate as sp_ip
import scipy.cluster.vq as sq_clust_vq
import scipy.signal as sp_signal
import scipy.sparse as sp_sparse
import scipy.signal._arraytools as sp_sig_arraytools
import scipy.stats as sp_stats
import bottleneck as bn

def linregress(x, y=None):
    """
    Calculate a regression line
    """
    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            msg = ("If only `x` is given as input, it has to be of shape "
                   "(2, N) or (N, 2), provided shape was %s" % str(x.shape))
            raise ValueError(msg)
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    xmean = np.mean(x, None)
    ymean = np.mean(y, None)

    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

    slope = r_num / ssxm
    intercept = ymean - slope * xmean
    return slope, intercept, r

def arr_prefilled(sz,fill_val):
    ''' Returns a NP array prefilled with a specified value'''
    new_arr=np.empty(sz)
    new_arr.fill(fill_val)
    return new_arr

def nan_ratio(arr):
    ''' Returns the NAN ratio of an array '''
    nan_count=np.sum(np.isnan(arr))
    return nan_count,nan_count/arr.size
