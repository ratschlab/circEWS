''' Label functions'''

import sys
import os
import os.path
import ipdb
import datetime
import timeit
import random
import gc
import psutil
import csv
import glob

import pandas as pd
import numpy as np
import scipy as sp

import matplotlib
matplotlib.use("pdf")

import matplotlib.pyplot as plt

IMPUTE_GRID_PERIOD_SECS=300.0

def nan_exists_transition(event1_arr,event2_arr,event3_arr,
                          maybe1_arr,maybe2_arr,maybe3_arr,
                          not1_arr,not2_arr,not3_arr):
    assert(event1_arr.size==event2_arr.size==event3_arr.size==maybe1_arr.size==maybe2_arr.size==maybe3_arr.size)
    assert(event1_arr.size>0)
    output_arr=np.zeros_like(event1_arr)

    # Define starting out in any events as being a deterioration from no observation.
    if maybe1_arr[0]==1.0 or maybe2_arr[0]==1.0 or maybe3_arr[0]==1.0:
        output_arr[0]=np.nan

    elif event1_arr[0]==1.0 or event2_arr[0]==1.0 or event3_arr[0]==1.0:
        output_arr[0]=1.0
    
    for idx in np.arange(1,event1_arr.size):

        # Transition into any of the events from a lower severity level. From the mutual exclusivity this condition can 
        # be simplified by checking if no down-wards transition took place.
        if maybe1_arr[idx-1]==1.0 or maybe1_arr[idx]==1.0 or maybe2_arr[idx-1]==1.0 or maybe2_arr[idx]==1.0 or maybe3_arr[idx-1]==1.0 or maybe3_arr[idx]==1.0:
            output_arr[idx]=np.nan

        elif event1_arr[idx-1]==0.0 and event1_arr[idx]==1.0 and event2_arr[idx-1]==0.0 and event3_arr[idx-1]==0.0 or \
             event2_arr[idx-1]==0.0 and event2_arr[idx]==1.0 and event3_arr[idx-1]==0.0 or \
             event3_arr[idx-1]==0.0 and event3_arr[idx]==1.0:
            output_arr[idx]=1.0

    return output_arr


def patient_instability(event1_arr,event2_arr,event3_arr,maybe1_arr,maybe2_arr,maybe3_arr):
    assert(event1_arr.size==event2_arr.size==event3_arr.size==maybe1_arr.size==maybe2_arr.size==maybe3_arr.size)
    assert(event1_arr.size>0)
    output_arr=np.zeros_like(event1_arr)

    for idx in np.arange(event1_arr.size):

        if maybe1_arr[idx]==1.0 or maybe2_arr[idx]==1.0 or maybe3_arr[idx]==1.0:
            output_arr[idx]=np.nan

        if event1_arr[idx]==1.0 or event2_arr[idx]==1.0 or event3_arr[idx]==1.0:
            output_arr[idx]=1.0

    return output_arr



def any_positive_transition(event_arr, lhours, rhours, grid_step_seconds):
    assert(rhours>=lhours)
    gridstep_per_hours=int(3600/grid_step_seconds)
    out_arr=np.zeros_like(event_arr)
    sz=event_arr.size

    for idx in range(event_arr.size):
        event_val=event_arr[idx]

        if np.isnan(event_val):
            out_arr[idx]=np.nan
            continue

            future_arr=event_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]
            
            if future_arr.size==0:
                continue

            elif np.isnan(future_arr).all():
                out_arr[idx]=np.nan

            if event_val==0.0 and (future_arr==1.0).any():
                out_arr[idx]=1.0

    return out_arr


def time_to_worse_state_binned(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    mins_per_gridstep=int(grid_step_secs/60)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size

    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]
        
        if e_val=="unknown" or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        if future_arr.size==0:
            out_arr[idx]=-1.0
            continue

        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                min_idxs=[]
                
                if (future_arr=="event 1").any():
                    min_idxs.append(np.where(future_arr=="event 1")[0][0])

                if (future_arr=="event 2").any():
                    min_idxs.append(np.where(future_arr=="event 2")[0][0])

                if (future_arr=="event 3").any():
                    min_idxs.append(np.where(future_arr=="event 3")[0][0])
                
                time_to_det=mins_per_gridstep*np.min(min_idxs)
                quant_time_to_det=time_to_det-time_to_det%30
                out_arr[idx]=quant_time_to_det

        elif e_val=="event 1": 
            if (future_arr=="event 2").any() or (future_arr=="event 3").any() :
                min_idxs=[]

                if (future_arr=="event 2").any():
                    min_idxs.append(np.where(future_arr=="event 2")[0][0])

                if (future_arr=="event 3").any():
                    min_idxs.append(np.where(future_arr=="event 3")[0][0])

                time_to_det=mins_per_gridstep*np.min(min_idxs)
                quant_time_to_det=time_to_det-time_to_det%30
                out_arr[idx]=quant_time_to_det

        elif e_val=="event 2": 
            if (future_arr=="event 3").any():
                out_arr[idx]=1.0
                time_to_det=mins_per_gridstep*np.where(future_arr=="event 3")[0][0]
                quant_time_to_det=time_to_det-time_to_det%30
                out_arr[idx]=quant_time_to_det

        elif e_val=="event 3":
            out_arr[idx]=np.nan

    return out_arr


def time_to_worse_state(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    mins_per_gridstep=int(grid_step_secs/60)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size

    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]
        
        if e_val=="unknown" or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        if future_arr.size==0:
            out_arr[idx]=-1.0
            continue

        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                min_idxs=[]
                
                if (future_arr=="event 1").any():
                    min_idxs.append(np.where(future_arr=="event 1")[0][0])

                if (future_arr=="event 2").any():
                    min_idxs.append(np.where(future_arr=="event 2")[0][0])

                if (future_arr=="event 3").any():
                    min_idxs.append(np.where(future_arr=="event 3")[0][0])
                
                time_to_det=mins_per_gridstep*np.min(min_idxs)
                out_arr[idx]=time_to_det

        elif e_val=="event 1": 
            if (future_arr=="event 2").any() or (future_arr=="event 3").any() :
                min_idxs=[]

                if (future_arr=="event 2").any():
                    min_idxs.append(np.where(future_arr=="event 2")[0][0])

                if (future_arr=="event 3").any():
                    min_idxs.append(np.where(future_arr=="event 3")[0][0])

                time_to_det=mins_per_gridstep*np.min(min_idxs)
                out_arr[idx]=time_to_det

        elif e_val=="event 2": 
            if (future_arr=="event 3").any():
                time_to_det=mins_per_gridstep*np.where(future_arr=="event 3")[0][0]
                out_arr[idx]=time_to_det

        elif e_val=="event 3":
            out_arr[idx]=np.nan

    return out_arr


def exists_stable_to_event1_transition(event1_arr,event2_arr,event3_arr):
    assert(event1_arr.size==event2_arr.size==event3_arr.size)
    assert(event1_arr.size>0)
    output_arr=np.zeros_like(event1_arr)

    if event1_arr[0]==1.0 and event2_arr[0]==0.0 or event3_arr[0]==0.0:
        output_arr[0]=1.0

    for idx in np.arange(1,event1_arr.size):
        if event1_arr[idx-1]==0.0 and event1_arr[idx]==1.0 and event2_arr[idx-1]==0.0 and event2_arr[idx]==0.0 \
           and event3_arr[idx-1]==0.0 and event3_arr[idx]==0.0:
            output_arr[idx]=1.0
    
    return output_arr




def shifted_exists_future_interval(label_in_arr,forward_lbound,forward_rbound,invert_label=False):
    pos_label=0.0 if invert_label else 1.0
    gridstep_per_hours=int(3600/IMPUTE_GRID_PERIOD_SECS)
    output_arr=np.zeros_like(label_in_arr)

    for idx in np.arange(label_in_arr.size):

        full_sz=label_in_arr.size

        if forward_lbound==0:
            lwindow_idx=idx+1
        else:
            lwindow_idx=idx+int(forward_lbound*gridstep_per_hours)
        rwindow_idx=idx+int(forward_rbound*gridstep_per_hours)

        if lwindow_idx < full_sz:
            output_arr[idx]=1.0 if (label_in_arr[lwindow_idx:min(full_sz,rwindow_idx)]==pos_label).any() else 0.0
        else:
            output_arr[idx]=np.nan
    
    return output_arr


def time_to_event(label_in_arr,forward_rbound,invert_label=False):
    pos_label=0.0 if invert_label else 1.0
    gridstep_per_hours=12
    output_arr=np.zeros_like(label_in_arr)

    for idx in np.arange(label_in_arr.size):
        full_sz=label_in_arr.size
        lwindow_idx=idx+1
        rwindow_idx=idx+forward_rbound*gridstep_per_hours
        
        if lwindow_idx < full_sz:
            tent_event_arr=label_in_arr[lwindow_idx:min(full_sz,rwindow_idx)]
            event_idxs=np.argwhere(tent_event_arr==pos_label)
            if event_idxs.size==0:
                output_label=-1.0
            else:
                output_label=(event_idxs.min()+1)*5.0
            output_arr[idx]=output_label
        else:
            output_arr[idx]=np.nan
    
    return output_arr






def future_deterioration(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, 
                         pn1_arr, pn2_arr, pn3_arr, l_hours, r_hours, grid_step_secs):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros_like(event1_arr)
    sz=event1_arr.size
    
    for idx in range(event1_arr.size):
        e1_val=event1_arr[idx]
        e2_val=event2_arr[idx]
        e3_val=event3_arr[idx]
        m1_val=maybe1_arr[idx]
        m2_val=maybe2_arr[idx]
        m3_val=maybe3_arr[idx]

        # We cannot determine in which state we started off with...
        if np.isnan(e1_val) or np.isnan(e2_val) or np.isnan(e3_val) or m1_val==1.0 or m2_val==1.0 or m3_val==1.0:
            out_arr[idx]=np.nan
            continue

        lead1_arr=event1_arr[idx: min(sz, idx+gridstep_per_hours*l_hours)]
        lead2_arr=event2_arr[idx: min(sz, idx+gridstep_per_hours*l_hours)]
        lead3_arr=event3_arr[idx: min(sz, idx+gridstep_per_hours*l_hours)]

        future1_arr=event1_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future2_arr=event2_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future3_arr=event3_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]

        # No future to consider, => no deterioration
        if future1_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future1_arr).all() or np.isnan(future2_arr).all() or np.isnan(future3_arr).all() or \
             np.isnan(lead1_arr).all() or np.isnan(lead2_arr).all() or np.isnan(lead3_arr).all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e1_val==0.0 and e2_val==0.0 and e3_val==0.0:
            if ((future1_arr==1.0).any() or (future2_arr==1.0).any() or (future3_arr==1.0).any()) \
               and not (lead1_arr==1.0).any() and not (lead2_arr==1.0).any() and not (lead3_arr==1.0).any():
                out_arr[idx]=1.0

        # State 1: Low severity patient state
        elif e1_val==1.0: 
            if ((future2_arr==1.0).any() or (future3_arr==1.0).any()) \
               and not (lead2_arr==1.0).any() and not (lead3_arr==1.0).any():
                out_arr[idx]=1.0
            
        # State 2: Intermediate severity patient state
        elif e2_val==1.0: 
            if (future3_arr==1.0).any() and not (lead3_arr==1.0).any():
                out_arr[idx]=1.0


    return out_arr



def future_worse_state( endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        # We cannot determine in which state we started off with...
        if e_val=="unknown" or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

        # State 1: Low severity patient state
        elif e_val=="event 1": 
            if (future_arr=="event 2").any() or (future_arr=="event 3").any() :
                out_arr[idx]=1.0
            
        # State 2: Intermediate severity patient state
        elif e_val=="event 2": 
            if (future_arr=="event 3").any():
                out_arr[idx]=1.0

        # State 3: No deterioration from this level is possible, so we will not use these segments
        elif e_val=="event 3":
            out_arr[idx]=np.nan

    return out_arr


def future_worse_state_soft( endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        # We cannot determine in which state we started off with...
        if e_val=="unknown" or "maybe" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any() \
               or (future_arr=="maybe 1").any() or (future_arr=="maybe 2").any() or (future_arr=="maybe 3").any() \
               or (future_arr=="probably not 1").any() or (future_arr=="probably not 2").any() or (future_arr=="probably not 3").any():
                out_arr[idx]=1.0

        # State 0.5 Intermediate state
        elif e_val in ["probably not 1", "probably not 2", "probably not 3"]:
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

        # State 1: Low severity patient state
        elif e_val=="event 1": 
            if (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0
            
        # State 2: Intermediate severity patient state
        elif e_val=="event 2": 
            if (future_arr=="event 3").any():
                out_arr[idx]=1.0

        # State 3: No deterioration from this level is possible, so we will not use these segments
        elif e_val=="event 3":
            out_arr[idx]=np.nan

    return out_arr


def future_worse_state_from_0( endpoint_status_arr, l_hours, r_hours, grid_step_secs):

    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 1", "event 2", "event 3"] or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

    return out_arr


def future_worse_state_soft_from_0( endpoint_status_arr, l_hours, r_hours, grid_step_secs):

    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 1", "event 2", "event 3"] or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val=="event 0":
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any() \
               or (future_arr=="maybe 1").any() or (future_arr=="maybe 2").any() or (future_arr=="maybe 3").any() \
               or (future_arr=="probably not 1").any() or (future_arr=="probably not 2").any() or (future_arr=="probably not 3").any():
                out_arr[idx]=1.0

    return out_arr



def future_worse_state_from_pn(endpoint_status_arr, l_hours, r_hours, grid_step_secs):

    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 0","event 1", "event 2", "event 3"] or "maybe" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # Probably not pre-state
        if e_val in ["probably not 1", "probably not 2", "probably not 3"]:
            if (future_arr=="event 1").any() or (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

    return out_arr



def future_worse_state_from_1(endpoint_status_arr, l_hours, r_hours, grid_step_secs):

    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 0","event 2", "event 3"] or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # Event 1 state
        if e_val=="event 1":
            if (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

    return out_arr


def future_worse_state_from_2(endpoint_status_arr, l_hours, r_hours, grid_step_secs):

    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 0","event 1", "event 3"] or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # Event 2 state
        if e_val=="event 2":
            if (future_arr=="event 3").any():
                out_arr[idx]=1.0

    return out_arr


def future_worse_state_from_1_or_2(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if e_val in ["unknown","event 0","event 3"] or "maybe" in e_val or "probably not" in e_val:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif (future_arr=="unknown").all():
            out_arr[idx]=np.nan
            continue

        # Event 1 state
        if e_val=="event 1":
            if (future_arr=="event 2").any() or (future_arr=="event 3").any():
                out_arr[idx]=1.0

        if e_val=="event 2":
            if (future_arr=="event 3").any():
                out_arr[idx]=1.0

    return out_arr



def exists_stability_to_any(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, l_hours, r_hours):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/IMPUTE_GRID_PERIOD_SECS)
    out_arr=np.zeros_like(event1_arr)
    sz=event1_arr.size
    
    for idx in range(event1_arr.size):
        e1_val=event1_arr[idx]
        e2_val=event2_arr[idx]
        e3_val=event3_arr[idx]
        m1_val=maybe1_arr[idx]
        m2_val=maybe2_arr[idx]
        m3_val=maybe3_arr[idx]

        # We cannot determine in which state we started off with, or patient is currently not stable
        if np.isnan(e1_val) or np.isnan(e2_val) or np.isnan(e3_val) or m1_val==1.0 or m2_val==1.0 or m3_val==1.0 or e1_val==1.0 or e2_val==1.0 or e3_val==1.0:
            out_arr[idx]=np.nan
            continue

        future1_arr=event1_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future2_arr=event2_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]
        future3_arr=event3_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]

        # No future to consider, => no deterioration
        if future1_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future1_arr).all() or np.isnan(future2_arr).all() or np.isnan(future3_arr).all():
            out_arr[idx]=np.nan
            continue

        if (future1_arr==1.0).any() or (future2_arr==1.0).any() or (future3_arr==1.0).any():
            out_arr[idx]=1.0

    return out_arr
    


def exists_stability_to_1(event1_arr, event2_arr, event3_arr, maybe1_arr, maybe2_arr, maybe3_arr, l_hours, r_hours):
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/IMPUTE_GRID_PERIOD_SECS)
    out_arr=np.zeros_like(event1_arr)
    sz=event1_arr.size
    
    for idx in range(event1_arr.size):
        e1_val=event1_arr[idx]
        e2_val=event2_arr[idx]
        e3_val=event3_arr[idx]
        m1_val=maybe1_arr[idx]
        m2_val=maybe2_arr[idx]
        m3_val=maybe3_arr[idx]

        # We cannot determine in which state we started off with or patient is currently not stable
        if np.isnan(e1_val) or np.isnan(e2_val) or np.isnan(e3_val) or m1_val==1.0 or m2_val==1.0 or m3_val==1.0 or e1_val==1.0 or e2_val==1.0 or e3_val==1.0:
            out_arr[idx]=np.nan
            continue

        future1_arr=event1_arr[min(sz, idx+gridstep_per_hours*l_hours): min(sz, idx+gridstep_per_hours*r_hours)]

        # No future to consider, => no deterioration
        if future1_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future1_arr).all():
            out_arr[idx]=np.nan
            continue

        if (future1_arr==1.0).any():
            out_arr[idx]=1.0

    return out_arr







