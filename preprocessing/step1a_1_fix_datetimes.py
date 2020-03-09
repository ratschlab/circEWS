#!/usr/bin/env python
import os
import gc

import pandas as pd
import numpy as np

import sys
sys.path.append('../utils')
import preproc_utils

import matplotlib.pyplot as plt


def date_correction(new_year, new_month, new_day, old_month):
    month30 = [4, 6, 9, 11]
    month31 = [1, 3, 5, 7, 8, 10, 12]
    is_leap_year = lambda year: year%4 == 0
    
    # Correct the day value if the new day is equal to 0 or larger than 31.
    if new_day == 0:
        new_month-=1
        if new_month in month30: 
            new_day = 30 
        elif new_month in month31:
            new_day = 31
        else:
            new_day = 29 if is_leap_year(new_year) else 28

    elif new_day > 31:
        if new_month in month30:       
            new_day -= 30 
        elif new_month in month31:
            new_day -= 31
        else:
            new_day -= 29 if is_leap_year(new_year) else 28
        new_month+=1
            
    # Correct the month value is the new month is non-positive or larger than 12
    if new_month <=0:
        new_month += 12
        new_year -= 1
        
    elif new_month > 12:
        new_month = new_month%12
        new_year += 1
    
    # If the old date is the end of the old month, make the new date the end of the new month to be consistent
    if new_day in [30, 31] and new_month == 2:
        new_day = 29 if is_leap_year(new_year) else 28
        
    elif new_day == 31 and new_month in month30:
        new_day = 30
        
    elif new_day == 30 and new_month in month31 and old_month in month30:
        new_day = 31
        
    elif new_day == 28 and new_month!=2 and old_month == 2:
        new_day = 31 if new_month in month31 else 30
        
    return new_year, new_month, new_day


def DISPLAY_BEFORE(idx, show_case, case, where='pre'):
    if show_case in [0, case]:
        print('===== Patient %d | CASE %d | Admission Time %s ====='%(pid, case, general.loc[pid].AdmissionTime))
        print('----- BEFORE -----')
        if type(idx) in [int, np.int64]:
            idx = [idx]
        for i in idx:
            if where=='pre':
                display(df.iloc[:i+6])
            elif where=='center':
                display(df.iloc[i-5:i+6] if i>5 else df.iloc[:i+6])
            elif where=='post':
                display(df.iloc[i-5:])
        if tbl_name == 'pharmarec':
            for i in idx:
                vid = df.iloc[i+1].VariableID
                iid = df.iloc[i+1].InfusionID
                display(df[np.logical_and(df.VariableID==vid, df.InfusionID==iid)])


def DISPLAY_AFTER(idx, show_case, case, where='pre'):
    if show_case in [0, case]:
        print('----- AFTER -----')
        if type(idx) in [int, np.int64]:
            idx = [idx]
        for i in idx:
            if where=='pre':
                display(df.iloc[:i+6])
            elif where=='center':
                display(df.iloc[i-5:i+6] if i>5 else df.iloc[:i+6])
            elif where=='post':
                display(df.iloc[i-5:])
        if tbl_name == 'pharmarec':
            for i in idx:
                vid = df.iloc[i+1].VariableID
                iid = df.iloc[i+1].InfusionID
                display(df[np.logical_and(df.VariableID==vid, df.InfusionID==iid)])

fix_year = lambda x, new_year: x.replace(year=new_year)
fix_month = lambda x, new_month: x.replace(month=new_month)


def fix_df_with_very_long_gap(show_case=None):
    AdmTime = np.datetime64(general.loc[pid].AdmissionTime.date())
    index_long_gap = np.where(diff_dt > 31)[0]
    
    
    if len(index_long_gap)==1:
        # If there is only one such gap 
        idx = index_long_gap[0]
        
        # Compute the difference in the year, month and day of the Datetime right before and right after the gap
        dt_gap_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx].Datetime.year 
        dt_gap_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx].Datetime.month
        dt_gap_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx].Datetime.day    

        # Datetime and Entertime difference before the gap
        be_dt_et_yr = df.iloc[idx].Datetime.year - df.iloc[idx].EnterTime.year
        be_dt_et_mo = df.iloc[idx].Datetime.month - df.iloc[idx].EnterTime.month
        be_dt_et_dd = df.iloc[idx].Datetime.day - df.iloc[idx].EnterTime.day

        # Datetime and Entertime difference after the gap
        af_dt_et_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx+1].EnterTime.year
        af_dt_et_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx+1].EnterTime.month
        af_dt_et_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx+1].EnterTime.day

        # DT is short for Datetime, ET for EnterTime and AT for AdmissionTime
        if be_dt_et_yr!=0 and be_dt_et_mo == 0:
            # Case 1: Different in year, the same in month for DT and ET before the gap
            # Solution: Replace the DT year with the ET year in records before the gap
            case = 1
            DISPLAY_BEFORE(idx, show_case, case)

            new_yr = df.iloc[idx].EnterTime.year
            idx2fix = df.index[:idx+1]
            if df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,)).max() - df.Datetime.max() < 7 * time_unit:
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,))
            else:
                df.drop(df.index[:idx+1], inplace=True)
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif af_dt_et_yr!=0 and af_dt_et_mo == 0:
            # Case 2: Different in year, the same in month for DT and ET after the gap
            # Solution: Replace the DT year with the ET year in records after the gap
            case = 2
            DISPLAY_BEFORE(idx, show_case, case)
                
            new_yr = df.iloc[idx+1].EnterTime.year
            idx2fix = df.index[idx+1:]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,))
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif be_dt_et_yr!=0 and df.iloc[idx].EnterTime.year==df.iloc[idx+1].Datetime.year:
            # Case 3: Different DT year and ET year before the gap. And the ET year before the gap is the same 
            #         as the DT year after the gap, which means the ET is continuous when there is a gap in DT.
            # Solution: Delete records with DT before the AT.
            case = 3
            DISPLAY_BEFORE(idx, show_case, case)
                
            df.drop(df.index[df.Datetime<AdmTime], inplace=True)
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif dt_gap_yr!=0 and dt_gap_mo == 0 and af_dt_et_yr==0:
            # Case 4: The gap is only caused by DT year difference between two consecutive records, and the DT 
            #         and ET year after the gap is consistent. 
            # Solution: Replace the DT year in records before the gap with the DT year after the gap.
            case = 4
            DISPLAY_BEFORE(idx, show_case, case)
            
            new_yr = df.iloc[idx+1].Datetime.year
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,))                    
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif dt_gap_yr!=0 and dt_gap_mo == 0 and be_dt_et_yr==0:
            # Case 5: The gap is only caused by DT year difference between two consecutive records, and the DT 
            #         and ET year before the gap is consistent. 
            # Solution: Replace the DT year in records after the gap with the DT year before the gap.
            case = 5
            DISPLAY_BEFORE(idx, show_case, case)
                
            new_yr = df.iloc[idx].Datetime.year
            idx2fix = df.index[idx+1:]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,)) 
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif be_dt_et_yr!=0 or af_dt_et_yr!=0 or dt_gap_yr!=0:
            # ???
            case = 16
            DISPLAY_BEFORE(idx, show_case, case, where='center')
                
            df.drop(df.index[idx+1:], inplace=True)
            
            DISPLAY_AFTER(idx, show_case, case, where='center')
        
        elif be_dt_et_mo!=0 and np.abs(be_dt_et_dd)<2:
            # All the previous cases have coverd case where there is a year difference, from now on, all cases
            # do not have year difference.
            # Case 6: When there is month difference in DT and ET and the absolute day difference is smaller 
            #         2 days before the gap, it is very likely there is a typo in the month before the gap.
            # Solution: Replace the DT month with the ET month in records before the gap.
            case = 6
            DISPLAY_BEFORE(idx, show_case, case)
                
            new_mo = df.iloc[idx].EnterTime.month
            if tbl_name in ['monvals', 'observrec']:
                if df.iloc[idx].Datetime.hour - df.iloc[idx].EnterTime.hour > 22:
                    new_dd = df.iloc[idx].Datetime.day
                else:
                    new_dd = df.iloc[idx].EnterTime.day
                fix_func = lambda x: x.replace(month=new_mo, day=new_dd)
            else:
                fix_func = lambda x: x.replace(month=new_mo)
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif be_dt_et_mo!=0 and dt_gap_dd==0:
            # Case 7: When there is month difference in DT and ET before the gap and no day difference between
            #         the records before and after the gap, it is very likely there is a typo in the month before
            #         the gap. 
            # Solution: Replace the DT month before the gap with the DT month after the gap.
            case = 7
            DISPLAY_BEFORE(idx, show_case, case)
                
            new_mo = df.iloc[idx+1].Datetime.month
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_month, args=(new_mo,))
            
            DISPLAY_AFTER(idx, show_case, case)
        
        elif af_dt_et_mo!=0 and af_dt_et_dd==0:
            # Case 8: When there is month difference and no day difference in DT and ET, it is very likely there 
            #         is a typo in the month after the gap.
            # Solution: Replace the DT month with the ET month in records before the gap.
            case = 8
            DISPLAY_BEFORE(idx, show_case, case, where='post')
                
            new_mo = df.iloc[idx+1].EnterTime.month
            idx2fix = df.index[idx+1:]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_month, args=(new_mo,))
            
            DISPLAY_AFTER(idx, show_case, case, where='post')
        
        elif np.abs(be_dt_et_mo)==11:
            # Case 9: When the absolute month difference is 11 but no year difference before the gap, it means 
            #         that the records should be one year more to be continuous.
            # Solution: Add 1 year to the DT year before the gap.
            case = 9
            DISPLAY_BEFORE(idx, show_case, case)
                
            new_yr = df.iloc[idx].EnterTime.year + 1
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_year, args=(new_yr,))
            
            DISPLAY_AFTER(idx, show_case, case)

        elif np.abs(af_dt_et_mo)==11:
            # Case 10: When the absolute month difference is 11 but no year difference after the gap, it means 
            #          that the records should be one year less to be continuous.
            # Solution: Subtract 1 year to the DT year before the gap.
            case = 10
            DISPLAY_BEFORE(idx, show_case, case, where='post')
                
            new_yr = df.iloc[idx+1].EnterTime.year - 1
            fix_func = lambda x: x.replace(year=new_yr, day=x.day+1 if x.day<=30 else x.day)
            idx2fix = df.index[idx+1:]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            
            DISPLAY_AFTER(idx, show_case, case, where='post')

        elif ( np.abs(df.iloc[idx].Datetime.month-df.iloc[idx+1].Datetime.day)%10==0 and 
               np.abs(df.iloc[idx].Datetime.day-df.iloc[idx+1].Datetime.month) <=1 ):
            # Case 11: When the month and day before the gap are flipped.
            # Solution: Replace the month and day in records before the gap with the month and day after the gap.
            case = 11
            DISPLAY_BEFORE(idx, show_case, case, where='center')
                
            new_mo = df.iloc[idx+1].Datetime.month
            new_dd = df.iloc[idx+1].Datetime.day
            fix_func = lambda x: x.replace(month=new_mo, day=new_dd)
            idx2fix = df.index[:idx+1]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            
            DISPLAY_AFTER(idx, show_case, case, where='center')

        elif (len(df) - idx - 1)/len(df) < 0.05:
            # Case 12: When there are very few records after the long gap
            case = 12
            DISPLAY_BEFORE(idx, show_case, case, where='center')
            df.drop(df.index[idx+1:], inplace=True)                
            DISPLAY_AFTER(idx, show_case, case, where='center')

        elif df.iloc[0].Datetime >= AdmTime:
            # Case 13: When the DT of the first record is later than AT. 
            # Solution: Remove the records after the gap.
            case = 13
            DISPLAY_BEFORE(idx, show_case, case, where='post')
            if tbl_name  == 'monvals':
                df.drop(df.index[idx+1:], inplace=True)
            else:
                for k in np.arange(idx+1, len(df)):
                    tmp_diff_et_dt_yr = df.iloc[k].Datetime.year - df.iloc[k].EnterTime.year
                    tmp_diff_et_dt_mo = df.iloc[k].Datetime.month - df.iloc[k].EnterTime.month
                    tmp_diff_et_dt_dd = df.iloc[k].Datetime.day - df.iloc[k].EnterTime.day
                    idx2fix = df.index[k]
                    if tmp_diff_et_dt_yr == 0 and tmp_diff_et_dt_dd == 0:
                        new_mo = df.iloc[k].EnterTime.month
                        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(month=new_mo)
                    elif np.abs(tmp_diff_et_dt_mo) > 9 and np.abs(tmp_diff_et_dt_dd) > 27:
                        new_yr = df.iloc[k].EnterTime.year
                        new_mo = df.iloc[k].EnterTime.month - 1
                        new_dd = df.iloc[k].Datetime.day
                        old_month = df.iloc[k].Datetime.month
                        new_yr, new_mo, new_dd = date_correction(new_yr, new_mo, new_dd, old_month)
                        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr,
                                                                                        month=new_mo,
                                                                                        day=new_dd)
                DISPLAY_AFTER(idx, show_case, case, where='post')
            
        else:
            # Case 14: Some really weird cases.
            case = 14
            DISPLAY_BEFORE(idx, show_case, case, where='center')
                    
            str_mo = '{0:02}'.format(df.iloc[idx].Datetime.month)
            str_dd = '{0:02}'.format(df.iloc[idx].Datetime.day)
            new_mo = int(str_mo[1]+str_dd[1])
            new_dd = int(str_mo[0]+str_dd[0])
            if np.abs(df.iloc[idx+1].Datetime.month-new_mo) <=1 and np.abs(df.iloc[idx+1].Datetime.day-new_dd) <= 1:
                new_mo = df.iloc[idx+1].Datetime.month
                fix_func = lambda x: x.replace(month=new_mo, day=new_dd)
                idx2fix = df.index[:idx+1]
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
            else:
                df.drop(df.index[df.Datetime<AdmTime], inplace=True)                
                DISPLAY_AFTER(idx, show_case, case, where='post')


    else:
        # Case 15: A case where there are two gaps.
        # Solution: Remove the records before AT
        case = 15
        DISPLAY_BEFORE(index_long_gap, show_case, case, where='center')
        if np.sum(index_long_gap / len(df) > 0.95) > 0:
            idx = index_long_gap[index_long_gap / len(df) > 0.95][0]
            df.drop(df.index[idx+1:], inplace=True)
        df.drop(df.index[df.Datetime<AdmTime], inplace=True)
    return df, case


def fix_df_with_long_gap(show_case=None):
    time_unit = np.timedelta64(1,'h')
    AdmTime = np.datetime64(general.loc[pid].AdmissionTime.date())
    index_long_gap = np.where(np.logical_and(diff_dt>1, diff_dt<=31))[0]
    index_gap2drop = []

    for k, idx in enumerate(index_long_gap):
        if (np.abs(df.iloc[idx].Datetime-df.iloc[idx].EnterTime)/time_unit<=24 and 
            np.abs(df.iloc[idx+1].Datetime-df.iloc[idx+1].EnterTime)/time_unit<=24):
            # When the difference in DT and ET of the record before and after the gap is reasonably 
            # smaller than a day, it means that the gap is reasonable and not caused by a typo in DT.
            # These reasonable gaps are the not the cases that we would like to correct, therefore, they
            # are deleted from the list of gaps of interest.
            index_gap2drop.append(idx)
    if len(index_gap2drop) == len(index_long_gap):
        # Case 1: All gaps are reasonable gaps. But there might still be chunks of records before AT or after
        # the max DT of the monvals records, those chunks should be deleted.
        case = 1
        DISPLAY_BEFORE(index_long_gap, show_case, case, where='center')
        
        index_df2drop = []
        if tbl_name != 'monvals':
            max_monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                         where='PatientID=%d'%pid, mode='r').Datetime.max().date()
        
        idx_not_reasonable = []
        for idx in index_long_gap:
            if df.iloc[idx].Datetime.date() < AdmTime:
                # If the chunk of records are observed before AT, it should be deleted
                index_df2drop.extend( df.index[:idx+1] )
                idx_not_reasonable.append(idx)
            if tbl_name != 'monvals' and df.iloc[idx+1].Datetime.date() > max_monvals_dt:
                # If the chunk of records are observed after the max DT of monvals records, it should be deleted
                # as well
                index_df2drop.extend( df.index[idx+1:] )
                idx_not_reasonable.append(idx)
            if (len(df) - idx - 1) < 5 and (len(df) - idx - 1 ) / len(df) < 0.05:
                index_df2drop.extend( df.index[idx+1:] )
                idx_not_reasonable.append(idx)                
                
        if len(index_df2drop)>0:
            # Delete records if there are records to delete
            index_df2drop = np.unique(index_df2drop)
            df.drop(index_df2drop, inplace=True)
            DISPLAY_AFTER(np.unique(idx_not_reasonable), show_case, case, where='center')


        return df, case
    
    if len(index_gap2drop) > 0:
        # If only some gaps are reasonable, delete thme from the list of gaps of interest and continue
        index_long_gap = np.sort(list(set(index_long_gap) - set(index_gap2drop)))

    if len(index_long_gap) > 1:
        # Case 2: When there is more than 1 gap of interest. We look at each chunk separated by the gaps, if the 
        # difference between DT and ET of each record within a chunk is smaller than 36 hours, we consider them
        # to be normal without typos (see case 2.1). When there is no day difference but month difference between 
        # DT and ET of each record within a chunk, then we assume there is a typo in the DT month of records which
        # will be corrected to the ET month of the corresponding record (see case 2.2). When the time span of ET
        # of records is smaller than 6 hours, but the time span of DT is larger than a day, it means that these 
        # records are history long before the admission and were entered within a short period; these records 
        # should also be deleted because there are no monvals observation in that historical period (see case 2.3).
        # When the current chunk is measured before AT, when the DT is after ET or when there is a month difference 
        # in the DT and ET, the chunk of records looks weird and not correctable, thus deletable (see case 2.4).
        # And for the remaining cases, they are usually reasonable chunks therefore no correction is needed. 
        case = 2
        DISPLAY_BEFORE(index_long_gap, show_case, case, where='center')


        index_long_gap = np.concatenate(([0], index_long_gap+1, [len(df)]))
        index_df2drop = []
        for i in range(len(index_long_gap)-1):
            tmp = df.iloc[index_long_gap[i]:index_long_gap[i+1]].copy()
            get_mo = lambda x: x.month
            get_dd = lambda x: x.day
            if np.abs(tmp.Datetime - tmp.EnterTime).max() / np.timedelta64(1, 'h') <= 36:
                # case 2.1
                pass
            
            elif (np.sum(tmp.Datetime.apply(get_mo) - tmp.EnterTime.apply(get_mo)==0) == 0 and 
                np.sum(np.abs(tmp.Datetime.apply(get_dd) - tmp.EnterTime.apply(get_dd))>1) == 0):
                # case 2.2
                for k in range(len(tmp)):
                    new_yr = tmp.iloc[k].EnterTime.year
                    new_mo = tmp.iloc[k].EnterTime.month
                    df.loc[tmp.index[k],'Datetime'] = tmp.iloc[k].Datetime.replace(year=new_yr,
                                                                                   month=new_mo)
            elif ((tmp.iloc[-1].EnterTime - tmp.iloc[0].EnterTime) / time_unit<=6 and 
                  (tmp.iloc[-1].Datetime - tmp.iloc[0].Datetime) / time_unit>24 ):
                # case 2.3
                index_df2drop.extend(tmp.index)
                
            elif (tmp.Datetime.max() < AdmTime or 
                  np.sum((tmp.Datetime - tmp.EnterTime)/time_unit < 0) == 0 or
                  np.sum(tmp.Datetime.apply(get_mo) - tmp.EnterTime.apply(get_mo)==0) == 0):
                # case 2.4
                index_df2drop.extend(tmp.index)

        if len(index_df2drop) > 0:
            df.drop(index_df2drop, inplace=True)
            DISPLAY_AFTER(index_long_gap[1:-1], show_case, case, where='center')

        return df, case

    idx = index_long_gap[0]
    ### The remaining cases only have one gap of interest
    ### We first look at the record before the gap, then we look at the record after the gap
    dt_gap_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx].Datetime.month
    dt_gap_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx].Datetime.day

    be_dt_et_yr = df.iloc[idx].Datetime.year - df.iloc[idx].EnterTime.year
    be_dt_et_mo = df.iloc[idx].Datetime.month - df.iloc[idx].EnterTime.month
    be_dt_et_dd = df.iloc[idx].Datetime.day - df.iloc[idx].EnterTime.day
    be_dt_et_hr = df.iloc[idx].Datetime.hour - df.iloc[idx].EnterTime.hour

    af_dt_et_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx+1].EnterTime.year
    af_dt_et_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx+1].EnterTime.month
    af_dt_et_dd = df.iloc[idx+1].Datetime.day - df.iloc[idx+1].EnterTime.day
    af_dt_et_hr = df.iloc[idx+1].Datetime.hour - df.iloc[idx+1].EnterTime.hour
    
        
    if (((be_dt_et_yr == 0 and be_dt_et_mo != 0) or 
         (np.abs(be_dt_et_yr) == 1 and np.abs(be_dt_et_mo) == 11)) and 
        np.abs(be_dt_et_dd) < 2):
        # Case 3: When there is almost no difference in day between the DT and ET of the record before the gap,
        # but the month value is off, it means that the month of all records before the gap are wrong. So we 
        # correct the month of those records by subtracting the year and month difference of the record before
        # the gap from all records before the gap
        case = 3
        DISPLAY_BEFORE(idx, show_case, case, where='post')
        dt_gap_yr = df.iloc[idx+1].Datetime.year - df.iloc[idx].Datetime.year
        et_gap_yr = df.iloc[idx+1].EnterTime.year - df.iloc[idx].EnterTime.year
        

        dt_gap_mo = df.iloc[idx+1].Datetime.month - df.iloc[idx].Datetime.month
        et_gap_mo = df.iloc[idx+1].EnterTime.month - df.iloc[idx].EnterTime.month

        if (et_gap_yr != 0 and dt_gap_yr) == 0 or (et_gap_mo != 0 and dt_gap_mo == 0) :
            if len(df) - idx - 1 < 5 and (len(df) - idx - 1) / len(df) < 0.05:
                df.drop(df.index[idx+1:], inplace=True)
                DISPLAY_AFTER(idx, show_case, case, where='post')
        else:
            for k in range(idx+1):
                new_yr = df.iloc[k].Datetime.year - be_dt_et_yr
                new_mo = df.iloc[k].Datetime.month - be_dt_et_mo
                new_dd = df.iloc[k].Datetime.day
                old_mo = df.iloc[k].Datetime.month

                new_yr, new_mo, new_dd = date_correction(new_yr, new_mo, new_dd, old_mo)
                idx2fix = df.index[k]
                df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr, 
                                                                                month=new_mo, 
                                                                                day=new_dd)
            DISPLAY_AFTER(idx, show_case, case, where='post')

    elif (np.abs(be_dt_et_dd) >= 9 and 
          np.abs(be_dt_et_dd)%10 in [0,1,9] and
          np.abs(df.iloc[idx].Datetime - df.iloc[idx].EnterTime) / time_unit > 24 * 9):
        # Case 4: When the day between the DT and ET of the record before the gap is off by multiple of 10 days.
        case = 4
        DISPLAY_BEFORE(idx, show_case, case)
        if df.iloc[idx].Datetime.date() == AdmTime:
            pass
        else:
            new_diff_dd = -31 if be_dt_et_dd == -30 else int( np.round(be_dt_et_dd/10) * 10)
            if be_dt_et_mo==0:
                fix_func = lambda x: x-np.timedelta64(1,'D')*new_diff_dd
            else:
                fix_func = lambda x: x.replace(month=df.iloc[idx].EnterTime.month)-np.timedelta64(1,'D')*new_diff_dd

            df.loc[df.index[:idx+1],'Datetime'] = df.iloc[:idx+1].Datetime.apply(fix_func)

            if tbl_name != 'monvals':
                max_monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                             where='PatientID=%d'%pid, mode='r').Datetime.max().date()
                if df.Datetime.max().date() > max_monvals_dt:
                    df.drop(df.index[df.Datetime>np.datetime64(max_monvals_dt)], inplace=True)

            DISPLAY_AFTER(idx, show_case, case)
            
    elif (np.abs(be_dt_et_dd)==27 and 
          df.iloc[idx].Datetime.month == 2 and 
          df.iloc[idx].EnterTime.month==2): 
        # Case 5: When the DT is supposed to be March 1st but misentered as Jan 1st when the ET is Feb 28th, which
        # means that the DT should be the next day of after the ET.
        case = 5
        DISPLAY_BEFORE(idx, show_case, case)

        fix_func = lambda x: x-np.timedelta64(1,'D')*(be_dt_et_dd-1)
        idx2fix = df.index[:idx+1]
        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
        
        DISPLAY_AFTER(idx, show_case, case)

        # When there is only one gap in the time series. 
    elif (((af_dt_et_yr == 0 and af_dt_et_mo != 0) or 
           (np.abs(af_dt_et_yr) == 1 and np.abs(af_dt_et_mo) == 11)) and 
          np.abs(af_dt_et_dd) < 2):
        # Case 6: When there is almost no day difference in DT and ET of the record after the gap, but the month
        # is off, so there might be a typo in the month. So correct the month of DT. 
        case = 6
        DISPLAY_BEFORE(idx, show_case, case, where='post')

        for k in np.arange(idx+1, len(df)):
            new_yr = df.iloc[k].Datetime.year - af_dt_et_yr
            new_mo = df.iloc[k].Datetime.month - af_dt_et_mo
            new_dd = df.iloc[k].Datetime.day
            old_mo = df.iloc[k].Datetime.month
            
            new_yr, new_mo, new_dd = date_correction(new_yr, new_mo, new_dd, old_mo)
            idx2fix = df.index[k]
            df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].replace(year=new_yr, 
                                                                            month=new_mo, 
                                                                            day=new_dd)
            
        DISPLAY_AFTER(idx, show_case, case, where='post')

    elif (np.abs(af_dt_et_dd)>=9 and 
          np.abs(af_dt_et_dd)%10 in [0,1,9] and
          np.abs(df.iloc[idx+1].Datetime - df.iloc[idx+1].EnterTime) / time_unit > 24 * 9):
        # Case 7: When the day between the DT and ET of the record after the gap is off by multiple of 10 days.
        case = 7
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        if len(df) - idx - 1 < 5 and (len(df) - idx - 1) / len(df) < 0.05:
            df.drop(df.index[idx+1:], inplace=True)
            DISPLAY_AFTER(idx, show_case, case, where='center')
        else:
            new_diff_dd = -31 if af_dt_et_dd == -30 else int( np.round(af_dt_et_dd/10) * 10)
            fix_func = lambda x: x-np.timedelta64(1,'D')*new_diff_dd
            df.loc[df.index[idx+1:],'Datetime'] = df.iloc[idx+1:].Datetime.apply(fix_func)

            is_reduce_day = np.logical_and(df.iloc[idx+1:].Datetime > df.iloc[idx+1:].EnterTime, 
                                           df.iloc[idx+1:].Datetime.apply(lambda x: x.hour)==23)
            index_reduce_day = df.index[idx+1:][is_reduce_day]
            if len(index_reduce_day) > 0:
                fix_func = lambda x: x-np.timedelta64(1,'D')
                df.loc[index_reduce_day,'Datetime'] = df.loc[index_reduce_day,'Datetime'].apply(fix_func)
                DISPLAY_AFTER(idx, show_case, case, where='center')

    elif (np.abs(af_dt_et_dd)==27 and 
          df.iloc[idx+1].Datetime.month == 2 and 
          df.iloc[idx+1].EnterTime.month==2): 
        # Case 8: when the DT of the record after the gap is supposed to be March 1st but is written as Feb 1st.
        case = 8
        DISPLAY_BEFORE(idx, show_case, case, where='post')
            
        fix_func = lambda x: x-np.timedelta64(1,'D')*(af_dt_et_dd-1)
        idx2fix = df.index[idx+1:]
        df.loc[idx2fix,'Datetime'] = df.loc[idx2fix,'Datetime'].apply(fix_func)
        
        DISPLAY_AFTER(idx, show_case, case, where='post')

    elif (df.iloc[0].Datetime >= AdmTime and
          tbl_name == 'pharmarec'):
        # Case 9: When the DT of the record is later than AT and the table is pharmarec, if the status is 780, 
        # just correct the day of the DT to the day of ET; but if the status is not 780, then compute the rate 
        # of the drug and if the rate makes the std of rate smaller than no change, otherwise replace the last 
        # rate with the median rate and recompute the last time point.
        case = 9
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        pass
    
    elif df.iloc[0].Datetime >= AdmTime:
        # Case 10: if the DT of the first record is later than AT and the table is not pharmarec, then if 
        # the DT of the last record with the DT of the last record in monvals, then delete all records after the 
        # gap.
        case = 10
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        if tbl_name == 'monvals':
            if (idx+1) / len(df) > 0.95:
                df.drop(df.index[idx+1:], inplace=True)
                DISPLAY_AFTER(idx, show_case, case, where='center')
        else:
            monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                     where='PatientID=%d'%pid, mode='r').Datetime

            if len(monvals_dt) > 0:
                diff_monvals_dt = np.diff(np.sort(monvals_dt)) / np.timedelta64(1, 'h')
                if (np.argmax(diff_monvals_dt)+1) / len(diff_monvals_dt) > 0.95 and (idx+1) / len(df) > 0.95:
                    df.drop(df.index[idx+1:], inplace=True)
                    DISPLAY_AFTER(idx, show_case, case, where='center')
                else:
                    max_monvals_dt = monvals_dt.max().date()
                    if df.iloc[idx+1].Datetime.date() > max_monvals_dt:
                        df.drop(df.index[idx+1:], inplace=True)
                        DISPLAY_AFTER(idx, show_case, case, where='center')

    elif df.iloc[idx].Datetime < AdmTime:
        # Case 11: When the DT before the gap is earlier than AT. If the percentage of records is less than 10% of
        # the time series, then delete all records before the gap, otherwise if the DT before the gap is earlier
        # than the min DT from monvals or the DT after the gap is later than max DT from monvals, then delete the 
        # records before or after the gap.
        case = 11
        DISPLAY_BEFORE(idx, show_case, case, where='center')
        
        if idx / len(df) < 0.1:
            df.drop(df.index[:idx+1], inplace=True)
            DISPLAY_AFTER(idx, show_case, case, where='center')
        elif tbl_name!='monvals':
            monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                     where='PatientID=%d'%pid, mode='r').Datetime
            min_monvals_dt = monvals_dt.min().date()
            max_monvals_dt = monvals_dt.max().date()
            index_df2drop = []
            print(min_monvals_dt)
            if df.iloc[idx].Datetime.date() < min_monvals_dt:
                index_df2drop.extend(df.index[:idx+1])
            if df.iloc[idx+1].Datetime.date() > max_monvals_dt:
                index_df2drop.extend(df.index[idx+1:])
            if len(index_df2drop) > 0:
                df.drop(index_df2drop, inplace=True)
                DISPLAY_AFTER(idx, show_case, case, where='center')

    else:
        # Case 12: The rest of the cases. If the DT of the record after the gap is later than the max DT 
        # of records in monvals then delete all records after the gap.
        case = 12
        DISPLAY_BEFORE(idx, show_case, case, where='post')
        
        if tbl_name!='monvals':
            max_monvals_dt = pd.read_hdf(os.path.join(input_path, 'monvals.h5'), columns=['Datetime'], 
                                         where='PatientID=%d'%pid, mode='r').Datetime.max().date()
            if df.iloc[idx+1].Datetime.date() > max_monvals_dt:
                df.drop(df.index[idx+1:], inplace=True)
                    
        DISPLAY_AFTER(idx, show_case, case, where='post')

    return df, case


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', help='excel ref version')
    parser.add_argument('-tbl_name', help='Table name')
    parser.add_argument('--index_chunk', type=int, default=0, help='Index of the chunk file.')
    parser.add_argument('--output_to_disk', action='store_true')
    args = parser.parse_args()

    version = args.version
    tbl_name = args.tbl_name
    index_chunk = args.index_chunk
    output_to_disk = args.output_to_disk

    if not output_to_disk:
        print('TEST MODE: WILL NOT WRITE THE RESULTS TO DISK.')

    # Data paths
    input_path = os.path.join(preproc_utils.datapath, '1_hdf5_consent', '180704')
    output_path = os.path.join(preproc_utils.datapath, '1a_hdf5_clean', 
                               version, 'datetime_fixed', tbl_name)
    if output_to_disk and not os.path.exists(output_path):
        os.makedirs(output_path)

    chunking_info = preproc_utils.get_chunking_info(version=version)
    pid_list = chunking_info.index[chunking_info.ChunkfileIndex==index_chunk].values

#     pid_list = pd.read_csv('VeryLongGap_observrec.csv', names=['PatientID']).PatientID.values

    output_path = os.path.join(output_path, '%s_%d_%d--%d.h5'%(tbl_name, 
                                                               index_chunk, 
                                                               np.min(pid_list), 
                                                               np.max(pid_list)))

    general = pd.read_hdf(os.path.join(input_path, 'generaldata.h5'), mode='r').set_index('PatientID')
    general = general[general.index.isin(pid_list)]
    gc.collect()

    time_unit = np.timedelta64(1, 'D')

    tmp_pids = []
    tmp_gaps = []

    pids2fix = []
    for n, pid in enumerate(pid_list):
        df = pd.read_hdf(os.path.join(input_path, '%s.h5'%tbl_name), where='PatientID=%d'%pid, mode='r') 
        if len(df) == 0:
            print('Patient', pid, 'has no data.')
            continue

        # Rename some columns so that the columns names are consistent across all tables
        df.rename(columns={'DateTime': 'Datetime', 'PharmaID': 'VariableID', 'SampleTime': 'Datetime', 
                           'GivenDose': 'Value', 'Entertime': 'EnterTime'}, inplace=True)

        # Remove records with NaN values
        df.drop(df.index[df.Value.isnull()], inplace=True)


        # Remove duplicates with the exact same values
        important_columns = ['Datetime', 'PatientID', 'VariableID', 'Value', 'Status']
        if tbl_name == 'pharmarec':
            important_columns.append('InfusionID')
        df.drop_duplicates(important_columns, inplace=True)

        gc.collect()
        if len(df) == 0:
            print('Patient', pid, 'have no valid data.')
            continue

        df.sort_values('Datetime', inplace=True)

        # The time difference between adjecent records in days
        diff_dt = np.diff(df.Datetime) / time_unit

        # If there exists gap longer than 31 days
        if np.sum(diff_dt > 31) > 0:
            pids2fix.append(pid)
            df, case = fix_df_with_very_long_gap()
            df.sort_values('Datetime', inplace=True)
            diff_dt = np.diff(df.Datetime) / time_unit

        # If there exists gap longer between 1 and 31 days
        if np.sum(np.logical_and(diff_dt>1, diff_dt<=31)) > 0:
            pids2fix.append(pid)
            df, case = fix_df_with_long_gap()
            df.sort_values('Datetime', inplace=True)
            diff_dt = np.diff(df.Datetime) / time_unit

        # Reverse the renaming for observrec table
        if tbl_name == 'pharmarec':
            df.rename(columns={'Value': 'GivenDose', 'VariableID': 'PharmaID'}, inplace=True)

        if output_to_disk:
            df.to_hdf(output_path, 'data', append=True, data_columns=True, 
                      complevel=5, complib='blosc:lz4', format='table')
            if tbl_name=='monvals':
                df_svo2 = df[df.VariableID==4200]
                if len(df_svo2) > 0:
                    df_svo2.to_hdf(output_path.replace('monvals', 'monvals_svo2'), 'data', 
                                   append=True, data_columns=True, format='table',
                                   complevel=5, complib='blosc:lz4')


        # if np.sum(diff_dt > 31) > 0:
        #     print('Patient %d still have gaps larger than a month.'%pid)
        #     tmp_diff_dt = diff_dt[diff_dt > 31]
        #     tmp_pids.extend(pid*np.ones((len(tmp_diff_dt),))) 
        #     tmp_gaps.extend(tmp_diff_dt)

        # if np.sum(np.logical_and(diff_dt > 1, diff_dt <= 31)) > 0:
        #     print('Patient %d still have gaps larger than a day.'%pid)
        #     tmp_diff_dt = diff_dt[np.logical_and(diff_dt > 1, diff_dt <= 31)]
        #     tmp_pids.extend(pid*np.ones((len(tmp_diff_dt),))) 
        #     tmp_gaps.extend(tmp_diff_dt)

        if (n+1)%50 == 0:
            print('%d / %d'%(n+1, len(pid_list)))
        gc.collect()

np.save('pids2fix_%s_%d.npy'%(tbl_name, index_chunk), np.unique(pids2fix))
# plt.figure()
# plt.hist(tmp_gaps)
# plt.show()
# plt.close()
# tmp_pids = np.array(tmp_pids)
# tmp_gaps = np.array(tmp_gaps)
# np.unique(tmp_pids[np.logical_and(tmp_gaps > 1, tmp_gaps <= 31)])

