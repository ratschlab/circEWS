#!/usr/bin/env python
import sys
sys.path.append('../../utils')

import numpy as np
import pandas as pd

from os import mkdir
from os.path import join, exists
from preproc_utils import voi_id_name_mapping
from preproc_utils import get_datapath

datapath = get_datapath()
outputpath = join(datapath, 'misc_derived', 'pharma')

if not exists(outputpath):
    mkdir(outputpath)

def get_reference(tbl_name, mode='derived'):
    if tbl_name not in ['unitref', 'pharmaref', 'DoseUnitChange', 'FormUnitChange']:
        raise Exception('Wrong reference table name.')
    if mode == 'derived':
        if tbl_name == 'unitref':
            ref_filename = 'derived-unitref.csv'
        elif tbl_name == 'pharmaref':
            ref_filename = 'standardized-pharmaref.csv'
        else:
            ref_filename = tbl_name+'.csv'            
        ref = pd.read_csv(join(outputpath, ref_filename), 
                          sep='\t', encoding='cp1252', parse_dates=['ArchTime'])
    else:
         ref = pd.read_csv(join(datapath, '0_csv_exports', 'expot-%s.csv'%tbl_name), 
                           sep='\t', encoding='cp1252', parse_dates=['ArchTime'])
    return ref
            
def generic_mapping():
    info = []
    for i in range(len(pref)):
        dose = pref.iloc[i].DoseFormRatio * pref.iloc[i].PerNUnits
        if round(dose) == dose:
            num_str = str(int(dose))
        else:
            num_str = str(dose)

        if num_str in pref.iloc[i].PharmaName:
            PharmaName = pref.iloc[i].PharmaName
            PharmaName = PharmaName.lstrip('z')
            PharmaName = PharmaName.replace(' ', '')
            PharmaName_comp = pref.iloc[i].PharmaName.lstrip('z').lstrip(' ')
            PharmaName_comp = PharmaName_comp.replace('  ', ' ')
            PharmaName_comp = PharmaName_comp.replace(' %', '%')
            PharmaName = PharmaName.lower()
            loc_low = PharmaName.find(num_str)
            PharmaName = PharmaName[:loc_low]
            info.append(tuple([i, PharmaName, dose, PharmaName_comp]))

    info = np.array(list(set(info)))
    pName = np.array([x[1] for x in info])
    pName_set = np.unique(pName)

    mapping = []
    for x in pName_set:
        idx = np.where(pName == x)[0]
        if len(idx) > 1:
            doses = [float(x[2]) for x in info[idx]]
            idx_sort = idx[np.argsort(doses)]
            doses = np.sort(doses)
            info_tmp = info[idx_sort]
            ilocs = [int(x[0]) for x in info_tmp]
            pref_map = pref.iloc[ilocs[0]]
            if pref_map.PharmaID in [1000541, 1000300]:
                tmp = pref.iloc[ilocs][['DoseUnitID', 'DoseFormRatio', 'PerNUnits', 'InFluidFormRatio', 'PharmaName', 'ArchTime']].sort_values('ArchTime')
            for i in range(len(ilocs)):
                pref_curr = pref.iloc[ilocs[i]]
                if pref_curr.DoseUnitID == pref_map.DoseUnitID:
                    ratio = pref_curr.DoseFormRatio / pref_map.DoseFormRatio
                elif pref_curr.QuantityID == pref_map.QuantityID:
                    ratio = ( pref_curr.DoseFormRatio * pref_curr.Coefficient ) / ( pref_map.DoseFormRatio * pref_map.Coefficient )

                mapping.append(tuple([pref_curr.PharmaID, pref_map.PharmaID, ratio, info_tmp[i][3]]))

    df_mapping = pd.DataFrame(np.array(mapping), columns=['PharmaID', 'Mapped_PharmaID', 'Ratio', 'PharmaName'])
    df_mapping.drop_duplicates(['PharmaID', 'Mapped_PharmaID', 'Ratio'], inplace=True)
    df_mapping.to_csv('PharmaUnitChange.csv', sep='\t', index=False, 
                      dtype={'PharmaID': np.int64, 'Mapped_PharmaID':np.int64, 'Ratio': np.float64, 'PharmaName': np.str_})

def unitref_standardization():
    '''
    Simplify the unitref table: merge unitIDs with the same meaning at different time
    '''
    uref = get_reference('unitref', 'original')

    # Manually fix some typos in the unitref table
    uref.set_value(uref[uref.Name=='Celsius'].index, 'Name', 'Celcius')
    uref.set_value(uref[uref.Name=='Tag'].index, 'Name', 'day')
    uref.set_value(uref[uref.Name=='zzzzz'].index, 'Name', 'Puder')
    uref.set_value(uref[uref.Abbreviation=='_mg'].index, 'Abbreviation', 'mg')

    # Force DoseUnitID=1003 to use the same reference unit as DoseUnitID=12
    # DoseUnitID = 1003 & 12 both mean milligram, but the coefficient is 1 and .001 respectively
    uref.set_value(uref[uref.UnitID==1003].index, 'Coefficient', 1e-3)
    uref.set_value(uref[uref.UnitID==1003].index, 'QuantityID', 1)

    # Remove the previous reference record if the new reference is only the duplicate of the old one
    uref.drop_duplicates(subset=uref.columns.difference(['Abbreviation', 'ArchTime']), 
                         keep='last', inplace=True)

    # QuantityID indicate the unit of measure, for instance, mass, length or time
    uref.sort_values(['QuantityID', 'UnitID'], inplace=True)
    uref.to_csv(join(outputpath, 'derived-unitref.csv'), sep='\t', index=False)

def collapse_ref_by_coefratio(df):
    df_no_duplicates = df.copy()
    df_no_duplicates.loc[:, 'CoefRatio'] = np.float64(df_no_duplicates.loc[:, 'CoefRatio'])
    df_no_duplicates.drop_duplicates(['PharmaID', 'CoefRatio', 'OriginalUnitID'],
                                     keep='last', inplace=True)
    cnts = df_no_duplicates.PharmaID.value_counts()
    idx_uchange = cnts[cnts>1].index
    if len(idx_uchange) == 0:
        return pd.DataFrame([], columns=df.columns)
    else:
        df_uchange = df[[(x in idx_uchange) for x in df.PharmaID]].copy()
        bool_keep = []
        for i in range(len(df_uchange)-1):
            row1 = df_uchange.iloc[i]
            row2 = df_uchange.iloc[i+1]
            if row1.PharmaID==row2.PharmaID and float(row1.CoefRatio)==float(row2.CoefRatio) and row1.OriginalUnitID==row2.OriginalUnitID:
                bool_keep.append(False)
            else:
                bool_keep.append(True)
        bool_keep.append(True)
        df_collapse = df_uchange[bool_keep]
        return df_collapse
    
        
def pharmaref_standardization():
    '''
    Standardize the unit of the same drugs in the pharmaref table
    '''
    uref = pd.read_csv(join(outputpath, 'derived-unitref.csv'), sep='\t')
    pref = pd.read_csv(join(datapath, '0_csv_exports', 'expot-pharmaref.csv'), sep='\t', encoding="cp1252")

    pID_DoseUnitChange = []
    pID_FormUnitChange = []

    # DoseUnitID == 1003 & 12 mean the same unit: milligramm 
    pref.set_value(pref[pref.DoseUnitID==1003].index, 'DoseUnitID', 12)
    pref.set_value(pref[pref.FormUnitID==1003].index, 'FormUnitID', 12)
    for pID in pref.PharmaID.unique():
        pref_tmp = pref[pref.PharmaID == pID].copy().sort_values('ArchTime')
        if len(pref_tmp.DoseUnitID.unique()) == 1 and len(pref_tmp.FormUnitID.unique()) == 1:
            continue

        # Use the latest unit as the reference unit and standardize units at other time to the reference unit
        DoseUnitID_ref = pref_tmp.iloc[-1]['DoseUnitID']
        DoseQuantityID_ref = uref[uref.UnitID == DoseUnitID_ref].iloc[0]['QuantityID']
        DoseCoef_ref = uref[uref.UnitID == DoseUnitID_ref].iloc[0]['Coefficient']
        DoseUnitAbbr_ref = uref[uref.UnitID == DoseUnitID_ref].iloc[0]['Abbreviation']

        FormUnitID_ref = pref_tmp.iloc[-1]['FormUnitID']            
        FormQuantityID_ref = uref[uref.UnitID == FormUnitID_ref].iloc[0]['QuantityID']
        FormCoef_ref = uref[uref.UnitID == FormUnitID_ref].iloc[0]['Coefficient']
        FormUnitAbbr_ref = uref[uref.UnitID == FormUnitID_ref].iloc[0]['Abbreviation']

        for idx in pref[pref.PharmaID == pID].index:
            DoseUnitID_tmp = pref.loc[idx]['DoseUnitID']
            FormUnitID_tmp = pref.loc[idx]['FormUnitID']
            ArchTime_tmp = pref.loc[idx]['ArchTime']
            PharmaName_tmp = pref.loc[idx]['PharmaName']
            DoseFormRatio_new = pref.loc[idx]['DoseFormRatio']

            DoseCoef_tmp = uref[uref.UnitID == DoseUnitID_tmp].iloc[0]['Coefficient']
            DoseUnitAbbr_tmp = uref[uref.UnitID == DoseUnitID_tmp].iloc[0]['Abbreviation']
            DoseQuantityID_tmp = uref[uref.UnitID == DoseUnitID_tmp].iloc[0]['QuantityID']
            CoefRatio = 1
            if DoseUnitID_ref != 0 and DoseUnitID_tmp != 0:
                CoefRatio = (DoseCoef_tmp / DoseCoef_ref)
            if DoseQuantityID_ref == 1 and DoseQuantityID_tmp == 6: # from ml -> mg
                CoefRatio *= 1000.0
            elif DoseQuantityID_ref == 6 and DoseQuantityID_tmp == 1: # from mg -> ml
                CoefRatio *= 1e-3
            elif DoseQuantityID_ref == 1 and DoseQuantityID_tmp == 7: # from mmol -> mg
                CoefRatio *= 18.0
            elif DoseQuantityID_ref == 7 and DoseQuantityID_tmp == 6: # from ml -> mmol
                CoefRatio *= 1000.0 / 18.0
            DoseFormRatio_new *= CoefRatio
            pref.set_value(idx, 'DoseUnitID', DoseUnitID_ref)
            pID_DoseUnitChange.append([pID, CoefRatio, DoseUnitID_tmp, DoseUnitID_ref, DoseUnitAbbr_tmp, 
                                       DoseUnitAbbr_ref, ArchTime_tmp, PharmaName_tmp])

            FormCoef_tmp = uref[uref.UnitID == FormUnitID_tmp].iloc[0]['Coefficient']
            FormUnitAbbr_tmp = uref[uref.UnitID == FormUnitID_tmp].iloc[0]['Abbreviation']
            FormQuantityID_tmp = uref[uref.UnitID == FormUnitID_tmp].iloc[0]['QuantityID']
            CoefRatio = 1
            if FormUnitID_ref != 0 and FormUnitID_tmp != 0:
                CoefRatio = FormCoef_ref / FormCoef_tmp
            if FormQuantityID_ref == 1 and FormQuantityID_tmp == 6: # from ml -> mg
                CoefRatio *= 1000.0
            elif FormQuantityID_ref == 6 and FormQuantityID_tmp == 1: # from mg -> ml
                CoefRatio *= 1e-3
            DoseFormRatio_new /= CoefRatio
            pref.set_value(idx, 'FormUnitID', FormUnitID_ref)
            pID_FormUnitChange.append([pID, CoefRatio, FormUnitID_tmp, FormUnitID_ref, FormUnitAbbr_tmp, 
                                       FormUnitAbbr_ref, ArchTime_tmp, PharmaName_tmp])

            pref.set_value(idx, 'DoseFormRatio', DoseFormRatio_new)
 
    pref.to_csv(join(outputpath, 'standardized-pharmaref.csv'), sep='\t', index=False) # Write the standardized pharmaref for future use
    
    # Write the pharmaID with unit change to files for reference
    columns = ['PharmaID', 'CoefRatio', 'OriginalUnitID', 'StandardUnitID', 'OriginalUnitAbbr', 
               'StandardUnitAbbr', 'ArchTime', 'PharmaName']

    pID_DoseUnitChange = pd.DataFrame(np.array(pID_DoseUnitChange), columns=columns).sort_values(['PharmaID', 'ArchTime'])
    pID_DoseUnitChange.to_csv(join(outputpath, 'DoseUnitChange.csv'), sep='\t', index=False)

    pID_FormUnitChange = pd.DataFrame(np.array(pID_FormUnitChange), columns=columns).sort_values(['PharmaID', 'ArchTime'])
    pID_FormUnitChange.to_csv(join(outputpath, 'FormUnitChange.csv'), sep='\t', index=False)

def standard_pharmaref_validation():
    '''
    Validate if the standardization of the pharmref is done correctly
    '''
    pID_DoseUnitChange = get_reference('DoseUnitChange')
    pID_FormUnitChange = get_reference('FormUnitChange')
    uref = get_reference('unitref')
    uref.set_index('UnitID', inplace=True)

    pref_n = get_referenc('pharmaref')
    pref_o = get_referenc('pharmaref', 'original')
    id_voi = voi_id_name_mapping('pharmarec').index.unique()
    id_voi_uchange = (set(pID_FormUnitChange.PharmaID.unique()) | set(pID_DoseUnitChange.PharmaID.unique())) & set(id_voi)
    local_outputpath = join(outputpath, 'unit_change')
    if not exists(local_outputpath):
        mkdir(local_outputpath)
    func_get_abbr = lambda x: uref.loc[x]['Abbreviation']
    for pID in id_voi_uchange:
        cols2read = ['PharmaName', 'DoseFormRatio', 'DoseUnitID', 'FormUnitID', 'PerNUnits', 'ArchTime', 'PharmaID']
        pref_o_tmp = pref_o[pref_o.PharmaID==pID][cols2read].copy().sort_values('ArchTime')
        pref_o_tmp['Dose'] = pref_o_tmp.DoseFormRatio * pref_o_tmp.PerNUnits
        pref_o_tmp['Type'] = 'Original'
        pref_o_tmp['DoseUnitAbbr'] = pref_o_tmp.DoseUnitID.apply(func_get_abbr)
        pref_o_tmp['FormUnitAbbr'] = pref_o_tmp.FormUnitID.apply(func_get_abbr)

        pref_n_tmp = pref_n[pref_n.PharmaID==pID][cols2read].copy().sort_values('ArchTime')
        pref_n_tmp['Dose'] = pref_n_tmp.DoseFormRatio * pref_n_tmp.PerNUnits
        pref_n_tmp['Type'] = 'Standardized'
        pref_n_tmp['DoseUnitAbbr'] = pref_n_tmp.DoseUnitID.apply(func_get_abbr)
        pref_n_tmp['FormUnitAbbr'] = pref_n_tmp.FormUnitID.apply(func_get_abbr)

        pref_tmp_aggr = pd.concat([pref_o_tmp, pref_n_tmp], ignore_index=True)
        columns = ['PharmaID', 'Type', 'ArchTime', 'Dose', 'DoseFormRatio', 'PerNUnits', 'DoseUnitID', 
                   'DoseUnitAbbr', 'FormUnitID', 'FormUnitAbbr', 'PharmaName']
        pref_tmp_aggr.to_csv(join(local_outputpath, 'p%d.csv'%pID), sep='\t', index=False, columns=columns)

def print_unit_change():
    for filename in ['DoseUnitChange.csv', 'FormUnitChange.csv']:
        print(filename)
        df = pd.read_csv(join(outputpath, filename), sep='\t')
        print('| PharmaID | Unit Changes |')
        print('|---|---|')
        for pid in df.PharmaID.unique():
            df_tmp = df[df.PharmaID == pid]
            print('| %d | %s |'%(pid, ' -> '.join([df_tmp.iloc[i]['OriginalUnitAbbr'] for i in range(len(df_tmp))])))


def pharmarec_normalization_test(used_after_archtime=True):
    pref = get_reference('pharmaref', 'original')
    ref = get_reference('DoseUnitChange')
    form_ref = get_reference('FormUnitChange')
    columns = ['PatientID', 'DateTime', 'GivenDose']
    f = open('pharmarec_normalization_res.txt', 'w')
    pharmaID_voi = voi_id_name_mapping('pharmarec', True, True).index.unique()

    # ref_voi = set(pharmaID_voi) & set(ref.PharmaID)
    # ref_voi = list(ref_voi)
    # for pharmaID in ref_voi:
    #     print('-------PharmaID=%d------'%( pharmaID ))
    #     f.write('-------PharmaID=%d------\n'%( pharmaID ))
    #     pharma = pd.read_hdf(join(datapath, '1_hdf5_consent', 'pharmarec.h5'), where='PharmaID=%d'%pharmaID,
    #                          columns=columns, mode='r')
    #     ref_tmp = ref[ref.PharmaID == pharmaID]
    #     ref_tmp_len = len(ref_tmp)
    #     pharma_splits = split_by_archtime(pharma, ref_tmp)
    #     ref_unit = ref_tmp.iloc[-1]['StandardUnitAbbr']
    #     # for i, split in enumerate(pharma_splits):
    #     #     if np.sum(np.logical_not(np.isnan(split.GivenDose))) != 0:
    #     #         pharma.loc[split.index, 'GivenDose'] *= ref_tmp.iloc[max(i-1,0)]['CoefRatio']

    #     for i, split in enumerate(pharma_splits):
    #         givendose = np.array(split['GivenDose'].dropna().tolist())
    #         givendose = givendose[givendose!=0]
    #         split_len = len(givendose)
    #         if split_len != 0:
    #             split_median = np.median(givendose)
    #             if used_after_archtime:
    #                 split_unit = ref_tmp.iloc[max(i-1,0)]['OriginalUnitAbbr']
    #                 split_coef = ref_tmp.iloc[max(i-1,0)]['CoefRatio']
    #                 pharmaName = ref_tmp.iloc[max(i-1,0)]['PharmaName']
    #             else:
    #                 split_unit = ref_tmp.iloc[min(i, ref_tmp_len-1)]['OriginalUnitAbbr']
    #                 split_coef = ref_tmp.iloc[min(i, ref_tmp_len-1)]['CoefRatio']
    #                 pharmaName = ref_tmp.iloc[min(i, ref_tmp_len-1)]['PharmaName']
    #             split_median_original = split_median
    #             split_median *= split_coef
    #             print('%s, # records: %d, original median: %g, normalized median: %g, unit: %s -> %s'%(pharmaName, split_len, split_median_original, split_median, split_unit, ref_unit))
    #             f.write('%s, # records: %d, original median: %g, normalized median: %g, unit: %s -> %s\n'%(pharmaName, split_len, split_median_original, split_median, split_unit, ref_unit))

    form_ref_voi = set(pharmaID_voi) & set(form_ref.PharmaID)
    form_ref_voi = list(form_ref_voi)
    for pharmaID in form_ref_voi:
        print('-------PharmaID=%d------'%( pharmaID ))
        f.write('-------PharmaID=%d------\n'%( pharmaID ))
        pharma = pd.read_hdf(join(datapath, '1_hdf5_consent', 'pharmarec.h5'), where='PharmaID=%d'%pharmaID,
                             columns=columns, mode='r')
        form_ref_tmp = form_ref[form_ref.PharmaID == pharmaID]
        form_ref_tmp_len = len(form_ref_tmp)
        pharma_splits = split_by_archtime(pharma, form_ref_tmp)
        form_ref_unit = form_ref_tmp.iloc[-1]['StandardUnitAbbr']
        # for i, split in enumerate(pharma_splits):
        #     if np.sum(np.logical_not(np.isnan(split.GivenDose))) != 0:
        #         pharma.loc[split.index, 'GivenDose'] *= form_ref_tmp.iloc[max(i-1,0)]['CoefRatio']

        for i, split in enumerate(pharma_splits):
            givendose = np.array(split['GivenDose'].dropna().tolist())
            givendose = givendose[givendose!=0]
            split_len = len(givendose)
            if split_len != 0:
                split_median = np.median(givendose)
                if used_after_archtime:
                    split_unit = form_ref_tmp.iloc[max(i-1,0)]['OriginalUnitAbbr']
                    split_coef = form_ref_tmp.iloc[max(i-1,0)]['CoefRatio']
                    pharmaName = form_ref_tmp.iloc[max(i-1,0)]['PharmaName']
                else:
                    split_unit = form_ref_tmp.iloc[min(i, form_ref_tmp_len-1)]['OriginalUnitAbbr']
                    split_coef = form_ref_tmp.iloc[min(i, form_ref_tmp_len-1)]['CoefRatio']
                    pharmaName = form_ref_tmp.iloc[min(i, form_ref_tmp_len-1)]['PharmaName']
                split_median_original = split_median
                split_median *= split_coef
                print('%s, # records: %d, original median: %g, normalized median: %g, unit: %s -> %s'%(pharmaName, split_len, split_median_original, split_median, split_unit, form_ref_unit))
                f.write('%s, # records: %d, original median: %g, normalized median: %g, unit: %s -> %s\n'%(pharmaName, split_len, split_median_original, split_median, split_unit, form_ref_unit))
    f.close()

def split_by_archtime(pharmarec, ref):
    for col in pharmarec.columns:
        if 'time' in col.lower():
            col_datetime = col

    pharma_splits = [ pharmarec[pharmarec[col_datetime]<ref.iloc[0]['ArchTime']] ]
    for i in range(len(ref)-1):
        within_interval = np.logical_and(pharmarec[col_datetime]>=ref.iloc[i]['ArchTime'], 
                                              pharmarec[col_datetime]<ref.iloc[i+1]['ArchTime'])
        pharma_splits.append( pharmarec[within_interval] )
    pharma_splits.append( pharmarec[pharmarec[col_datetime]>=ref.iloc[-1]['ArchTime']] )
    return pharma_splits

if __name__=='__main__':

    pharmarec_normalization_test(True)
