#!/usr/bin/env python
import sys
sys.path.append('../../utils')
import numpy as np
import pandas as pd
import csv
import re

from os.path import join, exists
from os import mkdir, remove, listdir

from preproc_utils import voi_id_name_mapping, time_difference
from unit_normalization import get_reference

data_path = '/cluster/work/grlab/Inselspital/DataReleases/01-19-2017/InselSpital/'
output_path = './'

if not exists(output_path):
    os.mkdir(output_path)

def get_doseformratio(df_ref, PharmaID, timestamp):
    '''
    Get the DoseFormRatio value of a given PharmaID for computing the GivenDose value
    '''
    df_tmp = df_ref[df_ref.PharmaID==PharmaID]
    idx = df_tmp[df_tmp.ArchTime>timestamp].index
    if len(idx)==0:
        return df_tmp.iloc[-1]['DoseFormRatio']
    else:
        return df_tmp.loc[idx[0], 'DoseFormRatio']

def compute_givendose_err(verbose=False):
    pref = get_reference('pharmaref')
    uref = get_reference('unitref')
    uchange = get_reference('DoseUnitChange')

    pharmarec_path = join(data_path, '1_hdf5_consent', 'pharmarec.h5')

    local_output_path = join(output_path, 'compdose_err')
    if not exists(local_output_path):
        mkdir(local_output_path)

    id_voi = voi_id_name_mapping('pharmarec').index.tolist()
    
    for pharmaID in id_voi[:10]:
        print('--------PharmaID = %d-------'%pharmaID)
        pharma_all = pd.read_hdf(pharmarec_path, 'raw_import', where='PharmaID = %d'%pharmaID, mode='r')
        pharma_all.sort_values(['PatientID', 'InfusionID', 'DateTime'], inplace=True)
        pharma_all['ComputedDose'] = np.nan
        uchange_tmp = uchange[uchange.PharmaID==pharmaID]

        with open(join(local_output_path, 'pharma_%d.csv'%pharmaID), 'w') as f:
            writer = csv.writer(f, delimiter='\t') 
            writer.writerow(['PatientID', 'nCompDose', 'leq_1p', '1eq_5p', 'leq_10p'])
            
            for patientID in pharma_all.PatientID.unique():
                pharma = pharma_all[pharma_all.PatientID == patientID]

                pharma = pharma[np.logical_or(np.logical_or(pharma.Status==524, pharma.Status==520), 
                                              np.logical_or(pharma.Status==776, pharma.Status==8))]
                idx_nostart = np.where(pharma.Status!=524)[0]
                
                if len(idx_nostart) == 0:
                    continue
                
                t_interval = time_difference(pharma.iloc[idx_nostart-1]['DateTime'], 
                                             pharma.iloc[idx_nostart]['DateTime'])
                DoseFormRatio = np.array([get_doseformratio(pref, pharmaID, pharma.iloc[x]['DateTime']) for x in idx_nostart])
                
                pharma.loc[pharma.index[idx_nostart], 'ComputedDose'] = ( np.array(pharma.iloc[idx_nostart-1]['Rate'].tolist())
                                                                          * DoseFormRatio * t_interval )
                
                if len(uchange_tmp) > 0:
                    idx_tmp = pharma[pharma.DateTime<uchange_tmp.iloc[0]['ArchTime']].index
                    pharma.loc[idx_tmp, 'GivenDose'] = pharma.loc[idx_tmp, 'GivenDose'] * uchange_tmp.iloc[0]['CoefRatio']
                    for i in range(len(uchange_tmp)-1):
                        idx_tmp = pharma[( pharma.DateTime>=uchange_tmp.iloc[i]['ArchTime'])
                                        & (pharma.DateTime<uchange_tmp.iloc[i+1]['ArchTime'] )].index
                        
                        pharma.loc[idx_tmp, 'GivenDose'] = pharma.loc[idx_tmp, 'GivenDose'] * uchange_tmp.iloc[i+1]['CoefRatio']

                DoseDiff = np.abs(pharma.iloc[idx_nostart]['ComputedDose'] - pharma.iloc[idx_nostart]['GivenDose'])
                Denominator = np.array( pharma.iloc[idx_nostart]['GivenDose'].tolist() )
                Denominator[Denominator==0] = 1
                DoseDiff /= Denominator
                
                nCompDose = len(DoseDiff)
                row = [patientID, nCompDose, np.sum(DoseDiff<=1e-2), np.sum(DoseDiff<=5e-2), np.sum(DoseDiff<=1e-1)]
                writer.writerow(row)
                if verbose:
                    print(row)

def summarize_givendose_err():
    pref = pd.read_csv(join(output_path, 'standardized-pharmaref.csv'), sep='\t', parse_dates=['ArchTime'])
    pref['PharmaName'] = pref['PharmaName'].apply(lambda x: x.encode('cp1252'))
    pref.sort_values(['PharmaID', 'ArchTime'], inplace=True)
    template = 'pharma_(?P<pharmaID>\w+)'
    mat = []
    filename_list = listdir(join(output_path, 'compdose_err'))
    for filename in filename_list:
        res = re.match(template, filename)
        if res:
            pharmaID = int(res.group('pharmaID'))
            try:
                df = pd.read_csv(join(output_path, 'compdose_err', filename), sep='\t')
                if len(df)==0:
                    # remove(filename)
                    continue
                print('-----pharmaID=%d-------'%pharmaID)
                print('# Patients with err_rate > 10%%: %d (out of %d)'%(np.sum(np.sum(df.iloc[:,2:], axis=1)==0), len(df)))
                pharmaName = pref[pref.PharmaID==pharmaID].iloc[-1].PharmaName
                nCompDose = np.sum(df.iloc[:,1], axis=0)
                mat.append([pharmaID, pharmaName] + (np.sum(df.iloc[:,2:].as_matrix(), axis=0)/nCompDose).tolist() + [nCompDose])
                print('Average percentage of measurements: %s'%(mat[-1]))
            except pd.io.common.EmptyDataError:
                continue
    df = pd.DataFrame(np.array(mat), columns=['PharmaID', 'PharmaName', 'less_equal_1_percent','less_equal_5_percent','less_equal_10_percent', 'num_computed_given_dose'])
    df.sort_values('less_equal_1_percent', inplace=True)
    df.to_csv(join(output_path, 'compdose_err_freq.csv'), sep=',', index=False)


if __name__=='__main__':
    compute_givendose_err(True)
    # summarize_givendose_err()

