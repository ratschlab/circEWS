#!/usr/bin/env ipython
# Refer to the excel

import pandas as pd
from numpy import save
import pdb

v5 = False

if v5:
    excel_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/doc/Parameters ICUscoring v_5_final_minimalset.xlsx'
    derived_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/stephanie/pharma_acting_period_v5.npy'
else:
    excel_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/ref_excel/varref_excel_v6.tsv'
    derived_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/stephanie/pharma_acting_period_v6.npy'
    meta_derived_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/stephanie/pharma_acting_period_v6_meta.npy'

# the period is in minutes
enum_ref = {'very short': 5, 'short': 1*60, '4h': 4*60, '6h': 6*60, '12h': 12*60, '24h': 24*60, '3d': 3*24*60}

if v5:
    # this just gets the first sheet
    excel = pd.read_excel(excel_path)      # v5
else:
    excel = pd.read_csv(excel_path, sep='\t', encoding='cp1252')
drugs = excel.loc[excel['Type'] == 'Pharma', :]
drugs_meta = excel.loc[excel['Type'] == 'Pharma', :]

if v5:
    drugs['ID'] = list(map(lambda x: 'p' + str(x), drugs['ID'].values))
    drugs['Pharma acting period'] = list(map(lambda x: enum_ref[x], drugs['Pharma acting period'].values))
    drugs.set_index('ID', inplace=True)
    pID_to_period = drugs['Pharma acting period'].to_dict()
else:
    drugs['VariableID'] = list(map(lambda x: 'p' + str(int(x)), drugs['VariableID'].values))
    drugs_meta['MetaVariableID'] = list(map(lambda x: 'pm' + str(int(x)), drugs['MetaVariableID'].values))
    drugs['PharmaActingPeriod'] = list(map(lambda x: enum_ref[x], drugs['PharmaActingPeriod'].values))
    drugs_meta['PharmaActingPeriod'] = list(map(lambda x: enum_ref[x], drugs_meta['PharmaActingPeriod'].values))
    drugs.set_index('VariableID', inplace=True)
    drugs_meta.set_index('MetaVariableID', inplace=True)
    pID_to_period = drugs['PharmaActingPeriod'].to_dict()
    meta_pID_to_period = drugs_meta['PharmaActingPeriod'].to_dict()

#save(derived_path, pID_to_period)
save(meta_derived_path, meta_pID_to_period)
