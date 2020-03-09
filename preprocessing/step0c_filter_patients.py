#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../utils')
import preproc_utils
import pickle


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-version')
args = parser.parse_args()
version = args.version


# patient with consents
pids = set(preproc_utils.get_consent_patient_ids())
print('# Consenting patients:', len(pids))


# patient with variables of interest
idlist_path = os.path.join(preproc_utils.datapath, 'misc_derived', 'id_lists', version)
pids_with_voi_list = []
for tbl_name in ['monvals', 'dervals', 'observrec', 'pharmarec', 'labres']:
    tmp = set(pd.read_csv(os.path.join(idlist_path, 'patients_with_voi_in_%s.csv'%tbl_name)).PatientID)
    print(tbl_name, len(tmp))
    pids_with_voi_list.append(tmp)
    
pids_voi = pids_with_voi_list[0]
for i in np.arange(1, len(pids_with_voi_list)):
    pids_voi |= pids_with_voi_list[i]

pids = pids & pids_voi
print('# Consenting patients with variables of interest:', len(pids))


# patient who are admitted after 2008
generaldata = pd.read_hdf(os.path.join(preproc_utils.datapath, '1_hdf5_consent', '180704', 'generaldata.h5'))
generaldata['AdmissionYear'] = generaldata.AdmissionTime.apply(lambda x: x.year)
generaldata['Age'] = generaldata.AdmissionYear - generaldata.birthYear
generaldata = generaldata[generaldata.AdmissionYear>=2008]
pids_after_2008 = set(generaldata.PatientID)
pids = pids & pids_after_2008
print('# Consenting patients with variables of interest, admitted after 2008:', len(pids))


# patient whose age is between 16 and 100
generaldata = generaldata[np.logical_and(generaldata.Age >= 16, generaldata.Age <= 100)]
pids_age_appropriate = set(generaldata.PatientID)
pids = pids & pids_age_appropriate
print('# Consenting patients with variables of interest, admitted after 2008, whose age is between 16 and 100:', len(pids))


# patient who are not on ECMO
ecmo_path = '/cluster/work/grlab/clinical/Inselspital/DataReleases/11-28-2017/0_db_export_hdf/p_patcareseqs'
df = pd.concat([pd.read_hdf(os.path.join(ecmo_path, f)) for f in os.listdir(ecmo_path) if '.h5' in f], axis=0)
df['name'] = df.name.astype(str)
df['placename'] = df.placename.astype(str)
contain_assist = df.name.apply(lambda x: 'assist' in x.lower())
contain_ecmo = np.logical_or(df.name.apply(lambda x: 'ecmo' in x.lower() or 'assist' in x.lower()), df.placename.apply(lambda x: 'ecmo' in x.lower() or 'assist' in x.lower()))
# contain_ecmo = np.logical_or(df.name.apply(lambda x: 'ecmo' in x.lower()), df.placename.apply(lambda x: 'ecmo' in x.lower()))
df = df[np.logical_or(contain_assist, contain_ecmo)]
pids_on_ecmo = set(df.patientid)
pids -= pids_on_ecmo
print('# Consenting patients with variables of interest, admitted after 2008, whose age is between 16 and 100, not on ecmo:', len(pids))


pids = np.reshape(list(pids), (-1,1))
df_pid = pd.DataFrame(pids, columns=['PatientID'])
df_pid.to_csv(os.path.join(idlist_path, 'patients_in_datetime_fixed.csv'), index=False)


num_chunk = 50
if not os.path.exists(os.path.join(idlist_path, 'patients_in_clean_chunking_%d.csv'%num_chunk)):

    pids = np.sort(pids)
    chunksize = int(np.ceil( len(pids)/num_chunk ))
    pids_chunk_index = []
    for i in range(num_chunk):
        pids_chunk = pids[i*chunksize:min((i+1)*chunksize,len(pids))]
        pids_chunk_index.append(np.hstack((pids_chunk, np.ones((len(pids_chunk),1))*i)))
    pids_chunk_index = np.vstack(tuple(pids_chunk_index))
    df_chunk_index = pd.DataFrame(pids_chunk_index, columns=['PatientID', 'ChunkfileIndex'], dtype=np.int64)
    df_chunk_index.to_csv(os.path.join(idlist_path, 'patients_in_datetime_fixed_chunking_%d.csv'%num_chunk), index=False)
else:
    print(os.path.join(idlist_path, 'patients_in_datetime_fixed_chunking_%d.pkl'%num_chunk), 'alreadt exists.')
    print('Please delete to rewrite.')




