import os
import pandas as pd
import numpy as np
import pdb
import math
import pickle
import glob
import sys

#enviroment variables
data_path = "/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/"
output_dir = "derived_mafacz/bern_apacheII"
shap_data_path = os.path.join(data_path, "8_predictions/181108/reduced/held_out/WorseStateFromZero_0.0_8.0_shap_top500_features_lightgbm_full")
value_data_path = os.path.join(data_path, "7_ml_input/180918/reduced/held_out/AllLabels_0.0_8.0/X")
data_file_shaplets = os.path.join(data_path, "9_s3m/feature_matrices/Shapelet_features_held_out.h5")
output_data_file = os.path.join(data_path, output_dir, "feature_shap_values.h5")
data_file_static = os.path.join(data_path, "1a_hdf5_clean/v6b/static.h5")

PIDS_OF_TESTSET=os.path.join(data_path, "misc_derived/temporal_split_180918.tsv")
BERN_PID_MAP_PATH= os.path.join(data_path, "misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle")

#load metadata csv file and patient data mappingd
pkl_file = open(BERN_PID_MAP_PATH, 'rb')
pkl_obj = pickle.load(pkl_file)
pidChunkMapping = pkl_obj['pid_to_chunk']
chunk_to_pids = pkl_obj['chunk_to_pids']

#load patientid file
df_ids = pd.read_csv(PIDS_OF_TESTSET, '\t')
uniquePatientIDs = df_ids[df_ids['held_out'] == 'test'].pid.values

#Read full static file to memory
df_data_static = pd.read_hdf(data_file_static)
final_dataframe = pd.DataFrame()
loaded_patient_chunk = -1
key_list = []
#Loop trough all patients and add them to preprocessed table
i = 0
for patientID in uniquePatientIDs:
	i=i+1

	#read in patient data
	currentPatientChunk = pidChunkMapping[patientID]
	shap_chunk_store = os.path.join(shap_data_path,"batch_{}.h5".format(currentPatientChunk))
	value_chunk_store = os.path.join(value_data_path,"batch_{}.h5".format(currentPatientChunk))
	print("Patient {0}, beeing {1} from chunk {2}".format(patientID, i, currentPatientChunk))

	#load available keys from store if not yet available
	if currentPatientChunk != loaded_patient_chunk:
		loaded_patient_chunk = currentPatientChunk
		with pd.HDFStore(shap_chunk_store, mode='r') as hdf:
			key_list = hdf.keys()
	
	key = "p" + str(patientID)
	if ("/" + key) in key_list:
		df_shap = pd.read_hdf(shap_chunk_store, key)
		df_value = pd.read_hdf(value_chunk_store, str(patientID))
		df_shaplets = pd.read_hdf(data_file_shaplets, str(patientID))

		#generate column list
		pdb.set_trace()
		columns = ['PatientID']
		columns_shaplets = ['PatientID', 'RelDatetime']
		for column in df_shap.columns:
			if column[:8] == "RawShap_" and 'static' not in column and 'dist-set' not in column:
				columns.append(column[8:])
			if column[:8] == "RawShap_" and 'static' not in column and 'dist-set' in column:
				columns_shaplets.append(column[8:])

		pdb.set_trace()
		df_value = df_value.loc[:,columns] 
		df_shaplets = df_shaplets.loc[:,columns_shaplets]
		df_data = pd.merge(df_shap, df_value, on=['RelDatetime', 'PatientID'])
		df_data = pd.merge(df_data, df_shaplets, on=['RelDatetime', 'PatientID'])
		df_data['static_Age'] = df_data_static.loc[df_data_static.PatientID == patientID,'Age'].values[0]
		df_data['static_Height'] = df_data_static.loc[df_data_static.PatientID == patientID,'Height'].values[0]
		df_data.reset_index()
		df_data = df_data.astype({"PatientID": int})
		df_data.to_hdf(output_data_file, 'pivoted', append=True, complevel=5, complib='blosc:lz4', data_columns=['PatientID'], format='table')
	