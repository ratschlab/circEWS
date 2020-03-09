# --- internal paths --- #
mimic_root_dir = '/cluster/work/grlab/clinical/mimic/MIMIC-III/cdb_1.4/'
root_dir = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'

# --- all about mimic --- #
source_data = mimic_root_dir + 'source_data/'
derived = mimic_root_dir + 'derived_stephanie/'
chartevents_path = source_data + 'CHARTEVENTS.csv'
labevents_path = source_data + 'LABEVENTS.csv'
outputevents_path = source_data + 'OUTPUTEVENTS.csv'
inputevents_cv_path = source_data + 'INPUTEVENTS_CV.csv'
inputevents_mv_path = source_data + 'INPUTEVENTS_MV.csv'
datetimeevents_path = source_data + 'DATETIMEEVENTS.csv'
procedureevents_mv_path = source_data + 'PROCEDUREEVENTS_MV.csv'
admissions_path = source_data + 'ADMISSIONS.csv'
patients_path = source_data + 'PATIENTS.csv'
icustays_path = source_data + 'ICUSTAYS.csv'
services_path = source_data + 'SERVICES.csv'
csv_folder = derived + 'derived_csvs/'

# --- all about our data on leomed --- #
validation_dir = root_dir + 'external_validation/'
misc_dir = root_dir + 'misc_derived/stephanie/'
vis_dir = validation_dir + 'vis/'

D_ITEMS_path = validation_dir + 'ref_lists/D_ITEMS.csv'
D_LABITEMS_path = validation_dir + 'ref_lists/D_LABITEMS.csv'
GDOC_path = validation_dir + 'ref_lists/mimic_vars.csv'

chunks_file = validation_dir + 'chunks.csv'

csvs_dir = validation_dir + 'csvs/'
hdf5_dir = validation_dir + 'hdf5/'
# mimic is always reduced
merged_dir = validation_dir + 'merged/'
endpoints_dir = validation_dir + 'endpoints/'

predictions_dir = validation_dir + 'predictions/'

id2string = root_dir + 'misc_derived/visualisation/id2string_v6.npy'
mid2string = root_dir + 'misc_derived/visualisation/mid2string_v6.npy'
