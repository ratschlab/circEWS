#!/usr/bin/env ipython
# paths
root_dir = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'

clean_dir = root_dir + '1a_hdf5_clean/'
pivoted_dir = root_dir + '2_pivoted/'
merged_dir = root_dir + '3_merged/'
endpoints_dir = root_dir + '3a_endpoints/'
# anachronous - contains version
imputed_dir = root_dir + '5_imputed/imputed_180221/'
predictions_dir = root_dir + '8_predictions/'

observrec = root_dir + '2_pivoted/prepro_observrec.h5'
misc_dir = root_dir + 'misc_derived/stephanie/'

chunks_file = root_dir + 'misc_derived/id_lists/v6b/patients_in_clean_chunking_50.csv'

id2string = root_dir + 'misc_derived/visualisation/id2string_v6.npy'
mid2string = root_dir + 'misc_derived/visualisation/mid2string_v6.npy'
