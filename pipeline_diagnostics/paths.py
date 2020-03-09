#!/usr/bin/env ipython
# )paths for pipeline diagnostics

v6a = False
v6b = True

base_dir = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/'

splits_file = base_dir + '/misc_derived/temporal_split_180918.tsv'
chunks_file = base_dir + '/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.csv'
misc_dir = base_dir + '/misc_derived/stephanie/'
csvs_0 = base_dir + '0_csv_exports/'
hdf_consent_dir = base_dir + '1_hdf5_consent/'
derived_dir = base_dir + 'misc_derived/pipeline_diagnostics/'

apache_info_path = base_dir + 'derived_mafacz/bern_apacheII/appache_group.h5'

if v6a:
    hdf_clean_dir = base_dir + '1a_hdf5_clean/180704/'
    pivoted_dir = base_dir + '2_pivoted/180704/'
    merged_dir = base_dir + '3_merged/180704/'
    merged_dir_reduced = merged_dir + 'reduced/'
    endpoints_dir = base_dir + '3a_endpoints/180704/'
    endpoints_dir_reduced = endpoints_dir + 'reduced/'
    imputed_dir = base_dir + '5_imputed/imputed_180830/reduced/'
    labels_dir = base_dir + '6_labels/targets_180830/reduced/'
    mlinput_dir = base_dir + '7_ml_input/180830/reduced/'
    predictions_dir = base_dir + '/8_predictions/180830/reduced/'

    id2string = base_dir + '/misc_derived/visualisation/id2string_v6.npy'
    mid2string = base_dir + '/misc_derived/visualisation/mid2string_v6.npy'
elif v6b:
    hdf_clean_dir = base_dir + '1a_hdf5_clean/v6b/'
    pivoted_dir = base_dir + '2_pivoted/v6b/'
    merged_dir = base_dir + '3_merged/v6b/'
    merged_dir_reduced = merged_dir + 'reduced/'
    endpoints_dir = base_dir + '3a_endpoints/v6b/'
    endpoints_dir_reduced = endpoints_dir + 'reduced/'
    imputed_dir = base_dir + '5_imputed/imputed_180918/reduced/'
    labels_dir = base_dir + '6_labels/targets_180918/reduced/'
    mlinput_dir = base_dir + '/7_ml_input/180918/reduced/'
    print('WARNING: PREDICTIONS DIRECTORY IS DIFFERENT')
    predictions_dir = base_dir + '/8_predictions/181108/reduced/'

    id2string = base_dir +'/misc_derived/visualisation/id2string_v6.npy'
    mid2string = base_dir + '/misc_derived/visualisation/mid2string_v6.npy'
elif v6:
    hdf_clean_dir = base_dir + '1a_hdf5_clean/180704/'
    pivoted_dir = base_dir + '2_pivoted/180704/'
    merged_dir = base_dir + '3_merged/180704/'
    merged_dir_reduced = merged_dir + 'reduced/'
    endpoints_dir = base_dir + '3a_endpoints/180704/'
    endpoints_dir_reduced = endpoints_dir + 'reduced/'
    imputed_dir = base_dir + '5_imputed/imputed_180731/reduced/'
    labels_dir = base_dir + '6_labels/targets_180731/reduced/'
    mlinput_dir = base_dir + '7_ml_input/180731/reduced/'
    predictions_dir = base_dir + '/8_predictions/180731/reduced/'

    id2string = base_dir + '/misc_derived/visualisation/id2string_v6.npy'
    mid2string = base_dir + '/misc_derived/visualisation/mid2string_v6.npy'
else:
    # this is kept for legacy reasons
    hdf_clean_1a = base_dir + '1a_hdf5_clean/180214/'
    pivoted_2 = base_dir + '2_pivoted/180214/'
    merged_3 = base_dir + '3_merged/180214/'
    merged_3_reduced = merged_3 + 'reduced/'
    endpoints_3a = base_dir + '3a_endpoints/180214/'
    endpoints_3a_reduced = endpoints_3a + 'reduced/'
    imputed_5 = base_dir + '5_imputed/imputed_180317/reduced/'
    # these need updating
    labels_6 = base_dir + '6_labels/targets_180424/reduced/'
    mlinput_7 = base_dir + '7_ml_input/180430/reduced/'
