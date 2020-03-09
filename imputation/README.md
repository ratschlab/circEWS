Scripts related to imputation of the circEWS data to a fixed time grid and 
imputation of static data.

### cluster_impute.py
Cluster dispatcher for the dynamic data imputation script.

### cluster_save_imputation_params.py
Cluster dispatcher for the imputation parameter script.

### impute_one_batch.py
Imputation of patient data to a fixed time grid, supporting
different imputation strategies.

### impute_static_data.py
Imputation of static admission data for each patient, based on statistics from the 
training data-set.

### save_imputation_params.py
Saving of imputation parameters based on statistics from the training set of a split.

### save_var_auxiliary_dicts.py
Get meta-data about the individual variables from the Excel sheet, to be
used in imputation.