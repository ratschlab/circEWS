Classes for feature generation/imputation/machine learning.

### feat_gen_nan.py
Feature generation transformer that can deal with missing input data.

### feat_gen.py
Transformer for generation of non-shapelet features.

### imputer_ffill.py
Imputation transformer using a simple forward filling strategy.

### imputer_none.py
Imputation model creating only a fixed time grid, but does not
fill in missing values.

### imputer.py
Transformer using an adaptive imputation strategy where 
different strategies are applied to each variables based on 
clinical knowledge and statistics from the training set.

### imputer_static.py
Imputation model for static data known at the patient admission time.

### label_gen.py
Transformer for label generation from imputed data and state annotations
for circulatory failure.

### lgbm_model.py
Machine learning constructing feature matrices, and fitting to supervised
learning data, using the LightGBM library, also other baselines like
logistic regression and other gradient boosting libraries are supported.
