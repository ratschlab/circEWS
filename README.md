# Code for *Early prediction of circulatory failure in the intensive care unit using machine learning*

Public repository containing research code for the circEWS project accompanying the manuscript
 *Early prediction of circulatory failure in the intensive care unit using machine learning*

When using code from this repository, please consider citing

> Hyland, S.L. et al. Early prediction of circulatory failure in the intensive care unit using machine learning. Nat Med (2020). https://doi.org/10.1038/s41591-020-0789-4.

The code is organized in several sub-directories, which contain the following content:

* **`binarize`** 
Binarize time-grid data to only keep measurement patterns.

* **`calibration`** 
Calibration analysis of continuous risk scores of circEWS.

* **`circews`** 
Classes and utility functions.

* **`circulatory_status`** 
Annotation of time series with status of stability or stages of circulatory failure.

* **`dimensionality_reduction`** 
Merging of raw HIRID variables corresponding to identical clinical concepts into meta-variables.

* **`evaluation`** 
Evaluation of alarm system performance.

* **`external_validation`** 
Code for external validation on the MIMIC data-set.

* **`features`**  
Contains code for generation of non-shapelet features from imputed data.

* **`finetuning`**  
Interpolation of MIMIC/HIRID based models to fine-tune circEWS towards the MIMIC database.

* **`imputation`**  
Code concerned with transforming HIRID data to a fixed time grid, making it suitable for 
feature generation and fitting of machine learning models.

* **`labels`**  
Code for creating labels where positive labels correspond to time points where it 
is desirable to raise an alarm, located in the 8 hours prior to circulatory failure events.

* **`learning`**  
Supervised learning scripts for learning a continuous risk score for predicting
circulatory failure.

* **`lstm`**  
LSTM model implementation.

* **`pipeline_diagnostics`**  
Diagnostic code for tracking PIDs in different pipeline stages, and others.

* **`preprocessing`**  
Code for preprocessing the HIRID data, including artifact deletion strategies and others.

* **`shapelet_features`**  
Code concerned with discovering and applying shapelet features on the HIRID data
indicative of future circulatory failure.

* **`splits`**  
Code concerned with splitting PIDs for cluster processing and generating data
splits for the experimental design.

* **`visualization`**  
Code concerned with visualizing patient stays, using data from different pipeline stages.






