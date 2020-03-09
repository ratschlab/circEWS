Performing external validation on MIMIC-III requires selecting the relevant variables, and processing MIMIC to the same format as the HiRID dataset, so we can run the rest of the processing pipeline on it.

The `run_mimic_prep.py` is the wrapper which calls on the other scripts, so starting there is recommended.
