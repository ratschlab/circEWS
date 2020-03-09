# Main `Makefile` for running the current version of the complete
# pipeline.
MAKEFLAGS=--warn-undefined-variables
# Store the variables defined to this point, we need this to filter out make-internal variables
VARS_OLD := $(.VARIABLES)

########################################################################
# Configuration
########################################################################

DATASET_VERSION      ?= v6b_reduced

# Duration (in hours) for time series extraction, as well as the delta
# (in hours) to the onset.
TIME_SERIES_DURATION ?= 8
TIME_SERIES_DT       ?= 0.1

S3M_SAMPLE_SIZE      ?= 300
N_WORKERS            ?= 25
N_SELECTED_SHAPELETS ?= 10
SHORT_SHAPELET		?=6
LONG_SHAPELET		?=12

SHAPELET_SELECT  ?=min-max#random top
SHAPELET_TRANSFORMS  ?=dist-set#distance #counts normalized_counts min

# FIXME: document
DROP_VARIABLES_CLASSIFICATION   ?=SampleStatus Sex

########################################################################
# Paths
########################################################################

OUTPUT_DIRECTORY                ?=/cluster/work/borgw/Bern_ICU_Sanctuary/$(DATASET_VERSION)

SORTED_TRANSFORMS              ?= $(sort $(SHAPELET_TRANSFORMS))
SHAPELET_TRANSFORMS_NAME        ?= $(shell echo "$(SORTED_TRANSFORMS)" | sed -e 's/ /_/g')
SHAPELET_CLASSIFIER_NAME        ?=Shapelet_$(SHAPELET_SELECT)_$(SHAPELET_TRANSFORMS_NAME)
CLASSICAL_CLASSIFIER_NAME       ?=Classical
COMBINED_CLASSIFIER_NAME        ?=Combined_$(SHAPELET_SELECT)_$(SHAPELET_TRANSFORMS_NAME)

S3M_OUTPUT_DIRECTORY            ?=$(OUTPUT_DIRECTORY)/s3m/
S3M_OUTPUT_DIRECTORY_TEMP       ?=$(OUTPUT_DIRECTORY)/s3m_tmp/
S3M_OUTPUT_DIRECTORY_SL			?=$(OUTPUT_DIRECTORY)/s3m_$(SHAPELET_CLASSIFIER_NAME)/#FOr selection
SHAPELET_FEATURES               ?=$(OUTPUT_DIRECTORY)/$(SHAPELET_CLASSIFIER_NAME)_features.h5

CLASSICAL_CLASSIFICATION_MATRIX ?=$(OUTPUT_DIRECTORY)/Classical_classification_matrix.h5
SHAPELET_CLASSIFICATION_MATRIX  ?=$(OUTPUT_DIRECTORY)/$(SHAPELET_CLASSIFIER_NAME)_classification_matrix.h5
COMBINED_CLASSIFICATION_MATRIX  ?=$(OUTPUT_DIRECTORY)/$(COMBINED_CLASSIFIER_NAME)_classification_matrix.h5


CLASSIFIERS_PATH                ?=$(OUTPUT_DIRECTORY)/trained_classifiers
SHAPELET_CLASSIFIER             ?=$(CLASSIFIERS_PATH)/$(SHAPELET_CLASSIFIER_NAME).pickle
CLASSICAL_CLASSIFIER            ?=$(CLASSIFIERS_PATH)/$(CLASSICAL_CLASSIFIER_NAME).pickle
COMBINED_CLASSIFIER             ?=$(CLASSIFIERS_PATH)/$(COMBINED_CLASSIFIER_NAME).pickle

PREDICTIONS_PATH                ?=$(OUTPUT_DIRECTORY)/predictions
SHAPELET_PREDICTIONS            ?=$(PREDICTIONS_PATH)/$(SHAPELET_CLASSIFIER_NAME).h5
CLASSICAL_PREDICTIONS           ?=$(PREDICTIONS_PATH)/$(CLASSICAL_CLASSIFIER_NAME).h5
COMBINED_PREDICTIONS            ?=$(PREDICTIONS_PATH)/$(COMBINED_CLASSIFIER_NAME).h5

EVALUATION_PATH                 ?=$(OUTPUT_DIRECTORY)/event_evaluation
# Borgwardt evaluation
SHAPELET_EVALUATION_BORGW       ?=$(EVALUATION_PATH)/$(SHAPELET_CLASSIFIER_NAME)_borgw.npz
CLASSICAL_EVALUATION_BORGW      ?=$(EVALUATION_PATH)/$(CLASSICAL_CLASSIFIER_NAME)_borgw.npz
COMBINED_EVALUATION_BORGW       ?=$(EVALUATION_PATH)/$(COMBINED_CLASSIFIER_NAME)_borgw.npz
EVALUATION_ALL_BORGW			?=$(OUTPUT_DIRECTORY)/plots/

# Ratsch evaluation
SHAPELET_EVALUATION_RATSCH      ?= $(EVALUATION_PATH)/$(SHAPELET_CLASSIFIER_NAME)_ratsch.npz
CLASSICAL_EVALUATION_RATSCH     ?= $(EVALUATION_PATH)/$(CLASSICAL_CLASSIFIER_NAME)_ratsch.npz
COMBINED_EVALUATION_RATSCH      ?= $(EVALUATION_PATH)/$(COMBINED_CLASSIFIER_NAME)_ratsch.npz

# Hyperparameter search
CLASSIFIER_PARAMETERS ?=misc/ratschlab_parameters.json
HYPERPARAMETER_GRID ?=misc/ratschlab_hyperparameter_space.json

HYPERPARAMETERS_PATH            ?=$(OUTPUT_DIRECTORY)/hyperparameters
SHAPELET_HYPERPARAMETERS         ?=$(HYPERPARAMETERS_PATH)/$(SHAPELET_CLASSIFIER_NAME).json
CLASSICAL_HYPERPARAMETERS        ?=$(HYPERPARAMETERS_PATH)/$(CLASSICAL_CLASSIFIER_NAME).json
COMBINED_HYPERPARAMETERS         ?=$(HYPERPARAMETERS_PATH)/$(COMBINED_CLASSIFIER_NAME).json
SHAPELET_HYPERPARAMETERS_SUMMARY         ?=$(HYPERPARAMETERS_PATH)/$(SHAPELET_CLASSIFIER_NAME)_summary.csv
CLASSICAL_HYPERPARAMETERS_SUMMARY        ?=$(HYPERPARAMETERS_PATH)/$(CLASSICAL_CLASSIFIER_NAME)_summary.csv
COMBINED_HYPERPARAMETERS_SUMMARY         ?=$(HYPERPARAMETERS_PATH)/$(COMBINED_CLASSIFIER_NAME)_summary.csv

SPLIT_COLUMN_ARGUMENT ?=random_4

########################################################################
# Fixed paths
########################################################################
#
# For a regular run of the pipeline, nothing should have to be changed
# in the variables below.

IMPUTED_FILES             ?=$(wildcard $(shell python3 utils/datasets.py --dataset $(DATASET_VERSION) --get-setting imputed_variables_path)/batch_*.h5)

ifndef IMPUTED_FILES
IMPUTED_FILES             ?=$(wildcard $(shell python3 utils/datasets.py --dataset $(DATASET_VERSION) --get-setting imputed_variables_path)/*.h5)
endif

ENDPOINT_FILES            ?=$(wildcard $(shell python3 utils/datasets.py --dataset $(DATASET_VERSION) --get-setting endpointdir)/*.h5)
SPLIT_FILE                ?=$(shell python3 utils/datasets.py --dataset $(DATASET_VERSION) --get-setting split_path)
$(IMPUTED_FILES): ;
$(ENDPOINT_FILES): ;
$(SPLIT_FILE): ;
$(SELECTED_VARIABLES_LIST): ;
$(SELECTED_VARIABLES_FILE): ;


SELECTED_VARIABLES_FILE   ?=$(shell python3 utils/datasets.py --dataset $(DATASET_VERSION) --get-setting selected_variables)
SELECTED_VARIABLES_LIST   ?=$(shell cut -f 1 -d , $(SELECTED_VARIABLES_FILE))

COLLECTED_TIME_SERIES_DIR ?=$(OUTPUT_DIRECTORY)/CSV/dynamic_$(TIME_SERIES_DURATION)h_$(TIME_SERIES_DT)h
COLLECTED_TIME_SERIES_LOG ?=$(COLLECTED_TIME_SERIES_DIR).log

# Maps patients to their respective time points, exactly as they are
# stored in the imputed data files.
PATIENT_TIMEPOINTS        ?=$(OUTPUT_DIRECTORY)/patient_timepoints.json

# Stores all extracted time series along with a specific endpoint for
# S3M. These time series will be used for shapelet extraction.
EXTRACTED_ENDPOINT_TIMES  ?=$(OUTPUT_DIRECTORY)/endpoint_times/dynamic_$(TIME_SERIES_DURATION)h_${TIME_SERIES_DT}h.json

_SELECTED_SHAPELETS       ?=$(OUTPUT_DIRECTORY)/s3m/.ran_min_max_shapelet_selection

# Load a config file if specified
ifdef CONFIG_FILE
	include $(CONFIG_FILE)
endif
# CONFIG_VARIABLES contains all variables that were set during configuration phase
CONFIG_VARIABLES :=$(filter-out $(VARS_OLD) VARS_OLD,$(.VARIABLES))

# Filtering of features at different steps
# TODO: Extend to other cases
ifdef DROP_VARIABLES_CLASSIFICATION
	FILTER_CLASSIFICATION :=--drop-variables $(DROP_VARIABLES_CLASSIFICATION)
else
	FILTER_CLASSIFICATION :=
endif

ifdef KEEP_VARIABLES_BUILD_CLASSIFICATION_MATRIX
	FILTER_BUILD_CLASSIFICATION_MATRIX :=--keep-variables $(KEEP_VARIABLES_BUILD_CLASSIFICATION_MATRIX)
else
	FILTER_BUILD_CLASSIFICATION_MATRIX :=
endif


########################################################################
# Targets
########################################################################

all: print extract_time_series extract_shapelets shapelet_features create_classification_matrices fit_classifiers evaluate_all_borgw

print:
	$(info ---------------------- Variables ----------------------)
	$(foreach v, $(CONFIG_VARIABLES), $(info $(v) = $($(v))))
	$(info -------------------- End Variables --------------------)
	$(info )

########################################################################
# Time series extraction
########################################################################

extract_time_series: $(PATIENT_TIMEPOINTS) extract_time_series_s3m $(COLLECTED_TIME_SERIES_LOG)
	@echo Finished time series extraction

$(PATIENT_TIMEPOINTS): $(IMPUTED_FILES)
	@echo Extracting patient timepoints
	@mkdir -p $(dir $(PATIENT_TIMEPOINTS))
	@python3 -m shapelets.extract_all_patient_timepoints $(IMPUTED_FILES) --output $(PATIENT_TIMEPOINTS)

# Verbose target for S3M time series extraction. This makes it possible
# to call the target by name rather than by filename.
extract_time_series_s3m: $(EXTRACTED_ENDPOINT_TIMES)
	@echo Extracted time series for subsequent use in S3M

$(EXTRACTED_ENDPOINT_TIMES): $(ENDPOINT_FILES) $(PATIENT_TIMEPOINTS)
	@echo Extracting time series with S3M endpoints
	@mkdir -p $(dir $(EXTRACTED_ENDPOINT_TIMES))
	@python3 -m shapelets.extract_endpoint_times_dynamic $(ENDPOINT_FILES) --align-timepoints $(PATIENT_TIMEPOINTS) --duration $(TIME_SERIES_DURATION) --dt $(TIME_SERIES_DT) --output $(EXTRACTED_ENDPOINT_TIMES)

# FIXME: dependencies are not optimal here; we should check for the log
# to be written and *only* stop the script if said log already exists
$(COLLECTED_TIME_SERIES_LOG): $(PATIENT_TIMEPOINTS) $(EXTRACTED_ENDPOINT_TIMES)
	@echo Collecting time series for CSV creation
	@# Create everything but the *last* part of the requested directory, as
	@# this ensures that the target below always run at least once.
	@mkdir -p $(dir $(COLLECTED_TIME_SERIES_DIR))
	@python3 -m shapelets.collect_time_series $(IMPUTED_FILES) --variables $(SELECTED_VARIABLES_LIST) --event-times $(EXTRACTED_ENDPOINT_TIMES) --output $(COLLECTED_TIME_SERIES_DIR)/ --logfile $(COLLECTED_TIME_SERIES_LOG)

########################################################################
# Shapelet extraction and shapelet feature creation
########################################################################

extract_shapelets: extract_time_series train_test_split extract_all_variables #select_shapelets
	@echo Extracted shapelets

train_test_split: $(COLLECTED_TIME_SERIES_DIR)/train
	@echo Created test/train/validation split of extracted variables

# Splits the collected time series, i.e. every variable, into
# train/test/validation files. No shapelets are present yet.
$(COLLECTED_TIME_SERIES_DIR)/train:
	@echo Creating train/test/validation splits of pre-processed files
	@echo Using split $(SPLIT_FILE)
	@python3 -m shapelets.train_test_split -d $(COLLECTED_TIME_SERIES_DIR) -s $(SPLIT_FILE) -k $(SPLIT_COLUMN_ARGUMENT)

# Extracts all shapelets from all specified variables. This target uses
# a cop-out strategy so far because it only checks whether the *output*
# directory exists.
#
# This is a simple 'dummy' target in order to have readable names.
extract_all_variables: $(S3M_OUTPUT_DIRECTORY)
	@echo Extracted all variables for S3M

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
COMMA := ,
SELECTED_VARIABLES_COMMA_SEPARATED = $(subst $(SPACE),$(COMMA),$(SELECTED_VARIABLES_LIST))

$(S3M_OUTPUT_DIRECTORY):
	@echo Extracting all variables for S3M
	@mkdir -p $(S3M_OUTPUT_DIRECTORY)
	@mkdir -p $(S3M_OUTPUT_DIRECTORY_TEMP)
	@{                      \
    set -e               ;\
	  for m in $(SHORT_SHAPELET) $(LONG_SHAPELET); do  \
		  echo 'Length: $$m' ;\
			bash shapelets/batch_jobs_all_variables.sh -v $(SELECTED_VARIABLES_COMMA_SEPARATED) -l $(TIME_SERIES_DURATION) -t $(TIME_SERIES_DT) -d $(OUTPUT_DIRECTORY)/CSV -c $(S3M_OUTPUT_DIRECTORY_TEMP) -s $(S3M_SAMPLE_SIZE) -m $$m -r $(S3M_OUTPUT_DIRECTORY)           ;\
	  done                 ;\
   }
	@echo "Waiting until all submitted shapelet extraction jobs have ended"
	@bwait -w 'ended("S3M_EXTRACTION*")'


#This target is to explore shapelet feature enineering approaches
shapelet_features: select_shapelets shapelet_transform

# Dummy target for selecting shapelets. Currently, we use a min--max
# approach to increase variability.
select_shapelets: $(S3M_OUTPUT_DIRECTORY_SL)
	@echo Selected shapelets

# FIXME: this target cannot easily be re-made because it depends on
# a hidden file
$(S3M_OUTPUT_DIRECTORY_SL): $(S3M_OUTPUT_DIRECTORY)
	@echo Selecting shapelets
	@python3 -m shapelets.select_shapelets_min_max $(wildcard $(S3M_OUTPUT_DIRECTORY)/vm*_block*.json) -n $(N_SELECTED_SHAPELETS) -o $(S3M_OUTPUT_DIRECTORY_SL)
	@touch $@

shapelet_transform: $(SHAPELET_FEATURES)
	@echo Finished shapelet feature matrix creation

$(SHAPELET_FEATURES): $(IMPUTED_FILES) $(S3M_OUTPUT_DIRECTORY_SL)
	@echo Creating shapelet features
	@echo $(S3M_OUTPUT_DIRECTORY_SL)/*_min_max_shapelets.json
	@python3 -m shapelets.create_feature_matrix_from_s3m $(IMPUTED_FILES) --s3m-shapelets $(S3M_OUTPUT_DIRECTORY_SL)/*_min_max_shapelets.json --output $@ --n-workers $(N_WORKERS) --transforms $(SHAPELET_TRANSFORMS)

########################################################################
# Classification matrix generation (classic & shapelets)
########################################################################

create_classification_matrices: create_classical_classification_matrix create_shapelet_classification_matrix create_combined_classification_matrix
	@echo Created classification matrices

# Dummy target for creating the classical classification matrix, i.e.
# the one that contains only the 'initial' features.
create_classical_classification_matrix: $(CLASSICAL_CLASSIFICATION_MATRIX)
	@echo Created classical classification matrix: $<

# Actual target for creating the classical classification matrix, i.e.
# the one that contains only the 'initial' features. The target's name
# is the filename of the matrix, which ensures that the matrix is only
# built once.
$(CLASSICAL_CLASSIFICATION_MATRIX):
	@echo Creating classical classification matrix
	@python3 -m classification.create_classical_classification_matrix --output $@ --dataset-version $(DATASET_VERSION) $(FILTER_BUILD_CLASSIFICATION_MATRIX)

# Dummy target for creating the shapelet-based classification matrix,
# i.e. the matrix that only contains shapelet features.
create_shapelet_classification_matrix: $(SHAPELET_CLASSIFICATION_MATRIX)
	@echo Created shapelet classification matrix: $<

# Actual target for the shapelet-based classification matrix.
$(SHAPELET_CLASSIFICATION_MATRIX): $(SHAPELET_FEATURES)
	@echo Creating shapelet classification matrix
	@python3 -m classification.create_shapelet_classification_matrix --shapelet-features $< --output $@ --dataset-version $(DATASET_VERSION) $(FILTER_BUILD_CLASSIFICATION_MATRIX)

# Dummy target for creating the combined classification matrix, i.e. the
# one that is a 'hybrid' of shapelet and non-shapelet features.
create_combined_classification_matrix: $(COMBINED_CLASSIFICATION_MATRIX)
	@echo Created combined classification matrix: $<

# Actual target for the combined classification matrix.
$(COMBINED_CLASSIFICATION_MATRIX): $(SHAPELET_CLASSIFICATION_MATRIX) $(CLASSICAL_CLASSIFICATION_MATRIX)
	@echo Creating combined classification matrix
	@python3 -m classification.merge_classification_matrices --classification-matrices $^ --output $@ --n-workers $(N_WORKERS)


########################################################################
# Fitting of classifiers
########################################################################

fit_classifiers: fit_shapelet_classifier fit_classical_classifier fit_combined_classifier
	@echo Finished fitting classifiers

fit_shapelet_classifier: $(SHAPELET_PREDICTIONS)
	@echo Fitted shapelet classifer

fit_classical_classifier: $(CLASSICAL_PREDICTIONS)
	@echo Fitted classical classifier

fit_combined_classifier: $(COMBINED_PREDICTIONS)
	@echo Fitted combined classifier

$(SHAPELET_PREDICTIONS): $(SHAPELET_CLASSIFICATION_MATRIX) $(SPLIT_FILE)
	@echo Fitting shapelet classifier
	@-mkdir -p $(dir $(SHAPELET_CLASSIFIER))
	@-mkdir -p $(dir $@)
	@python3 -m classification.classifier --classification-matrix $(SHAPELET_CLASSIFICATION_MATRIX) --split $(SPLIT_FILE) --output-classifier $(SHAPELET_CLASSIFIER) --output-predictions $(SHAPELET_PREDICTIONS) --njobs $(N_WORKERS) $(FILTER_CLASSIFICATION)

$(CLASSICAL_PREDICTIONS): $(CLASSICAL_CLASSIFICATION_MATRIX) $(SPLIT_FILE)
	@echo Fitting classical classifier
	@-mkdir -p $(dir $(CLASSICAL_CLASSIFIER))
	@-mkdir -p $(dir $@)
	@python3 -m classification.classifier --classification-matrix $(CLASSICAL_CLASSIFICATION_MATRIX) --split $(SPLIT_FILE) --output-classifier $(CLASSICAL_CLASSIFIER) --output-predictions $(CLASSICAL_PREDICTIONS) --njobs $(N_WORKERS) $(FILTER_CLASSIFICATION)

$(COMBINED_PREDICTIONS): $(COMBINED_CLASSIFICATION_MATRIX) $(SPLIT_FILE)
	@echo Fitting combined classifier
	@-mkdir -p $(dir $(COMBINED_CLASSIFIER))
	@-mkdir -p $(dir $@)
	@python3 -m classification.classifier --classification-matrix $(COMBINED_CLASSIFICATION_MATRIX) --split $(SPLIT_FILE) --output-classifier $(COMBINED_CLASSIFIER) --output-predictions $(COMBINED_PREDICTIONS) --njobs $(N_WORKERS) $(FILTER_CLASSIFICATION)

########################################################################
# Evaluation of predictions
########################################################################

evaluate_classifiers: evaluate_shapelet_classifier evaluate_classical_classifier evaluate_combined_classifier
	@echo Evaluated all classifiers

evaluate_shapelet_classifier: evaluate_shapelet_classifier_borgw evaluate_shapelet_classifier_ratsch
evaluate_classical_classifier: evaluate_classical_classifier_borgw evaluate_classical_classifier_ratsch
evaluate_combined_classifier: evaluate_combined_classifier_borgw evaluate_combined_classifier_ratsch

evaluate_classifiers_borgw: evaluate_shapelet_classifier_borgw evaluate_classical_classifier_borgw evaluate_combined_classifier_borgw
	@echo Evaluated classifiers using Borgwardt scripts
evaluate_classifiers_ratsch: evaluate_shapelet_classifier_ratsch evaluate_classical_classifier_ratsch evaluate_combined_classifier_ratsch
	@echo Evaluated classifiers using Ratsch scripts

evaluate_shapelet_classifier_borgw: $(SHAPELET_EVALUATION_BORGW)
evaluate_classical_classifier_borgw: $(CLASSICAL_EVALUATION_BORGW)
evaluate_combined_classifier_borgw: $(COMBINED_EVALUATION_BORGW)

$(SHAPELET_EVALUATION_BORGW): $(SHAPELET_PREDICTIONS) $(ENDPOINT_FILES)
	@echo Evaluating shapelet classifier using Borgwardt script
	@-mkdir -p $(dir $@)
	@python3 -m analysis.event_evaluation_borgw --output $@ --pred-file $< --endpoints $(ENDPOINT_FILES)

$(CLASSICAL_EVALUATION_BORGW): $(CLASSICAL_PREDICTIONS) $(ENDPOINT_FILES)
	@echo Evaluating shapelet classifier using Borgwardt script
	@-mkdir -p $(dir $@)
	@python3 -m analysis.event_evaluation_borgw --output $@ --pred-file $< --endpoints $(ENDPOINT_FILES)

$(COMBINED_EVALUATION_BORGW): $(COMBINED_PREDICTIONS) $(ENDPOINT_FILES)
	@echo Evaluating shapelet classifier using Borgwardt script
	@-mkdir -p $(dir $@)
	@python3 -m analysis.event_evaluation_borgw --output $@ --pred-file $< --endpoints $(ENDPOINT_FILES)

evaluate_all_borgw: $(ENDPOINT_FILES)
	@echo Evaluating all predictions using Borgwardt script
	@-rm -r $(EVALUATION_ALL_BORGW)
	@-mkdir -p $(EVALUATION_ALL_BORGW)
	@-mkdir -p $(EVALUATION_PATH)
	@bsub -R "rusage[mem=15000]" -W 23:00 python -m analysis.plot_for_paper -b $(OUTPUT_DIRECTORY) -e $(ENDPOINT_FILES) -o $(EVALUATION_ALL_BORGW)


evaluate_all_ARL: $(ENDPOINT_FILES)
	@echo Evaluating all predictions using Borgwardt script
	@-mkdir -p $(EVALUATION_ALL_BORGW)
	@-mkdir -p $(EVALUATION_PATH)
	@bsub -R "rusage[mem=15000]" -W 120:00 python -m analysis.plot_for_RL -b $(OUTPUT_DIRECTORY) -e $(ENDPOINT_FILES) -o $(EVALUATION_ALL_BORGW)



evaluate_shapelet_classifier_ratsch: $(SHAPELET_EVALUATION_RATSCH)
evaluate_classical_classifier_ratsch: $(CLASSICAL_EVALUATION_RATSCH)
evaluate_combined_classifier_ratsch: $(COMBINED_EVALUATION_RATSCH)

$(SHAPELET_EVALUATION_RATSCH): $(SHAPELET_PREDICTIONS) $(ENDPOINT_FILES)
	@echo Evaluating shapelet classifier using Ratsch script
	@-mkdir -p $(dir $@)
	@python3 -m analysis.event_evaluation_ratsch --output $@ --pred-file $< --endpoints $(ENDPOINT_FILES)

$(CLASSICAL_EVALUATION_RATSCH): $(CLASSICAL_PREDICTIONS) $(ENDPOINT_FILES)
	@echo Evaluating shapelet classifier using Ratsch script
	@-mkdir -p $(dir $@)
	@python3 -m analysis.event_evaluation_ratsch --output $@ --pred-file $< --endpoints $(ENDPOINT_FILES)

$(COMBINED_EVALUATION_RATSCH): $(COMBINED_PREDICTIONS) $(ENDPOINT_FILES)
	@echo Evaluating shapelet classifier using Ratsch script
	@-mkdir -p $(dir $@)
	@python3 -m analysis.event_evaluation_ratsch --output $@ --pred-file $< --endpoints $(ENDPOINT_FILES)


search_hyperparameters: search_shapelet_hyperparameters search_classical_hyperparameters search_combined_hyperparameters

search_shapelet_hyperparameters: $(SHAPELET_HYPERPARAMETERS)
search_classical_hyperparameters: $(CLASSICAL_HYPERPARAMETERS)
search_combined_hyperparameters: $(COMBINED_HYPERPARAMETERS)

$(SHAPELET_HYPERPARAMETERS): $(SHAPELET_CLASSIFICATION_MATRIX) $(SPLIT_FILE)
	@echo Fitting shapelet classifier
	@-mkdir -p $(dir $@)
	@python3 -m classification.hyperparameter_search \
	    --classification-matrix $(SHAPELET_CLASSIFICATION_MATRIX) \
	    --split $(SPLIT_FILE) \
	    --classifier-parameters $(CLASSIFIER_PARAMETERS) \
	    --parameter-grid $(HYPERPARAMETER_GRID) \
	    --output-parameters $(SHAPELET_HYPERPARAMETERS) \
	    --output-summary $(SHAPELET_HYPERPARAMETERS_SUMMARY) \
	    --njobs $(N_WORKERS) $(FILTER_CLASSIFICATION)

$(CLASSICAL_HYPERPARAMETERS): $(CLASSICAL_CLASSIFICATION_MATRIX) $(SPLIT_FILE)
	@echo Fitting classical classifier
	@-mkdir -p $(dir $@)
	@python3 -m classification.hyperparameter_search \
	    --classification-matrix $(CLASSICAL_CLASSIFICATION_MATRIX) \
	    --split $(SPLIT_FILE) \
	    --classifier-parameters $(CLASSIFIER_PARAMETERS) \
	    --parameter-grid $(HYPERPARAMETER_GRID) \
	    --output-parameters $(CLASSICAL_HYPERPARAMETERS) \
	    --output-summary $(CLASSICAL_HYPERPARAMETERS_SUMMARY) \
	    --njobs $(N_WORKERS) $(FILTER_CLASSIFICATION)

$(COMBINED_HYPERPARAMETERS): $(COMBINED_CLASSIFICATION_MATRIX) $(SPLIT_FILE)
	@echo Fitting combined classifier
	@-mkdir -p $(dir $@)
	@python3 -m classification.hyperparameter_search \
	    --classification-matrix $(COMBINED_CLASSIFICATION_MATRIX) \
	    --split $(SPLIT_FILE) \
	    --classifier-parameters $(CLASSIFIER_PARAMETERS) \
	    --parameter-grid $(HYPERPARAMETER_GRID) \
	    --output-parameters $(COMBINED_HYPERPARAMETERS) \
	    --output-summary $(COMBINED_HYPERPARAMETERS_SUMMARY) \
	    --njobs $(N_WORKERS) $(FILTER_CLASSIFICATION)
