#!/bin/bash

#BSUB -R "rusage[mem=15000]"
# set -o xtrace
module load hdf5 python_cpu/3.6.1


DIR=../../../data/9_s3m/
mkdir -p $DIR/high
mkdir -p $DIR/med
mkdir -p $DIR/low
mkdir -p $DIR/s3m


make OUTPUT_DIRECTORY=$DIR/high DATASET_VERSION=HIRID SELECTED_VARIABLES_FILE=./misc/selected_variables_high.csv N_SELECTED_SHAPELETS=20 TIME_SERIES_DURATION=4 TIME_SERIES_DT=0.1 SHORT_SHAPELET=6 LONG_SHAPELET=12 print extract_shapelets

cp $DIR/high/s3m/*.json $DIR/s3m/


make OUTPUT_DIRECTORY=$DIR/med DATASET_VERSION=HIRID SELECTED_VARIABLES_FILE=./misc/selected_variables_med.csv N_SELECTED_SHAPELETS=20 TIME_SERIES_DURATION=36 TIME_SERIES_DT=0.1 SHORT_SHAPELET=144 LONG_SHAPELET=288 print extract_shapelets

cp $DIR/med/s3m/*.json $DIR/s3m/

make OUTPUT_DIRECTORY=$DIR/low DATASET_VERSION=HIRID SELECTED_VARIABLES_FILE=./misc/selected_variables_low.csv N_SELECTED_SHAPELETS=20 TIME_SERIES_DURATION=48 TIME_SERIES_DT=0.1 SHORT_SHAPELET=192 LONG_SHAPELET=384 print extract_shapelets

cp $DIR/low/s3m/*.json $DIR/s3m/

make OUTPUT_DIRECTORY=$DIR/ DATASET_VERSION=HIRID shapelet_features

