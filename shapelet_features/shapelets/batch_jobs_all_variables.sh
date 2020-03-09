#!/usr/bin/env bash
#
# Finds shapelets for a set of variables, using a set of pre-processed
# time series with an accompanying S3M endpoint. Note that this script
# will submit numerous jobs (using `bsub`) but will *not* wait for the
# execution.

usage="$(basename "$0") [-h] -d -v -s -o -l -t -c -r -m -M --- Convert files created through our pipeline to create S3M ready csv file

Available options:
  -h  show this help text
  -d  file directory of CSV files
  -v  variables comma separated (e.g. v200,v110)
  -s  sample size
  -o  output directory
  -l  lengths of time series comma separated (e.g. 8,9)
  -t  delta ts comma separated (e.g. 0.1,0.16,1)
  -c  directory of created, s3m readable files
  -r  directory of s3m results
  -m  min length of shapelet"

INDIR=""
VARIABLE=""
SAMPLE_SIZE=50

while getopts "hd:v:s:o:l:t:c:r:m:M:" option; do
    case "${option}" in
    h) echo "$usage"
      exit
      ;;
    d) INDIR=${OPTARG};;
    v) VARIABLES=${OPTARG};;
    s) SAMPLE_SIZE=${OPTARG};;
    o) OUTDIR=${OPTARG};;
    l) LENGTHS=${OPTARG};;
    t) DTS=${OPTARG};;
    c) DATA_PATH=${OPTARG};;
    r) RESULT_PATH=${OPTARG};;
    m) MIN_LENGTH=${OPTARG};;
    esac
done

if [ "$OUTDIR" == "" ] ; then
  OUTDIR=./
fi

if [ "$LENGTHS" == "" ] ; then
  echo "No shapelet length specified; defaulting to 10 points"
  LENGTHS=10
fi

if [ "$DTS" == "" ] ; then
  DTS=1
fi

# Parse arguments into interables; variables and lengths are assumed to be separated
# by commas but we want to parse them into a long string.
VARIABLES=$(echo $VARIABLES | tr "," "\n")
DTS=$(echo $DTS | tr "," "\n")
LENGTHS=$(echo $LENGTHS | tr "," "\n")

# Determine execution path of the script so that we are able to call
# a subordinate script in another directory.
EXECUTION_PATH=$(dirname "${BASH_SOURCE[0]}")

for VAR in ${VARIABLES}
do
  for DT in ${DTS}
  do
    for L in ${LENGTHS}
    do
     bash ${EXECUTION_PATH}/create_s3m_readable.sh -d ${INDIR} -v ${VAR} -t ${DT} -l ${L} -o ${DATA_PATH} -s ${SAMPLE_SIZE} || exit 1

     # TODO: counter could also be determined from filename; requires
     # regexp magic or something
     COUNTER=0
     for S3M_READABLE_FILE in ${DATA_PATH}/${VAR}_${SAMPLE_SIZE}_${L}h_${DT}h_*_S3M.csv; do
       bsub -J "S3M_EXTRACTION_${VAR}_${L}_${DT}_${MIN_LENGTH}" -R "rusage[mem=10240]" "s3m -i ${S3M_READABLE_FILE} -m ${MIN_LENGTH} -M ${MIN_LENGTH} -q -l 3 -e 0,1,2 -o ${RESULT_PATH}/${VAR}_$((SAMPLE_SIZE*2))_${L}h_${DT}h_m${MIN_LENGTH}_block_${COUNTER}.json"
       COUNTER=$((COUNTER+1))
     done
    done
  done
done


