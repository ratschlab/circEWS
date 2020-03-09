#!/usr/bin/env bash
#
# Creates files that are readable for S3M. Most notably, this script
# uses pre-processed cases and controls files to create *blocks* for
# a balanced shapelet extraction.

usage="$(basename "$0") [-h] -d -v -s -o -l -t -- Convert files created through our pipeline to create S3M ready csv file

Available options:
  -h  show this help text
  -d  file directory
  -v  variable 
  -s  sample size
  -o  output directory
  -l  length of time series
  -t  delta t "

INDIR=""
VARIABLE=""
SAMPLE_SIZE=50

while getopts "hd:v:s:o:l:t:" option; do
    case "${option}" in
    h) echo "$usage"
      exit
      ;;
    d) INDIR=${OPTARG};;
    v) VARIABLE=${OPTARG};;
    s) SAMPLE_SIZE=${OPTARG};;
    o) OUTDIR=${OPTARG};;
    l) LENGTH=${OPTARG};;
    t) DT=${OPTARG};;
    esac
done

if [ "$OUTDIR" == "" ] ; then
  OUTDIR=./
fi

if [ "$LENGTH" == "" ] ; then
  LENGTH=8
fi

if [ "$DT" == "" ] ; then
  DT=1
fi

INDIR=${INDIR}/dynamic_${LENGTH}h_${DT}h/train

CASE_FILE=${OUTDIR}/${VARIABLE}_${LENGTH}h_${DT}h_cases.csv
CONTROL_FILE=${OUTDIR}/${VARIABLE}_${LENGTH}h_${DT}h_controls.csv

echo "Creating cases and controls files for variable ${VARIABLE}, length ${LENGTH}, dt ${DT} of files in ${INDIR}"

awk -F"," '$4!=0 {print $0}' ${INDIR}/${VARIABLE}.csv > ${CASE_FILE}
awk -F"," '$4==0 {print $0}' ${INDIR}/${VARIABLE}.csv > ${CONTROL_FILE}

echo "Dividing cases and controls files into blocks of size $((SAMPLE_SIZE*2)) (sample size per class = ${SAMPLE_SIZE})"

python shapelets/make_blocks.py ${CASE_FILE} ${CONTROL_FILE} ${SAMPLE_SIZE} -o ${OUTDIR} -p ${VARIABLE}_${SAMPLE_SIZE}_${LENGTH}h_${DT}h || exit 1

echo "Blocks have been stored in ${OUTDIR}; use these files to run S3M"
