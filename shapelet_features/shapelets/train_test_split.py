#!/usr/bin/env python
#
# train_test_split.py: given a split file and a directory containing
# `vm*` files (that correspond to individual variables),

import argparse
import csv
import glob
import logging

import numpy as np
import pandas as pd

from os import makedirs
from os.path import join, dirname, basename, exists

from utils.cli import existing_file, nonexistant_file


logging = logging.getLogger(basename(__file__))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create train/test split from CSV files")
    # FIXME: check overwriting policy elsewhere, because only subsequent
    # calls will add directories
    parser.add_argument('-d', '--dir', required=True)
    parser.add_argument('-s', '--split', required=True, type=existing_file)
    parser.add_argument('-k', '--key')

    args = parser.parse_args()
    logging.info('Reading variables from directory {}'.format(args.dir))
    splitpath = args.split

    key = args.key
    if not key:
        for var_file in glob.glob(join(args.dir, 'vm*.csv')):
            logging.debug('Processing file {}...'.format(var_file))
            path_name = join(dirname(var_file), 'train')
            if not exists(path_name):
                makedirs(path_name)
            if "static" not in args.dir:
                var_df = pd.read_csv(var_file, header=None)
                var_df.rename(columns={0: 'PatientID'}, inplace=True)
                var_df.to_csv(join(path_name, basename(var_file)), header=False, index=False)
    else:
        # Load train split patient_ids
        if 'pickle' not in splitpath:
            full_split_df = pd.read_csv(splitpath, sep='\t')
            split_df = full_split_df[['pid', key]]
            for split in split_df[key].unique():
                if split != '-':
                    logging.info('Processing "{}" split'.format(split))
                    split_ids = split_df[split_df[key] == split]['pid'].values
                    #split_ids = split_df.query('@key == @split')['pid'].values
                    for var_file in glob.glob(join(args.dir, 'vm*.csv')):

                        logging.debug('Processing file {}...'.format(var_file))

                        path_name = join(dirname(var_file), split)
                        if not exists(path_name):
                            makedirs(path_name)
                        if "static" not in args.dir:
                            print(var_file)
                            # Old version only if file is exact
                            # var_df = pd.read_csv(var_file, header=None)
                            # More flexible for datetime incositencies
                            var_df = pd.read_csv(var_file, header=None, sep='\n')
                            var_df = var_df[0].str.split(',', expand=True)
                            var_df = var_df.dropna(axis='columns', how='any')
                            var_df.rename(columns={0: 'PatientID'}, inplace=True)
                            var_df['PatientID'] = pd.to_numeric(var_df['PatientID'], downcast='integer', errors='coerce')
                            var_df[1] = pd.to_datetime(var_df[1])
                            var_df[2] = pd.to_datetime(var_df[2])
                            for k in range(3, len(var_df.columns)):
                                var_df[k] = pd.to_numeric(var_df[k], downcast='float', errors='coerce')
                            # print(var_df.dtypes)
                            # print(type(split_ids[0]))
                            # print(var_df['PatientID'].values.tolist())
                            # print(list(split_ids))
                            # print('Intersection:', list(set(list(split_ids)) & set(var_df['PatientID'].values.tolist())))
                            df = var_df.query('PatientID in @split_ids')
                            df.to_csv(join(path_name, basename(var_file)), header=False, index=False)
                            print(df)
                        elif "static" in args.dir:
                            print("Reading static files")
                            with open(join(path_name, basename(var_file)), 'w') as res:
                                csv_writer = csv.writer(res, delimiter=',')
                                with open(var_file, 'r') as f:
                                    csv_reader = csv.reader(f)
                                    for row in csv_reader:
                                        if np.any(int(row[0]) == split_ids):
                                            csv_writer.writerow(row)
        else:
            import pickle
            file = open(splitpath, 'rb')
            split_df = pd.DataFrame(pickle.load(file))
            split_df = split_df[key]
            for split in split_df.keys():
                if split != '-':
                    logging.info('Processing "{}" split'.format(split))
                    split_ids = split_df[split]
                    for var_file in glob.glob(join(args.dir, 'vm*.csv')):

                        logging.debug('Processing file {}...'.format(var_file))

                        path_name = join(dirname(var_file), split)
                        if not exists(path_name):
                            makedirs(path_name)
                        if "static" not in args.dir:
                            var_df = pd.read_csv(var_file, header=None)
                            var_df.rename(columns={0: 'PatientID'}, inplace=True)
                            df = var_df.query('PatientID in @split_ids')
                            df.to_csv(join(path_name, basename(var_file)), header=False, index=False)
                        elif "static" in args.dir:
                            print("Reading static files")
                            with open(join(path_name, basename(var_file)), 'w') as res:
                                csv_writer = csv.writer(res, delimiter=',')
                                with open(var_file, 'r') as f:
                                    csv_reader = csv.reader(f)
                                    for row in csv_reader:
                                        if np.any(int(row[0]) == split_ids):
                                            csv_writer.writerow(row)
