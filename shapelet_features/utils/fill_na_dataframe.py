import os
import logging
import gc
import numpy as np

import pickle


def load_pickle(fpath):
    ''' Given a file path pointing to a pickle file, yields the object pickled in this file'''
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', required=True, type=str)
    parser.add_argument('--target_dataset', required=True, type=str)
    parser.add_argument('--number', required=True, type=int)

    # pd.set_option('io.hdf.default_format', 'table')

    fpath = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/misc_derived/id_lists/v6b/patients_in_clean_chunking_50.pickle'
    bern_batch_map = load_pickle(fpath)["pid_to_chunk"]
    print(bern_batch_map)

    args = parser.parse_args()

    columns = pd.read_hdf(args.source_dataset, key='/74353').columns.values
    file = args.target_dataset
    files = [i for i in glob.glob(file + '*.h5') if ('_all_sh' not in i) and ('batch' not in i)]

    for file in [args.target_dataset]:#[files[args.number]]:
        if not os.path.exists(file[:-3] + '/'):
            os.makedirs(file[:-3] + '/')
        first = True
        ifile = pd.HDFStore(file, 'r')
        keys = ifile.keys()
        ifile.close()
        counter = 0
        for k in keys:
            counter = counter + 1
            if first:
                print(k)
                print(k, bern_batch_map[int(str(k)[1:])])
            ofile = file[:-3] + '/fixed_batch_' + str(bern_batch_map[int(str(k)[1:])]) + '.h5'
            df = pd.read_hdf(file, k)

            if first:
                print(k, df)
                first = False

            for column in columns:
                if column not in df.columns:
                    df[column] = np.nan

            df.fillna(10000000000000, inplace=True)

            if first:
                print(k, df)
            else:
                print(k)

            # df.to_hdf(ofile, k)
            df.to_hdf(ofile, "/p{}".format(k), format="fixed", append=False, complevel=5, complib="blosc:lz4")

            gc.collect()
