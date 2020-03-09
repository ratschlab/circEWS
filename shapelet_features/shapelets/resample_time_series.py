#!/usr/bin/env python3
#
# resample_time_series.py: given a time series in CSV format, resamples
# it to another time interval. Currently, the resampling is done in the
# most trivial way that is imaginable, viz. by *skipping* a pre-defined
# number of points in the time series.

import argparse
import csv
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', type=str, help='Input files')
    parser.add_argument('-s', '--skip', default=12, type=int, help='Number of values to skip')
    parser.add_argument('-o', '--output-directory', default='~/tmp', help='Output directory of re-sampled data')

    args = parser.parse_args()
    output_directory = args.output_directory
    output_directory = os.path.expanduser(output_directory)

    # Defines the number of columns to skip in each row. This can be
    # used to skip fields such as 'PatientID', time stamps etc.
    n_columns_skip = 4

    for filename in args.FILES:
        basename = os.path.basename(filename)
        basename = os.path.splitext(basename)[0]
        basename += '_resampled.csv'

        with open(filename) as f, open(os.path.join(output_directory, basename), 'w') as g:
            reader = csv.reader(f)
            writer = csv.writer(g)
            for row in reader:
                prefix = row[:n_columns_skip]
                values = row[n_columns_skip::args.skip]

                # Only write the row if values could be successfully
                # extracted from it.
                if values:
                    writer.writerow(prefix+values)
