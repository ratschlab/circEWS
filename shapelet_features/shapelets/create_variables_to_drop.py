#!/usr/bin/env python3
#
# create_variables_to_drop.py: uses a list of selected variables and
# a directory of shapelets to determine which variables to *drop* in
# a classification scenario (because no shapelets could be extracted
# from them).

import argparse
import csv
import glob
import json
import logging
import re
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('VARIABLES', help='CSV file containing all selected variables')
    parser.add_argument('SHAPELETS_DIR', help='Directory containing extracted shapelets')

    args = parser.parse_args()
    variables = []

    logging.basicConfig(level=logging.INFO)

    with open(args.VARIABLES) as f:
        reader = csv.reader(f)
        for row in reader:
            variables.append(row[0])

    variables = set(variables)

    shapelets_files = [f for f in sorted(glob.glob(os.path.join(args.SHAPELETS_DIR, '*.json')))]
    re_var = r'(vm\d+)_.*'
    variables_with_shapelets = set()

    for shapelets_file in shapelets_files:
        basename = os.path.basename(shapelets_file)
        basename = os.path.splitext(basename)[0]
        m = re.match(re_var, basename)

        if not m:
            raise RuntimeError('Unable to extract variable from filename {}'.format(shapelets_file))

        variable = m.group(1)
        if variable not in variables:
            raise RuntimeError('Variable {} is not part of the selected variables'.format(variable))

        logging.info('Processing {}...'.format(shapelets_file))
        with open(shapelets_file) as f:
            data = json.load(f) 
            n_shapelets = len(data['shapelets'])
            if n_shapelets > 0:
                variables_with_shapelets.add(variable)

    variables_to_keep = variables_with_shapelets.intersection(variables)
    variables_to_drop = variables - variables_with_shapelets

    logging.info('Variables to keep: {}'.format(' '.join(variables_to_keep)))
    logging.info('Variables to drop: {}'.format(' '.join(variables_to_drop)))
