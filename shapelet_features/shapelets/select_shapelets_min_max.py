#!/usr/bin/env python3
#
# select_shapelets_min_max.py: given a set of extracted shapelets,
# potentially separated into blocks, pools all of them and selects
# a number of shapelets based on maximizing the dissimilarity.
#
# More precisely, the accuracy induced by the split of each shapelet is
# calculated and all shapelets are reduced to the top 100 ones or fewer
# if necessary. Afterwards, starting from the most accurate shapelet, a
# new shapelet is determined that maximizes the distance to the already
# selected ones. This process is repeated until the specified number of
# shapelets have been selected.

import argparse
import collections
import json
import logging
import os
import re
import sys

import numpy as np


def read_data(filename):
    '''
    Opens a shapelet JSON file and returns its contents.
    '''
    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = {'shapelets': []}

    return data


def calculate_accuracy(a, b, d, c):
    '''
    Given a contingency table of the form

        a | b
        -----
        d | c

    this function calculates the accuracy of the induced split. The underlying
    assumption is that the classes are balanced.
    '''

    n = a + b + c + d
    x = (a + c) / n     # pattern
    y = (b + d) / n     # anti-pattern

    return max(x, y)


def distance(S, T):
    '''
    Calculates the minimum distance of two shapelets to each other.
    '''

    n = len(S)
    m = len(T)
    d = sys.float_info.max

    # Switch the sequences such that S always refers to the shorter
    # sequence.
    if n > m:
        n, m = m, n
        S, T = T, S

    S = np.array(S)
    T = np.array(T)

    for i in range(m - n + 1):
        U = T[i:i + n]
        distance = np.linalg.norm(S - U, ord=2)
        d = min(d, distance)

    return d


def min_distance(shapelets, S):
    '''
    Determines the minimum distance of a new shapelet to a set of
    shapelets.
    '''

    d = sys.float_info.max

    for shapelet_data in shapelets:
        d = min(d, distance(shapelet_data['shapelet'], S))

    return d


from scipy import stats


def linearity_check(shapelet):
    '''
    Is the given shapelet linear?
    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(shapelet)), shapelet)
    if r_value > 0.95:
        return True
    else:
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='Path to output directory', type=str, required=True)
    parser.add_argument('-n', '--n-shapelets', help='Number of shapelets to extract', type=int, default=10)
    parser.add_argument('-t', '--type', default='min-max', choices=['min-max', 'random', 'top'], help='Type of selection', type=str)
    parser.add_argument('FILES', nargs='+', type=str)

    args = parser.parse_args()
    output_directory = args.output
    n_shapelets = args.n_shapelets

    logging.basicConfig(level=logging.WARNING)
    logging = logging.getLogger(os.path.basename(__file__))

    # Pool of shapelets (regardless of length), indexed by the variable
    # name, e.g. `vm5`.
    shapelets_per_variable = collections.defaultdict(list)

    for filename in args.FILES:

        # Identify the variable to which this file belongs; we need this
        # for pooling shapelets correctly.
        basename = os.path.basename(filename)
        basename = os.path.splitext(basename)[0]

        re_var = r'(vm\d+)_.*'
        m = re.match(re_var, basename)

        if not m:
            print(re_var)
            print(basename)
            print(args.FILES)
            raise RuntimeError('Unable to extract variable from filename')

        logging.debug('Processing file {}...'.format(basename))

        variable = m.group(1)
        data = read_data(filename)
        shapelets = data['shapelets']
        shapelets_per_variable[variable].extend(shapelets)

    logging.debug('Detected the following variables {}'.format(sorted(shapelets_per_variable.keys())))

    for variable, shapelets in shapelets_per_variable.items():

        logging.debug('Processing shapelets for variable {}...'.format(variable))

        # Reset the number of shapelets for each variable in order to
        # ensure that we *always* get the maximum number of shapelets
        # that are available, regardless of whether certain variables
        # only have fewer shapelets available.
        n_shapelets = args.n_shapelets

        # Which shapelets are linear?
        indicator = []
        for shapelet_data in shapelets:
            indicator.append(linearity_check(shapelet_data['shapelet']))

        # Do not exclude linear shapelets (bc there are not enough)
        linearity_exclusion = False

        accuracies = []
        for shapelet_data in shapelets:
            a, b, d, c = shapelet_data['table']
            accuracy = calculate_accuracy(a, b, d, c)
            accuracies.append(accuracy)

        indices = np.argsort(accuracies)[::-1][:100]

        # Rather skip the variable entirely if no shapelets are
        # available at all.
        if len(indices) == 0:
            logging.warning('Skipping variable {} because no shapelets are available'.format(variable))
            continue

        # Report all shapelets that can be reported if an insufficient
        # number of them has been identified.
        if len(indices) < n_shapelets:
            logging.warning('Variable {} has only {} shapelets are available, but {} have been requested'.format(variable, len(indices), n_shapelets))
            logging.warning('Will report {} shapelets'.format(len(indices)))

            # Just pretend that the client did not specify as many
            # shapelets as they did.
            n_shapelets = len(indices)
        elif len(indices) - sum(indicator) > n_shapelets:
            linearity_exclusion = True

        shapelets = list(map(lambda i: shapelets[i], indices))

        if args.type == 'min-max':
            # The list `shapelets` is now already ordered according to the
            # accuracy, so we can just select the best shapelet here.
            selected_shapelets = [shapelets[0]]

            for i in range(n_shapelets - 1):
                furthest_distance = 0.0
                furthest_shapelet = None
                for shapelet_data, lin in zip(shapelets, indicator):
                    if lin == False and linearity_exclusion == True:
                        shapelet = shapelet_data['shapelet']
                        dist = min_distance(selected_shapelets, shapelet)

                        if dist > furthest_distance:
                            furthest_distance = dist
                            furthest_shapelet = shapelet_data

                # Only add the shapelet if it maximizes the distance to the
                # selected ones.
                if furthest_shapelet is not None:
                    selected_shapelets.append(furthest_shapelet)
        elif args.type == 'top':
            # The list `shapelets` is now already ordered according to the
            # accuracy, so we can just select the best shapelets here.
            selected_shapelets = shapelets[:n_shapelets]
        elif args.type == 'random':
            # Seed the Random State always the same!!!
            r = np.random.RandomState(42)
            selected_shapelets = r.choice(shapelets, n_shapelets)

        all_shapelets = {
            'shapelets': selected_shapelets
        }

        output_filename = os.path.join(output_directory, variable + '_min_max_shapelets.json')

        logging.info('Storing output in {}...'.format(output_filename))

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            json.dump(all_shapelets, f, indent=4)
