#!/usr/bin/env python3
#
# analyse_extracted_shapelets.py: analyses a family of JSON files that
# contain extracted shapelets by calculating the accuracy induced by a
# contingency table. Various statistics such as ranks will be shown as
# well.

import collections
import json
import os
import re
import sys

import numpy as np

def read_data(filename):
    '''
    Opens a shapelet JSON file and returns its contents.
    '''

    with open(filename) as f:
        data = json.load(f)

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

if __name__ == '__main__':

    # Collated accuracies for each variable. The key is a variable
    # identifier such as `vm5`, while the value is dictionary that
    # contains all accuracies over all blocks. Each entry stores a
    # length.
    collated_accuracies = collections.defaultdict(lambda: collections.defaultdict(list))

    # Stores the top $k$ shapelets (with respect to accuracy) per
    # variable.
    top_shapelets_per_variable = collections.defaultdict(list)
    k = 3

    for filename in sys.argv[1:]:
        # Identify the variable to which this file belongs plus the
        # length of the extracted shapelets. This is required to do
        # a proper store later on.
        basename = os.path.basename(filename)
        basename = os.path.splitext(basename)[0]

        re_var = r'(vm\d+)_.*_(m\d+).*'
        m = re.match(re_var, basename)

        if not m:
            raise RuntimeError('Unable to extract variable and length information from filename')

        variable = m.group(1)
        length = m.group(2)

        data = read_data(filename)
        shapelets = data['shapelets']

        accuracies_per_file = []

        for shapelet_data in shapelets:
            a, b, d, c = shapelet_data['table']
            accuracy = calculate_accuracy(a, b, d, c)
            accuracies_per_file.append(accuracy)
            top_shapelets_per_variable[variable].append((accuracy, shapelet_data['shapelet']))

        # Extend the list of all accuracies per block for the given
        # variable; note that the same key can be accessed multiple
        # times because we do not know how many blocks there are.
        collated_accuracies[variable][length].extend(accuracies_per_file)

        if not accuracies_per_file:
            accuracies_per_file = [np.nan]

    # Keep only the top shapelets of each variable. This could be solved
    # more efficiently, but it is easier to do it like this.
    for variable in sorted(top_shapelets_per_variable.keys()):
        shapelets = top_shapelets_per_variable[variable]
        shapelets = sorted(shapelets, key=lambda x: x[0], reverse=True)
        shapelets = shapelets[:k]

        top_shapelets_per_variable[variable] = shapelets

    # Store top shapelets
    for variable in sorted(top_shapelets_per_variable.keys()):
        with open('/tmp/{}_top_{}.json'.format(variable, k), 'w') as f:
            shapelet_data = top_shapelets_per_variable[variable]
            data = list({'accuracy': accuracy, 'shapelet': shapelet} for accuracy, shapelet in shapelet_data)
            json.dump(data, f, indent=4)

    # Stores maximum achieved accuracy per variable, regardless of the
    # shapelet lengths. This is used to obtain hand-crafted features.
    max_accuracy_per_variable = dict()

    # Ditto for mean accuracy.
    max_mean_accuracy_per_variable = dict()

    for variable in collated_accuracies:
        print('-' * 72)
        print(variable)

        # Maximum accuracy of the variable, regardless of shapelet
        # length or anything else.
        max_accuracy = 0.0

        # Ditto for mean accuracy.
        max_mean_accuracy = 0.0

        for length in collated_accuracies[variable]:
            print(' {}'.format(length))

            accuracies = collated_accuracies[variable][length]

            if not accuracies:
                print('  No data')
            else:
                print('  Minimum accuracy: {:0.2f}'.format(np.min(accuracies)))
                print('  Maximum accuracy: {:0.2f}'.format(np.max(accuracies)))
                print('  Mean accuracy   : {:0.2f} +- {:0.2f}'.format(np.mean(accuracies), 2 * np.std(accuracies)))

                max_accuracy = max(max_accuracy, np.max(accuracies))
                max_mean_accuracy = max(max_mean_accuracy, np.mean(accuracies))

        max_accuracy_per_variable[variable] = max_accuracy
        max_mean_accuracy_per_variable[variable] = max_mean_accuracy


    print('-' * 72)
    print('Variables sorted by maximum accuracy:')
    for variable, max_accuracy in sorted(max_accuracy_per_variable.items(), key=lambda x:x[1], reverse=True):
        print('{:8} ({:0.2f})'.format(variable, max_accuracy))

    print('-' * 72)
    print('Variables sorted by maximum mean accuracy:')
    for variable, max_mean_accuracy in sorted(max_mean_accuracy_per_variable.items(), key=lambda x:x[1], reverse=True):
        print('{:8} ({:0.2f})'.format(variable, max_mean_accuracy))

