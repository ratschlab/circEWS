#!/usr/bin/env python3
#
# shapelet_provenance.py: given two files, one JSON file containing
# significant shapelets and one CSV file containing time series, we
# check the 'provenance' of a shapelet, i.e. which type of patient,
# case or control, it originates from.

import csv
import json
import sys


def calculate_accuracy(a, b, d, c):
    '''
    Given a contingency table of the form

        a | b
        -----
        d | c

    this function calculates the accuracy of the induced split. The underlying
    assumption is that the classes are balanced.

    TODO: shamelessly copied from shapelet analysis script :-) If this
    happens multiple times, we should think about a `util` module.
    '''

    n = a + b + c + d
    x = (a + c) / n     # pattern
    y = (b + d) / n     # anti-pattern

    return max(x, y)


if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        data = json.load(f)

    assert data
    shapelets = data['shapelets']

    labels = []
    with open(sys.argv[2]) as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[3]
            labels.append(label)

    selected_shapelets = []
    for shapelet_data in shapelets:
        index = shapelet_data['index']
        label = labels[index]

        if label == '1':
            accuracy = calculate_accuracy(*shapelet_data['table'])
            shapelet = shapelet_data['shapelet']
            index = shapelet_data['index']
            selected_shapelets.append({
                'accuracy': accuracy,
                'shapelet': shapelet,
                'index': index
            })

    selected_shapelets = sorted(selected_shapelets, key=lambda x: x['accuracy'], reverse=True)
    print(json.dumps(selected_shapelets, indent=4))
