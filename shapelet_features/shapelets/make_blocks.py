#!/usr/bin/env python3
#
# make_blocks.py: given an input file with cases and controls (in CSV
# format), creates as many blocks of a given size as possible. All of
# them will be stored in an output directory and indexed.

import argparse
import logging
import math
import os
import random
import sys


def zip_discard(*iterables):

    from functools import partial
    from itertools import zip_longest
    from operator import is_not

    return map(
        partial(filter, partial(is_not, None)),
        zip_longest(*iterables, fillvalue=None)
    )


def block_iter(iterable, n, allow_partial=False):
    '''
    Partitions an iterable object into a set of blocks of equal length,
    while only returning full blocks, i.e. blocks that have exactly the
    specified number of entries.

    If ``allow_partial`` is set, partially-filled blocks will be stored
    as well.
    '''

    iters = [iter(iterable)] * n
    if not allow_partial:
        return zip(*iters)
    else:
        return zip_discard(*iters)


def read_file(filename):
    '''
    Reads all lines in a file and returns them as a list. This is just
    an auxiliary function.
    '''

    lines = []
    with open(filename) as f:
        lines = f.readlines()

    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('CASES', help='Data for cases (CSV format)')
    parser.add_argument('CONTROLS', help='Data for controls (CSV format)')
    parser.add_argument('BLOCKSIZE', type=int, help='Block size (how many rows to use from cases and controls)')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Filename prefix')
    parser.add_argument('-o', '--output-directory', type=str, required=True, help='Path to output directory')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ####################################################################
    # Read data
    ####################################################################

    cases_filename = args.CASES
    controls_filename = args.CONTROLS
    blocksize = args.BLOCKSIZE

    logging.info('Reading data')
    logging.info('Input file for cases: {}'.format(cases_filename))
    logging.info('Input file for controls: {}'.format(controls_filename))

    all_cases = read_file(cases_filename)
    all_controls = read_file(controls_filename)

    ####################################################################
    # Shuffle
    ####################################################################
    #
    # Since cases and controls have already been separated prior to
    # calling this script, we do not have to worry about the labels
    # here.

    logging.info('Shuffling data')

    random.shuffle(all_cases)
    random.shuffle(all_controls)

    ####################################################################
    # Make blocks
    ####################################################################
    #
    # We check below how many *full* blocks we can obtain.

    blocks_cases = list(block_iter(all_cases, blocksize))
    blocks_controls = list(block_iter(all_controls, blocksize))
    num_blocks = min(len(blocks_cases), len(blocks_controls))

    logging.info('Obtained {} full blocks of size'.format(num_blocks, blocksize))

    # Default to make one block with *some* cases and *some* controls
    # that may not be balanced.
    if num_blocks == 0:
        logging.warning('Using partially-filled blocks because no full blocks are available')

        blocks_cases = list(block_iter(all_cases, blocksize, allow_partial=True))
        blocks_controls = list(block_iter(all_controls, blocksize, allow_partial=True))
        num_blocks = min(len(blocks_cases), len(blocks_controls))

        if num_blocks == 0:
            logging.warning('No blocks are available. Quitting.')
            sys.exit(0)

    num_digits = math.ceil(math.log10(num_blocks))
    cases_basename = os.path.splitext(os.path.basename(cases_filename))[0]
    controls_basename = os.path.splitext(os.path.basename(controls_filename))[0]
    fs = ':0{}d'.format(num_digits)

    for index, (cases_block, controls_block) in enumerate(zip(blocks_cases, blocks_controls)):
        filename = os.path.join(args.output_directory, (args.prefix + '_block_{' + fs + '}' + '_S3M.csv').format(index))

        logging.info('Storing data in {}'.format(filename))

        with open(filename, 'w') as f:
            f.writelines(cases_block)
            f.writelines(controls_block)
