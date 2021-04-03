#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd

from unsupervised_learner import process_data, run_experiment_1

def main(args):
    dataset = 'eye' if args.eye else 'diabetes'
    data = pd.read_csv(f'data/{dataset}.csv')
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    x_train, x_test, y_train, y_test = process_data(x, y)

    if args.exp1:
        run_experiment_1(x_train, dataset, args.output)
    print(f'\nCompleted final arguments: {args}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute Unsupervised Learning experiments')
    parser.add_argument('--eye', dest='eye', action='store_true', help="run on eye dataset, else will run on diabetes")
    parser.add_argument('--output',type=str,help='output directory, default value is "output"')
    parser.add_argument('--exp1', dest='exp1', action='store_true', help="run experiment 1, default True")
    parser.set_defaults(eye=False, output='output', exp1=False)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)
