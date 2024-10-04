#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 13:27:14 2023

@author: aschoen
"""

import argparse
import time
import pandas as pd
import numpy as np

from metrics import compute_JSD, compute_EMD, compute_PCD, compute_density_coverage, compute_authenticity, compute_CMD, compute_DKC

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('train', type=str, help="Path to the Training Dataframe (*.csv)")
    parser.add_argument('test', type=str, help="Path to the Testing Dataframe (*.csv)")
    parser.add_argument('generated', type=str, help="Path to the Generated (Synthetic) Dataframe (*.csv)")
    parser.add_argument('metric', type=str, help="Metrics to use")
    parser.add_argument('n_test', type=int, default=3, help="Number of times to compute the metric")

    parser.add_argument('--n_samples', type=int, default=None ,required=False, help="Number of samples used to compute")
    parser.add_argument('--confidence_range', type=bool, default=False ,required=False, help="If true, return the range of confidence")
    parser.add_argument('--time_elapsed', type=bool, default=False ,required=False, help="If true, return the duration of the computation")

    parser.add_argument('--number_of_categorical_value', type=int, default=50 ,required=False, help="Minimal numnber of distinct value for numerical features")
    parser.add_argument('--n_authenticity', type=int, default=500 ,required=False, help="Number of sample to consider in the authenticity test")
    parser.add_argument('--lag', type=int, default=1 ,required=False, help="The lag to consider ")
    parser.add_argument('--alpha', type=float, default=.05 ,required=False, help="Value for the range of confidence")

    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    generated = pd.read_csv(args.test)

    start_time = time.time()
    result = []

    for i in range(parser.n_test):
        tr = train.sample(args.n_samples)
        ts = test.sample(args.n_samples)
        g = generated.sample(args.n_samples)
        if args.metric == 'JSD':
            result.append(compute_JSD(ts, g))
        elif args.metrics == 'EMD':
            result.append(compute_EMD(ts, g))
        elif args.metrics == 'PCD':
            result.append(compute_PCD(ts, g))
        elif args.metrics == 'Density':
            result.append(compute_density_coverage(ts, g, args.number_of_categorical_value)[0])
        elif args.metrics == 'Coverage':
            result.append(compute_density_coverage(ts, g, args.number_of_categorical_value)[1])
        elif args.metrics == 'CMD':
            result.append(compute_CMD(ts, g))
        elif args.metrics == 'Novelty':
            result.append(compute_authenticity(tr, ts, g, args.number_of_categorical_value, args.n_authenticity))
        elif args.metrics == "DKC":
            result.append(compute_DKC(g))
    result = np.asarray(result)

    end_time = time.time()

    if args.confidence_range:
        print(np.percentile(result,5), np.median(result), np.percentile(result,5))
    else:
        print(np.median(result))
    if args.time_elapsed:
        print(" Time elapsed : ", end_time-start_time)