import dataloader as dl
from dataloader import Dataset
import search
import baseline_slim_sqs as baseline
import dataloader as dl
import pandas as pd
import time

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-method', type=str, required=True, help="Method to use for search, either 'sqs-based' or 'our'")
    parser.add_argument('-iteration', type=int, required=False, default=0)

    args = parser.parse_args()

    # path = 'data/CIDDS_001_train_small.csv'
    path = 'data/CIDDS_xp2_train.csv'
    preprocessing_start_time = time.time()
    dataset = dl.load_CIDDS_dataset(path)

    preprocessing_time = time.time() - preprocessing_start_time


    training_start_time = time.time()
    if args.method == 'sqs-based':
        m_search = baseline.search(dataset)
    elif args.method == 'our':
        m_search = search.search(dataset, load_checkpoint=args.iteration, model_name='CIDDSxp2-our-0')
    training_time = time.time() - training_start_time

    length_model= m_search.get_model_length()
    length_data = m_search.cover.compute_data_length()
    print(length_model, '+',  length_data, '=',length_model+length_data)
    cs = m_search.cover.get_cover_stats()
    for p,u in cs.get_pattern_usage().items():
        print(p,u)

    # Save model
    m_search.save_model(args.o)

    runtime_file = args.o.replace('.pkl', '_runtime.txt')
    with open(runtime_file, 'w+') as f:
        f.write(f"Preprocessing Time: {preprocessing_time} seconds\n")
        f.write(f"Training Time: {training_time} seconds\n")
