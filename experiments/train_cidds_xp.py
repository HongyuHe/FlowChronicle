
import time
import argparse
import pandas as pd

import search
import dataloader as dl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, required=True)

    args = parser.parse_args()

    if args.e > 3:
        raise Exception("Invalid experiment number")

    path = 'data/xp{}/CIDDS_xp{}_train.csv'.format(args.e, args.e)
    preprocessing_start_time = time.time()
    dataset = dl.load_CIDDS_dataset(path)
    preprocessing_time = time.time() - preprocessing_start_time

    training_start_time = time.time()
    m_search = search.search(dataset, model_name='CIDDSxp{}-our'.format(args.e))
    training_time = time.time() - training_start_time

    length_model= m_search.get_model_length()
    length_data = m_search.cover.compute_data_length()
    print(length_model, '+',  length_data, '=',length_model+length_data)
    cs = m_search.cover.get_cover_stats()
    for p,u in cs.get_pattern_usage().items():
        print(p,u)

    # Save model

    output_path = 'output/cidds_xps/CIDDS_xp{}-our.pkl'.format(args.e)

    m_search.save_model(output_path)

    runtime_file = output_path.replace('.pkl', '_runtime.txt')
    with open(runtime_file, 'w+') as f:
        f.write(f"Preprocessing Time: {preprocessing_time} seconds\n")
        f.write(f"Training Time: {training_time} seconds\n")
