import dataloader as dl
from dataloader import Dataset
import search
import baseline_slim_sqs as baseline
import pandas as pd

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-method', type=str, required=True, help="Method to use for search, either 'sqs-based' or 'our'")
    parser.add_argument('-iteration', type=int, required=False, default=0)

    args = parser.parse_args()

    # path = 'data/CIDDS_001_train_small.csv'
    path = 'data/CIDDS_001_train.csv'
    # df = pd.read_csv(path)
    # train = dl.preprocess_CIDDS_dataset(df)
    # train_d, discrete_dic = dl.discretize(train)

    # dataset = Dataset(train_d.copy())
    # dataset.cont_repr = get_CIDDS_cont_repr(train_d, discrete_dic)
    train_dataset = dl.load_CIDDS_dataset(path)

    if args.method == 'sqs-based':
        m_search = baseline.search(train_dataset)
    elif args.method == 'our':
        m_search = search.search(train_dataset, load_checkpoint=args.iteration, model_name='CIDDS_001_our')

    length_model= m_search.get_model_length()
    length_data = m_search.cover.compute_data_length()
    print(length_model, '+',  length_data, '=',length_model+length_data)
    cs = m_search.cover.get_cover_stats()
    for p,u in cs.get_pattern_usage().items():
        print(p,u)

    # Save model
    m_search.save_model(args.o)