import dataloader as dl
from dataloader import Dataset
import search
import baseline_slim_sqs as baseline
import dataloader as dl
import pandas as pd

import argparse


if __name__ == "__main__": #TODO have something like this for all datasets etc.
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-iteration', type=int, required=False, default=0)

    args = parser.parse_args()

    path = 'data/test_bed_data/socbed-bi.txt'
    df = pd.read_csv(path)

    train = dl.common_preprocess(df, '2023-07-27 19:40:12', '2023-07-27 21:41:00')
    # train['Out Byte']=train['Out Byte'].str.replace(' M','e6').astype(float)
    # train['In Byte']=train['In Byte'].str.replace(' M','e6').astype(float)

    train_d = dl.load_socbed_bi(train.copy(), eva = False)


    dataset = Dataset(train_d.copy())

    m_search = search.search(dataset, load_checkpoint=args.iteration)
    # m_search = baseline.search(dataset)
    length_model= m_search.get_model_length()
    length_data = m_search.cover.compute_data_length()
    print(length_model, '+',  length_data, '=',length_model+length_data)
    cs = m_search.cover.get_cover_stats()
    for p,u in cs.get_pattern_usage().items():
        print(p,u)
