
import dataloader as dl
from dataloader import Dataset
import search
import baseline_slim_sqs as baseline
from model import Model
from pattern import Pattern
from row_pattern import RowPattern
from attribute_value import AttributeValue, AttributeType
import dataloader as dl
import pickle
import pandas as pd

import argparse

if __name__ == "__main__":


    path = 'data/temp-fixip-debug.csv'
    df = pd.read_csv(path)

    train = dl.preprocess_CIDDS_dataset(df)
    print(train)
    train_d, dic = dl.discretize(train)
    print(train_d)

    dataset = Dataset(train_d.copy())

    dataset.save_model("output/temp-fixip-debug_sep_flags.dataset")
    exit()

    model = search.search(dataset)

    model.save_model("output/temp-fixip-debug_sep_flags.model")
    with open("output/temp-fixip-debug_sep_flags.dic", "wb") as f:
        pickle.dump(dic, f)
