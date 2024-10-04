
import dataloader as dl
from dataloader import Dataset
import search
import baseline_slim_sqs as baseline
from model import Model
from pattern import Pattern
from row_pattern import RowPattern
from attribute_value import AttributeValue, AttributeType
import dataloader as dl

import pandas as pd

import argparse

if __name__ == "__main__":
    '''
    TODO add dataset as argument and change name of script
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-save-model-path', type=str, required=True)
    parser.add_argument('-method', type=str, required=True, help="Method to use for search, either 'sqs-based' or 'our'")

    args = parser.parse_args()

    path = 'data/test_bed_data/socbed-bi.txt'
    df = pd.read_csv(path)

    train = dl.preprocess_socbed_bi(df, '2023-07-27 19:40:12', '2023-07-27 21:41:00')
    train_d = dl.discretize(train)

    dataset = Dataset(train_d.copy())
    if args.method == 'sqs-based':
        model = baseline.search(dataset)
    elif args.method == 'our':
        model = search.search(dataset)
    else:
        raise Exception('method not supported')

    model.save_model(args.save_model_path)
