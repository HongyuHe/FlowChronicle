import pandas as pd
import dataloader as dl

from dataloader import Dataset

import search as search
from preprocess import bytes_to_int


if __name__ == "__main__":
    path = 'data/quicktest_unidirectional.csv'
    df = pd.read_csv(path, low_memory=False)
    train = dl.common_preprocess(df, '2017-04-05 00:00:00.266', '2017-04-05 01:00:00.266')
    train = train.rename(columns={'Packets':'In Byte', 'Bytes':'Out Byte'})
    train['Out Byte'] = bytes_to_int(train['Out Byte'])

    print(train.columns)
    train_d = dl.load_socbed_bi(train.copy(), eva = True)
    train_d = train_d.sort_values(by=['Date first seen'])
    dataset = Dataset(train_d.copy())

    model = search.search(dataset)
    model.save_model('output/quicktest_unidirectional.model')
