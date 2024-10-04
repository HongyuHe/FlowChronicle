import dataloader as dl
from dataloader import Dataset
import search
import baseline_slim_sqs as baseline
import dataloader as dl
import pandas as pd

if __name__ == "__main__":
    model_path = 'models/CIDDSxp3_001_our_3.pkl'
    path = 'data/CIDDS_xp3_train.csv'
    df = pd.read_csv(path)
    train = dl.preprocess_CIDDS_dataset(df, '2017-04-05 00:00:36.907', '2017-04-11 23:54:43.817')
    train_d = dl.discretize(train)

    dataset = Dataset(train_d.copy())

    model = search.search(dataset, load_checkpoint=0, load_path=model_path, load_candidates=True)
