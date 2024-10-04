import sys
import pickle
sys.path.append("..")
import os
from dataloader import load_CIDDS_dataset
from our_train_and_generate import get_flows
import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    print("start load")
    parser=argparse.ArgumentParser()
    parser.add_argument("xp")
    parser.add_argument("n_split")
    parser.add_argument("split")
    args = parser.parse_args()
    xp = str(args.xp)
    n_split = int(args.n_split)
    split = int(args.split)
    
    path = f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{xp}/preprocessed/"
    with open(path+"patt_list.pkl","rb") as file : patts=pickle.load(file)
    with open(path+"idx_list.pkl","rb") as file : ids=pickle.load(file)
    with open(path+"metadata.pkl","rb") as file : metadata=pickle.load(file)
    
    patts = np.array_split(np.asarray(patts), n_split)
    ids = np.array_split(np.asarray(ids), n_split)
    patt = patts[split]
    idx = ids[split]
    
    del patts
    del ids
    
    df = get_flows(patt, metadata["col_name_map"], metadata["column_value_dict"], metadata["cont_repr"], metadata["time_stamps"], idxs = idx)
    df.to_csv(path+f"our_split_{split}.csv",index=False)
