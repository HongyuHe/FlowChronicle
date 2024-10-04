import sys
import pickle
sys.path.append("..")
import os
from model import Model, ChunkyModel
from dataloader import load_CIDDS_dataset
from temporal_sampling import PatternSampler
from our_train_and_generate import parrallelize_flows
import argparse
import pandas as pd

if __name__ == "__main__":
    print("start load")
    parser=argparse.ArgumentParser()
    parser.add_argument("xp")
    args = parser.parse_args()
    xp = str(args.xp)
    name = f"CIDDS_{xp}_split"
    name = f"our_split350"
    
    path = f"/home/aschoen/my_storage/aschoen/models/{xp}/"
    file_list = [os.path.join(path,f) for f in os.listdir(path) if name in f]
    model_list = [Model.load_model(os.path.join(path,f)) for f in file_list]
    
    train_dataset = load_CIDDS_dataset(f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{xp}/CIDDS_{xp}_train.csv")

    m = ChunkyModel(train_dataset, model_list)
    m.save_model(path+f"our_syn_{xp}.pkl")
    #m = Model.load_model("/home/aschoen/my_storage/aschoen/models/")
    print("finish load")
    c = m.cover
    cover_stats = c.get_cover_stats()
    patterns_usage = cover_stats.get_pattern_usage()

    c.fit_temporal_samplers(patterns_usage.keys())
    idx, syntetic_patterns=PatternSampler(patterns_usage).sample(sum(patterns_usage.values()), return_indices=True)
    with open(f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{xp}/preprocessed/patt_list.pkl","wb") as file: pickle.dump(syntetic_patterns, file)
    with open(f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{xp}/preprocessed/idx_list.pkl","wb") as file: pickle.dump(idx, file)
    
    metadata = {"col_name_map":train_dataset.col_name_map, "column_value_dict":train_dataset.column_value_dict, "cont_repr":train_dataset.cont_repr, "time_stamps":train_dataset.time_stamps}
    with open(f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{xp}/preprocessed/metadata.pkl","wb") as file: pickle.dump(metadata, file)

    print("done writing the lists")
