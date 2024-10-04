import sys
sys.path.append("..")
from model import Model, ChunkyModel
from temporal_sampling import PatternSampler
from our_train_and_generate import parrallelize_flows
import argparse
import pandas as pd

if __name__ == "__main__":
    print("start load")
    parser=argparse.ArgumentParser()
    parser.add_argument("xp")
    parser.add_argument("name_of_model")
    args = parser.parse_args()
    name = str(args.name_of_model)
    xp = str(args.xp)
    
    m = Model.load_model(f"/home/aschoen/my_storage/aschoen/models/{xp}/{name}.pkl")
    print("finish load")
    c = m.cover
    cover_stats = c.get_cover_stats()
    patterns_usage = cover_stats.get_pattern_usage()

    c.fit_temporal_samplers(patterns_usage.keys())
    idx, synthetic_patterns = PatternSampler(patterns_usage).sample(sum(patterns_usage.values()),return_indices=True)
    print(len(synthetic_patterns))
    
    synthetic_df = parrallelize_flows(synthetic_patterns, c.dataset.col_name_map, c.dataset.column_value_dict, c.dataset.cont_repr, c.dataset.time_stamps, idxs=idx, cpus=45)
    synthetic_df.insert(0, 'Date first seen', synthetic_df.pop('Date first seen'))
    synthetic_df.to_csv(f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{xp}/{name}_load_syn.csv", index=False)
