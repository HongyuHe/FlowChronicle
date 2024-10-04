import os
import argparse
import pandas as pd

if __name__ == "__main__":
    print("start load")
    parser=argparse.ArgumentParser()
    parser.add_argument("xp")
    args = parser.parse_args()
    xp = str(args.xp)
    
    path = f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{xp}/"
    file_list = [f for f in os.listdir(path+"preprocessed/") if ".csv" in f]
    csvs = [pd.read_csv(file)for file in file_list]
    pd.concat(csvs).to_csv(path+"our_load_multiple_syn.csv")
    print("done sampling")
