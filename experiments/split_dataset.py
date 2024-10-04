import sys
sys.path.append("..")
import dataloader as dl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('n_split')
    args = parser.parse_args()
    n_split = int(args.n_split)

    path = f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{args.experiment}/"

    train_dataset = dl.load_CIDDS_dataset(path+f"CIDDS_{args.experiment}_train.csv")

    train_dataset, dataset_chunked = dl.load_CIDDS_splitted_dataset(path+f"CIDDS_{args.experiment}_train.csv", n_split)

    for i,data in enumerate(dataset_chunked):
        data.save_model(path+f"preprocessed/CIDDS_{args.experiment}_train_{i}.pkl")

    print("splitting done")
