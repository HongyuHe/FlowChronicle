import sys
sys.path.append("..")
import dataloader as dl
import search
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('n_split')
    args = parser.parse_args()
    n_split = int(args.n_split)
    split = n_split > 0
    
    # Disable all logging and warnings (might disturb the mutithreading)
    #logging.disable(logging.CRITICAL + 1)
    #warnings.filterwarnings("ignore")

    path = f"/home/aschoen/my_storage/aschoen/dataset/flow_chronicle_dataset/{args.experiment}/"

    print(path)

    train_dataset = dl.Dataset.load_model(path+f"preprocessed/CIDDS_{args.experiment}_train_{n_split}.pkl")

    m = search.search(train_dataset, load_checkpoint=0, model_name=f'CIDDS_{args.experiment}_our')

    m.save_model(f"/home/aschoen/my_storage/aschoen/models/{args.experiment}/CIDDS_{args.experiment}_split{n_split}-our.pkl")

