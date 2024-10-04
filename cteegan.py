import argparse
import pandas as pd
import numpy as np
from scipy import stats
import time
#import sdv
import sdv.single_table as st
import sdv.metadata as md

parser = argparse.ArgumentParser(description="Use SD library to generate synthetic data")
parser.add_argument("model", metavar="M", type=str,help="The model to use")
parser.add_argument("input_path", metavar ="i", type=str, default="data/train.csv")
parser.add_argument("output_path", metavar="o", type=str, default="data/")

args = parser.parse_args()
print(args.experience)
print(args.model)

path = args.input_path
name=args.output_path+args.model+"_syn.csv"
print(path)

t0 = time.time()

train = pd.read_csv(path)
#train = pd.read_csv(path)
#test_b = pd.read_csv('/srv/tempdd/aschoen/dataset/ugr16_test_b.csv')

def preprocess_CIDDS(u):
    df = u.copy()
    df["Date first seen"] = pd.to_datetime(df["Date first seen"])
    df[["Dst Pt","Duration"]] = df[["Dst Pt","Duration"]].astype(float)
    df[["In Byte", "Out Byte", "In Packet", "Out Packet"]] = df[["In Byte","Out Byte", "In Packet", "Out Packet"]].astype(float).astype(int)
    return df

train = preprocess_CIDDS(train)

for c in train.columns:
    if train[c].nunique()<50 or 'IP' in c or 'Pt' in c:
        train[c]=train[c].astype(str)

#Define metadata
metadata=md.SingleTableMetadata()
#Detect metadata
metadata.detect_from_dataframe(train)
#Define model
if args.model == 'ctgan':
    model = st.CTGANSynthesizer(metadata, enforce_rounding=False, epochs=300, verbose=True)
if args.model == 'tvae':
    model = st.TVAESynthesizer(metadata, enforce_rounding=False, epochs=300)

t1 = time.time()
dif1 = divmod(int(t1-t0),3600)
print("Preprocessing time: {} hours, {} minutes and {} seconds".format(dif1[0], *divmod(dif1[1],60)))

print(metadata.to_dict())
#Train model
model.fit(train)
t2 = time.time()
dif2 = divmod(t2-t1,3600)
print("Training time: {} hours, {} minutes and {} seconds".format(dif2[0],*divmod(dif2[1],60)))
#Generate new data
result = model.sample(num_rows=len(train))
if args.model == 'ctgan':
    result.to_csv(name, index=None)
if args.model == 'tvae':
    result.to_csv(name, index=None)
t3 = time.time()
dif3 = divmod(t3-t2,3600)
print("Sampling time: {} hours, {} minutes and {} seconds".format(dif3[0],*divmod(dif3[1],60)))
