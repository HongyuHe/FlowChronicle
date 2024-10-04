#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:03:38 2024

@author: anonymous
"""
import sys
sys.path.append("..")
import argparse
import time
import torch
import pandas as pd
from preprocess_transformer import preprocess, postprocess, IntegerDataset
from transformer_model import TransformerModel

t0 = time.time()

path = "../data/"

orig_dataset = preprocess(path + "train.csv")

context_size = 60 #TODO : tackle the situation where context_size>len(training_sequence)

dataset = IntegerDataset(orig_dataset,context_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = TransformerModel(dataset, device)

t1 = time.time()
dif1 = divmod(t1-t0,3600)
print("Preprocessing time: {} hours, {} minutes and {} seconds".format(dif1[0], *divmod(dif1[1],60)))

batch_size = 900
num_epochs = 15

model.train(num_epochs, batch_size)
t2 = time.time()
dif2 = divmod(t2-t1,3600)
print("Training time: {} hours, {} minutes and {} seconds".format(dif2[0], *divmod(dif2[1],60)))
model.save_model(path+"models/transformer.pkl")
model.load_model(path+"models/transformer.pkl")
result = pd.DataFrame()
n = dataset.data.shape[0]
u=0

while u<n:
    df = model.sample_new_flows(n-u)
    df = postprocess(df,orig_dataset)
    df.dropna(inplace=True)
    result = pd.concat([result, df])
    u+=len(result)

t3 = time.time()

dif3 = divmod(t3-t2,3600)
print("Sampling time: {} hours, {} minutes and {} seconds".format(dif3[0], *divmod(dif3[1],60)))

result.to_csv(path+"transformer_syn.csv", index=False)
