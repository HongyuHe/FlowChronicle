#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:56:13 2024

@author: aschoen
"""

import torch
import pandas as pd
import numpy as np
#from dataloader import preprocess_CIDDS_dataset, discretize, Dataset
from dataloader import load_CIDDS_dataset, reconstruct_bytes
from bidict import bidict
from temporal_sampling import MarginalSampler
from attribute_value import AttributeType, AttributeValue

path = '../data/xp1/CIDDS_001_train.csv'
out_path ='../data/xp1/transformer_preprocessed.csv'

def preprocess(path):
#    df = preprocess_CIDDS_dataset(dataframe)
#    df = df[dataframe.columns]
#    df = discretize(df)
#    dataset = Dataset(df.copy())
    dataset = load_CIDDS_dataset(path)
#    df = df.iloc[:,1:] #Drop the timestamps from the initial dataframe
#    unique_value = pd.unique(df.values.ravel())
#    column_value_dict = {col: bidict(zip(range(len(unique_value)),unique_value)) for col in df.columns}
#    df.replace({col: dic.inverse for col, dic in column_value_dict.items()}, inplace=True) #Very long
#    df.columns = dataset.flow_features.columns
#    dataset.column_value_dict = column_value_dict
#    dataset.flow_features = df.astype(int)

    return dataset

def postprocess(results, orig_dataset):
    df = results.copy()
    attr = AttributeType(1)
    #col_map = {int(k):v for k,v in orig_dataset.col_name_map.items()}
    for col in df.columns:
        f = lambda x: AttributeValue(attr, x).get_real_value_repr(int(col), orig_dataset)
        #f = np.vectorize(f)
        df[col] = df[col].apply(f)
    #df.rename(columns= col_map,inplace=True)
    df.rename(columns= orig_dataset.col_name_map,inplace=True)
    df.replace(orig_dataset.column_value_dict,inplace=True)
    for c in ['In Byte', 'Out Byte', 'In Packet', 'Out Packet', 'Duration']:
        df[c] = reconstruct_bytes(df[c], orig_dataset.cont_repr.get_cutpoints())
        df[c] = df[c].round().astype(int)
    timestamp_sampler = MarginalSampler(orig_dataset.time_stamps.values).train()
    timestamps = timestamp_sampler.sample(len(df)).flatten()
    timestamps = timestamps/orig_dataset.time_precition
    timestamps = pd.to_timedelta(timestamps, unit='s')+orig_dataset.cont_repr.get_first_flow_time()
    df.insert(0, "Date first seen", timestamps)
    return df


class IntegerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, context_size):
        self.orig_dataset = dataset
        self.data = dataset.flow_features
        self.flow_len = self.data.shape[1]
        self.sequence = self.data.to_numpy().flatten()
        self.n_tokens = len(np.unique(self.sequence))
        self.sequence_length = context_size

    def __len__(self):
        return len(self.sequence) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequence[index:index+self.sequence_length]),
            torch.tensor(self.sequence[index+1:index+self.sequence_length+1])
        )
