#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:03:37 2024

@author: anonymous
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader

class TransformerModel():
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.config = GPT2Config(vocab_size=dataset.n_tokens, n_positions=dataset.sequence_length)
        self.model = GPT2LMHeadModel(self.config)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)
        self.device=device

    def train(self, n_epochs, batch_size):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer=torch.optim.Adam(self.model.parameters())
        loss_fn=torch.nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            for inputs, labels in tqdm(data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = loss_fn(outputs.logits.view(-1, self.config.vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, batch_size):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            inputs, _ = next(iter(data_loader))
            inputs = inputs.to(self.device)
            predictions = self.model(inputs)
            predicted_indices = torch.argmax(predictions.logits, dim=-1).cpu()
            predicted_numbers = predicted_indices[:, -1].tolist()
        return predicted_numbers


    def generate_sequence(self, seed_sequence, length=100):
        self.model.eval()
        sequence = seed_sequence.copy()
        sequence_length = self.dataset.sequence_length
        with torch.no_grad():
            for i in tqdm(list(range(length))):
                input_tensor = torch.tensor(sequence[-sequence_length:]).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)

                # Generate the next token
                output = self.model(input_tensor)
                next_token = torch.argmax(output.logits, dim=-1).cpu().squeeze().tolist()[-1]

                # Append the generated token to the sequence
                sequence.append(next_token)

                # # If the sequence gets too long, trim it to the last 'sequence_length' elements
                # if len(sequence) > sequence_length:
                #     sequence = sequence[-sequence_length:]

        return sequence[-length:]  # Return only the generated part

    def sample_new_flows(self, n):
        len_seq = n * self.dataset.flow_len
        seed = self.dataset.sequence[-self.dataset.sequence_length:]
        sequence = self.generate_sequence(list(seed), len_seq)
        sequence = np.asarray(sequence)
        df_value = np.asarray(np.split(sequence, n, axis=0))
        df = pd.DataFrame(df_value, columns=self.dataset.data.columns)
        return df

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        stat_dict = torch.load(file_path)
        new_state_dict = {key.replace("module.",""): value for key, value in stat_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        print(f"Model loaded from {file_path}")

