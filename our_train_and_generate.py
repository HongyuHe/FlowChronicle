import os
import dataloader as dl
import search
from temporal_sampling import PatternSampler, FlowSampler
import argparse
import time
import logging
import warnings
import pandas as pd
import concurrent.futures

def run_model_on_chunk_i(chunk, idx, model_name="No_Name"):
    #We return the index in order to keep the initial ordering at the end
    return idx, search.search(chunk, model_name=model_name)

def run_chunks(chunks, global_dataset, save_path=None):
    models=[None] * len(chunks)
    # Use ProcessPoolExecutor to parallelize the execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all the tasks to the executor
        futures = [executor.submit(run_model_on_chunk_i, chunk, i) for i, chunk in enumerate(chunks)]
        # Collect the results as they are completed
        for future in concurrent.futures.as_completed(futures):
            index, model_result = future.result()
            models[index] = model_result
            if save_path is not None:
                model_result.save_model(save_path+f"chunk{index}.pkl")
    model = ChunkyModel(global_dataset, models)
    return model

def get_flows(patterns_chunk, col_name_map, columns_value_dict, cont_repr, timestamps, start_id = 0, idxs=None):
    #This function get a list of pattern and sample network flow inside and return a dataset of network flow
    chunk_result = pd.DataFrame()
    for i, patt in enumerate(patterns_chunk):#For every pattern in the list
        id = start_id+i # Calculate the global id based on the chunk's start position
        if idxs is not None:#If we get the identifier of the patterns
            fl = FlowSampler(patt, col_name_map, columns_value_dict, cont_repr, timestamps).get_flows()#Sample flow for that pattern
            fl["pattern_idx"]=idxs[id]
            chunk_result = pd.concat([chunk_result, fl], axis=0)
        else:
            fl = FlowSampler(patt, col_name_map, columns_value_dict, cont_repr, timestamps).get_flows()#Sample flow for that pattern
            chunk_result = pd.concat([chunk_result, fl], axis=0)
    return chunk_result

def parrallelize_flows(synthetic_patterns, col_name_map, columns_value_dict, cont_repr, timestamps, cpus = os.cpu_count()/4, idxs=None):
    #This function generated flow from a model
    synthetic_flows= []
    n = len(synthetic_patterns)
    chunk_size = int(max(1, n // cpus)) # Ensure at least 1 item per chunk

    # Creating chunks of synthetic_patterns
    chunks = [synthetic_patterns[i:i + chunk_size] for i in range(0, n, chunk_size)]
    
    #Parralelization of the sampling
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(cpus)) as executor:
        futures = [executor.submit(get_flows, chunk, col_name_map, columns_value_dict, cont_repr, timestamps, i*chunk_size, idxs) for i, chunk in enumerate(chunks)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            synthetic_flows.append(result)
            logging.info(f"Completed processing chunk {chunk_id + 1}/{len(chunks)}")
    return pd.concat(synthetic_flows, axis=0)

if __name__ == "__main__":
    n_split = 350
    split = n_split > 0
    save_model = False
    
    path = "data/"

    print(path)

    t0 = time.time()

    train_dataset, dataset_chunked = dl.load_CIDDS_splitted_dataset(path+"train.csv", n_split)

    t1 = time.time()
    dif1 = divmod(t1-t0,3600)
    print("Preprocessing time: {} hours, {} minutes and {} seconds".format(dif1[0], *divmod(dif1[1],60)))

    m = run_chunks(dataset_chunked, train_dataset, save_path=f"data/chunks/our_split{n_split}_") if split else search.search(train_dataset, load_checkpoint=0, model_name=f'CIDDS_{args.experiment}_our')

    t2 = time.time()
    dif2 = divmod(t2-t1,3600)
    print("Training time: {} hours, {} minutes and {} seconds".format(dif2[0], *divmod(dif2[1],60)))

    if save_model:
        m.save_model(f"models/CIDDS-our.pkl")

    c = m.cover
    cover_stats = c.get_cover_stats()
    patterns_usage = cover_stats.get_pattern_usage()

    c.fit_temporal_samplers(patterns_usage.keys())
    idx, synthetic_patterns = PatternSampler(patterns_usage).sample(sum(patterns_usage.values()), return_indices=True)
    
    synthetic_df = parrallelize_flows(synthetic_patterns, c.dataset.col_name_map, c.dataset.columns_value_dict, c.dataset.cont_repr, c.dataset.time_stamps, idxs=idx)
    synthetic_df = synthetic_df.sort_values('Date first seen')

    synthetic_df.insert(0, 'Date first seen', synthetic_df.pop('Date first seen'))

    t3 = time.time()
    dif3 = divmod(t3-t2,3600)
    print("Sampling time: {} hours, {} minutes and {} seconds".format(dif3[0], *divmod(dif3[1],60)))

    synthetic_df.to_csv(path+"our_new_syn.csv", index=False)
