import argparse
import os
import concurrent.futures
from dataloader import Dataset
from model import ChunkyModel
import search

def run_chunk(chunk):

    filename = os.path.basename(chunk)

    d = Dataset.load_model(chunk)
    m_search = search.search(d, model_name=filename)
    m_search.save_model('chunks/model_%s' % filename)

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
