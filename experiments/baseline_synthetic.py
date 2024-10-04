import numpy as np
import pandas as pd
import random
import argparse
import sys

import search
import baseline_slim_sqs as baseline
from dataset import SyntheticDataset, Dataset
from synthetic_data import generate_data, plant_patterns, pattern_to_patterns_objects, reshape_data, plant_high_frequency_columns
from model import Model
from pattern import Pattern
from row_pattern import RowPattern
from attribute_value import AttributeValue, AttributeType

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-true', action='store_true')
    parser.add_argument('--skip-null', action='store_true')
    parser.add_argument('--skip-search', action='store_true')
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    N_ROWS = 1000
    N_COLUMNS = 5
    POSSION_PARAMETERS = 2,3 # lambda, shift
    df = generate_data(N_ROWS, N_COLUMNS, POSSION_PARAMETERS)
    df = plant_high_frequency_columns(df, columns=[1,2], unique_values=[500,20])
    df,patterns = plant_patterns(df, unique_patterns=1, pattern_symbols=10, n_rows=20, total_patterns=50, placeholder_pair=(1,1))
    df,patterns = reshape_data(df,patterns)

    dataset = SyntheticDataset(df.copy(), never_fix=set([0]), placeholder_set_columns=set([0]), placeholder_get_columns=set([0]))

    DEBUG = False
    if DEBUG:
        m_debug = Model(dataset)
        # r_0 = RowPattern({2:AttributeValue(AttributeType.FIX,4), 3:AttributeValue(AttributeType.FIX,3), 6:AttributeValue(AttributeType.FIX,1), 8:AttributeValue(AttributeType.FIX,1)})
        # r_1 = RowPattern({4:AttributeValue(AttributeType.FIX,2), 7:AttributeValue(AttributeType.FIX,2)})
        # r_2 = RowPattern({1:AttributeValue(AttributeType.FIX,2), 6:AttributeValue(AttributeType.FIX,2)})
        # r_3 = RowPattern({1:AttributeValue(AttributeType.FIX,2), 3:AttributeValue(AttributeType.FIX,3)})
        # error_pattern = Pattern([r_0,r_1,r_2,r_3])
        error_pattern = pattern_to_patterns_objects(patterns[0])
        m_debug.test_add(error_pattern)
        sys.exit(0)

    if not args.skip_search:
        m_search = search.search(dataset)
        length_model= m_search.get_model_length()
        length_data = m_search.cover.compute_data_length()
        print("\n\n Search  model")
        print(length_model, '+',  length_data, '=',length_model+length_data)
        cs = m_search.cover.get_cover_stats()
        for p,u in cs.get_pattern_usage().items():
            print(p,u)

    if not args.skip_baseline:
        m_search = baseline.search(dataset)
        length_model= m_search.get_model_length()
        length_data = m_search.cover.compute_data_length()
        print("\n\n Baseline  model")
        print(length_model, '+',  length_data, '=',length_model+length_data)
        cs = m_search.cover.get_cover_stats()
        for p,u in cs.get_pattern_usage().items():
            print(p,u)

    if not args.skip_null:
        print("\n\n Null model")
        m_null = Model(dataset)
        length_model= m_null.get_model_length()
        length_data = m_null.cover.compute_data_length()
        print(length_model, '+',  length_data, '=',length_model+length_data)

    if not args.skip_true:
        print("\n\n True model")
        m_ture = Model(dataset)
        m_ture.set_pattern_set(list(map(pattern_to_patterns_objects,patterns)))
        length_model= m_ture.get_model_length()
        length_data = m_ture.cover.compute_data_length()
        print(length_model, '+',  length_data, '=',length_model+length_data)
        cs = m_ture.cover.get_cover_stats()
        for p,u in cs.get_pattern_usage().items():
            print(p,u)