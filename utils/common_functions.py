from typing import Union
import numpy as np
import random

import pandas as pd


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    """Reads DataFrame file."""
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    elif path_to_file.endswith('parquet'):
        return pd.read_parquet(path_to_file)
    else:
        raise ValueError("Unsupported file format")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def convert_lists_and_tuples_to_string(data):
    """
    Recursively replaces all lists in a dictionary (including nested dictionaries) with a string representation of the list.
    """
    data = data.copy()
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = convert_lists_and_tuples_to_string(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            data[key] = str(value)
        elif value is None:
            data[key] = str(value)
    return data

