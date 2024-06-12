import random

import numpy as np
import pandas as pd


def read_dataframe_file(path_to_file: str) -> pd.DataFrame:
    """Read DataFrame file.

    Args:
        path_to_file (str): The path to the file to read.

    Raises:
        ValueError: If the file format is not supported.

    Returns:
        pd.DataFrame: The DataFrame read from the file.
    """
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    if path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    if path_to_file.endswith('parquet'):
        return pd.read_parquet(path_to_file)
    raise ValueError('Unsupported file format!')


def set_seed(seed: int) -> None:
    """Set the random seed for the 'random' and 'np.random' modules.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)


def convert_lists_and_tuples_to_string(converting_input: dict) -> dict:
    """Replace recursively all lists and tuples in a dictionary.

    This function replaces all lists and tuples including nested dictionaries
    with a string representation of the list.

    Args:
        converting_input (dict): The dictionary to convert.

    Returns:
        dict: The converted dictionary.
    """
    converting_input = converting_input.copy()
    for key, input_value in converting_input.items():
        if isinstance(input_value, dict):
            converting_input[key] = convert_lists_and_tuples_to_string(
                input_value,
            )
        elif isinstance(input_value, (list, tuple)):
            converting_input[key] = str(input_value)
        elif input_value is None:
            converting_input[key] = str(input_value)
    return converting_input
