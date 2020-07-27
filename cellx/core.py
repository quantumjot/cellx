import os
import json
import numpy as np


def example_function() -> str:
    return 'hello world'


def read_json_data(filename: str) -> dict:
    """ read a list of data from a json file """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
