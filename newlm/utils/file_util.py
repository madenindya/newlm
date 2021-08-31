import os
import warnings
from typing import Dict, Any


def create_dir(dirpath, verbose=False):
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        if verbose:
            warnings.warn(f"Directory {dirpath} already exist")


def is_dir_empty(dirpath):
    if os.path.isdir(dirpath):
        ls = os.listdir(dirpath)
        return len(ls) == 0
    return True


def read_from_yaml(file_path: str) -> Dict[str, Any]:
    import yaml

    with open(file_path, "r+") as fr:
        data_dict = yaml.safe_load(fr)
    return data_dict
