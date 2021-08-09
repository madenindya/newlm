import os
import warnings
from typing import Dict, Any


def create_dir(dir, verbose=False):
    try:
        os.makedirs(dir)
    except FileExistsError:
        if verbose:
            warnings.warn(f"Directory {dir} already exist")


def read_from_yaml(file_path: str) -> Dict[str, Any]:
    import yaml

    with open(file_path, "r+") as fr:
        data_dict = yaml.safe_load(fr)
    return data_dict
