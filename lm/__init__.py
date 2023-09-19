from typing import Tuple
import numpy as np

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives
