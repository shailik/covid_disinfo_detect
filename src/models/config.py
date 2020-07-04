"""
Config file for generating embeddings component of project pipeline.
"""
import os
from random import seed
from numpy import random
from torch import manual_seed


SEED_VALUE = 8


def set_seed(SEED_VALUE):
    """
    Set the random seed for generating embeddings
    """
    # Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
    # Set `python` built-in pseudo-random generator at a fixed value
    seed(SEED_VALUE)
    # Set `numpy` pseudo-random generator at a fixed value
    random.seed(SEED_VALUE)
    # set `torch` pseudo-random generator at a fixed value
    manual_seed(SEED_VALUE)
