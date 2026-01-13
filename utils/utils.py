import json
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_results(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


