from utils import init_experiment
from data import preprocess_data
import numpy as np

if __name__ == "__main__":
    cfg = init_experiment(config_path="config.yaml")

    preprocess_data(config=cfg)
