from utils import init_experiment
from data import preprocess_data
from trainer import train_cnn_3d

if __name__ == "__main__":
    cfg = init_experiment(config_path="config.yaml")

    # preprocess_data(config=cfg)
    train_cnn_3d(config=cfg)
