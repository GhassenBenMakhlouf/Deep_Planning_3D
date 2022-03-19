import numpy as np
import glob


def load_dataset(dataset_path):
    filenames = glob.glob(dataset_path + "/*.npy")
    X = []
    Y = []
    for filename in filenames:
        array = np.load(filename)
        X.append(array[:,:,:,:2])
        Y.append(array[:,:,:,:2:])
    
    return np.array(X), np.array(Y)