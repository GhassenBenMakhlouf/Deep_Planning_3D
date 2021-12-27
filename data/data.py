import numpy as np
import glob
import math
from tensorflow.keras.utils import Sequence


class DP3dDataset(Sequence):
    def __init__(self, dataset_path, batch_size):
        self.filenames = glob.glob(dataset_path + "/*.npy")
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        batch_filenames = self.filenames[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = []
        batch_y = []
        for filename in batch_filenames:
            array = np.load(filename)
            batch_x.append(array[:, :, :, :2])
            batch_y.append(array[:, :, :, :2:])

        return np.array(batch_x), np.array(batch_y)
