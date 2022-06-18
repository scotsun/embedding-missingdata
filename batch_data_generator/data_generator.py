"""Data generator example."""

from tensorflow import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    """DataGenerator class."""

    def __init__(self, x_in, y_in, batch_size, shuffle=True):
        """Init."""
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x_in
        self.y = y_in
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Get batch indexes from shuffled indexes."""
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch

    def __len__(self):
        """Denote the number of batches per epoch."""
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)
