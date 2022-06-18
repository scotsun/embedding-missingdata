"""Data generator example."""

from tensorflow import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    """DataGenerator class."""

    def __init__(self, X, y, batch_size, shuffle=True):
        """Init."""
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._X = X
        self._y = y
        self._datalen = len(y)
        class_counts = np.bincount(y)
        self._num_classes = len(class_counts)
        self._indexes = []
        for c in range(self._num_classes):
            self._indexes.append(np.arange(self._datalen)[y == c])
            if self._shuffle:
                np.random.shuffle(self._indexes[c])

    def __getitem__(self, index):
        """Get batch indexes from shuffled indexes."""
        n = int(self._batch_size / 2)
        batch_indexes_list = []
        for c in range(self._num_classes):
            _batch_indexes = circular_slicing(self._indexes[c], index * n, n)
            batch_indexes_list.append(_batch_indexes)
        batch_indexes = np.concatenate(batch_indexes_list)
        X_batch = self._X[batch_indexes]
        y_batch = self._y[batch_indexes]
        return X_batch, y_batch

    def __len__(self):
        """Denote the number of batches per epoch."""
        return self._datalen // self._batch_size

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self._indexes = []
        for c in range(self._num_classes):
            self._indexes.append(np.arange(self._datalen)[self._y == c])
            if self._shuffle:
                np.random.shuffle(self._indexes[c])


def circular_slicing(arr: np.ndarray, start: int, size: int):
    """Slice the given array in a circular way."""
    l = len(arr)
    start = start % l
    if size > l:
        raise ValueError("maybe consider smaller size.")
    if start + size > l:
        return np.append(arr[start:], arr[: (start + size - l)])
    else:
        return arr[start : (start + size)]
