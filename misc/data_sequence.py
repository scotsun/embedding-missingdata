"""Reformat tabulated time-series data to sequence data with the shape: (n_sample, n_timesteps, n_features)."""

import numpy as np
from typing import Iterable
from pandas.core.groupby.generic import DataFrameGroupBy


class DataSequence:
    """Reformat tabuelated time series data in shape (n_sample, timesteps, n_features)."""

    def __init__(self, timesteps: int, batch_size: int, columns: Iterable) -> None:
        """Init."""
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.data: dict[str, list] = dict()
        for col in columns:
            self.data[col] = []
        self._n_sample = 0

    @property
    def n_sample(self) -> int:
        """Get self.n_sample."""
        return self._n_sample

    def add(self, grouped_data: DataFrameGroupBy, id: int) -> None:
        """Add an element by its id."""
        self._n_sample += 1
        for col in self.data.keys():
            elem_data = (
                grouped_data.get_group(id)[col]
                .values.reshape((self.timesteps, 1))
                .tolist()
            )
            self.data[col].append(elem_data)

    def shuffle(self) -> None:
        """Shuffle the sample."""
        _seed = np.random.randint(10000)
        for col in self.data.keys():
            np.random.seed(_seed)
            np.random.shuffle(self.data[col])

    def __getitem__(self, index: int) -> list:
        """Get a batch by the index."""
        start = index * self.batch_size
        end = min(start + self.batch_size, self._n_sample)
        data_batch: list[np.ndarray] = []
        for col in self.data.keys():
            data_batch.append(np.array(self.data[col][start:end]))
        return data_batch

    def __len__(self) -> int:
        """Denote the number of batches."""
        return int(np.ceil(self._n_sample / self.batch_size))
