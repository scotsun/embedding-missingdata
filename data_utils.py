"""Data utility classes."""

from abc import abstractclassmethod
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from tqdm import tqdm
from keras.utils import Sequence


class DataSequence:
    """Store tabuelated time-serie data in shape (n_sample, timesteps, 1) for each variable."""

    def __init__(
        self, timesteps: int, batch_size: int, columns: Iterable, y_var_name: str
    ) -> None:
        """Init."""
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.data: dict[str, list] = dict()
        self.y_var_name = y_var_name
        for col in columns:
            self.data[col] = []
        self.n_sample = 0

    @property
    def n_sample(self) -> int:
        """Get self.n_sample."""
        return self.n_sample

    def add(self, grouped_data: DataFrameGroupBy, id: int) -> None:
        """Add a patient data by the id."""
        self._n_sample += 1
        for col in self.data.keys():
            elem_data = (
                grouped_data.get_group(id)[col]
                .values.reshape((self.timesteps, 1))
                .tolist()
            )
            self.data[col].append(elem_data)

    def shuffle(self) -> None:
        """Shuffle samples."""
        _seed = np.random.randint(1e4)
        for col in self.data.keys():
            np.random.seed(_seed)
            np.random.shuffle(self.data[col])

    def __getitem__(self, index: int) -> tuple:
        """Get a batch by the index."""
        start = index * self.batch_size
        end = min(start + self.batch_size, self.n_sample)
        X_batch: list[np.ndarray] = []
        for col in self.data.keys():
            if col != self.y_var_name:
                X_batch.append(np.array(self.data[col][start:end]))
        y_batch: np.ndarray = np.array(self.data[self.y_var_name][start:end])
        return X_batch, y_batch

    def __len__(self) -> int:
        """Denote the number of batches."""
        return int(np.ceil(self.n_sample / self.batch_size))


class DataLoader(Sequence):
    """Store the data and reformat them to the LSTM model."""

    def __init__(
        self, X: pd.DataFrame, y: pd.Series, ids: pd.Series, batch_size: int
    ) -> None:
        """init."""
        self._rawX = X
        self._rawy = y
        self._batch_size = batch_size
        self._rawX.index = ids
        self._rawy.index = ids
        self._n_samples = np.unique(ids).size

    def _parse_and_preprocess(self) -> None:
        """Parse data, and augment the data by including indicators and impute NAs with -999."""
        # categorical info: {var_name: num categories}
        cat_col_names = self._rawX.select_dtypes(include="object").columns
        categorical_vars_info: dict[str, int] = dict()
        for var in cat_col_names:
            self._rawX[var] = self._rawX[var].fillna("Missing")
            self._rawX[var] = self._rawX[var].astype("category")
            categorical_vars_info[var] = self._rawX[var].unique().size
        self._categorical_vars_info = categorical_vars_info
        self._categorical_vars_map = (
            self._rawX.select_dtypes(include="category")
            .apply(lambda x: dict(enumerate(x.cat.categories)))
            .values
        )
        self._rawX[cat_col_names] = self._rawX.select_dtypes(include="category").apply(
            lambda x: x.cat.codes
        )
        # continuous info: {var_name: if_has_na}
        self._continuous_vars_info = (
            self._rawX.select_dtypes(include=["int", "float"])
            .apply(lambda x: any(pd.isna(x)), axis=0)
            .to_dict()
        )
        p = len(self._continuous_vars_info)
        cont_col_names = list(self._continuous_vars_info.keys())
        cont_indicate = list(self._continuous_vars_info.values())
        cont_indicate_data = self.continous_indctr_data
        for i in range(p):
            if cont_indicate[i]:
                self._rawX[cont_col_names[i] + "_indctr"] = cont_indicate_data[:, i]
        self._rawX = self._rawX.fillna(-999)
        # sort predictors order to match deep learning model
        X_columns: list[str] = []
        for var_name, needs_embed in self._continuous_vars_info.items():
            X_columns.append(var_name)
            if needs_embed:
                X_columns.append(var_name + "_indctr")
        for var_name in self._categorical_vars_info.keys():
            X_columns.append(var_name)
        self._rawX = self._rawX[X_columns]
