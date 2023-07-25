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

    @property
    def continuous_vars_info(self) -> dict:
        """Get continuous_vars_info."""
        return self._continuous_var_info

    @property
    def categorical_vars_info(self) -> dict:
        """Get categorical_vars_info."""
        return self._categorical_vars_info

    @property
    def continuous_data(self) -> np.ndarray:
        """Get data from the continuous variables."""
        return self._rawX[self._continuous_vars_info.keys()].values

    @property
    def continuous_indctr_data(self) -> np.ndarray:
        """Get a indicator matrix w.r.t. the continuous data. 0 := missing, 1 := observed."""
        return 1 - np.isnan(self.continuous_data)

    @property
    def categorical_data(self) -> np.ndarray:
        """Get data from categorical variables in cat mode format."""
        return self._rawX[self._categorical_vars_info.keys()].values

    @property
    def n_sample(self) -> int:
        """Get the number of samples."""
        return self._n_samples

    @abstractclassmethod
    def __getitem__(self, index) -> tuple:
        """Get a batch data."""
        pass

    @abstractclassmethod
    def __len__(self) -> int:
        """Get the total number of batches per epoch."""
        pass

    @abstractclassmethod
    def on_epoch_end(self) -> None:
        """Shuffle data after each epoch."""
        pass


class DataSeqLoader(DataLoader):
    """DataSeqLoader, a subclass of DataLoader, which reshape the data into DataSequence objects."""

    def __init__(
        self, X: pd.DataFrame, y: pd.Series, ids: pd.Series, batch_size: int
    ) -> None:
        """Init."""
        super().__init__(X, y, ids, batch_size)
        self._populate_data_seq()

    def _populate_data_seq(self) -> None:
        """Reshape data as DataSequence."""
        data = self._rawX.copy()
        data[self._rawy.name] = self._rawy
        grouped_data = data.groupby(data.index.name)
        timesteps = grouped_data[self._rawy.name].count()
        # initialize
        data_seqs: list[DataSequence] = [
            DataSequence(
                timesteps=t,
                batch_size=self._batch_size,
                columns=data.columns,
                y_var_name=self._rawy.name,
            )
            for t in range(1, 15)
        ]
        # start populating
        for pid in tqdm(timesteps.indices):
            t = timesteps.get(pid)
            data_seqs[t - 1].add(grouped_data, pid)
        self.data_seqs = data_seqs

    def __getitem__(self, index) -> tuple:
        """Get a batch by a given index."""
        for seq in self.data_seqs:
            if index < len(seq):
                return seq[index]
            else:
                index -= len(seq)
        raise ValueError("index out of upper bound.")

    def __len__(self) -> int:
        """Get the size of the object."""
        return sum([len(data_seq) for data_seq in self.data_seqs])

    def on_epoch_end(self) -> None:
        """Shuffle sample randomly by the end of an epoch."""
        for data_seq in self.data_seqs:
            data_seq.shuffle()


class DataTableLoader(DataLoader):
    """DataTableLoader, a subclass of DataLoader, which maintain the tabular format of the raw data."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ids: pd.Series,
        batch_size: int,
        balance: bool = True,
    ) -> None:
        """Init."""
        super().__init__(X, y, ids, batch_size)
        self._balance = balance
        self._datalen = len(self._rawy)
        self._num_classes = len(np.unique(self._rawy))
        self._X = [self._rawX.values[:, i] for i in range(self._rawX.shape[1])]
        self._y = self._rawy.values
        self._indices = []
        for c in range(self._num_classes):
            self._indices.append(np.arange(self._datalen)[self._rawy.values == c])

    def __getitem__(self, index) -> tuple:
        """Get the batch by the given index."""
        # first obtain the row-level indices for the batch specified by the time index
        if self._balance:
            n = int(self._batch_size / 2)
            batch_indices_list = []
            for c in range(self._num_classes):
                _batch_indices = circular_slicing(self._indices[c], index * n, n)
                batch_indices_list.append(_batch_indices)
            batch_indices = np.concatenate(batch_indices_list)
        else:
            start = np.min([index * self._batch_size, self._rawy.size])
            end = np.min([(index + 1) * self._batch_size, self._rawy.size])
            batch_indices = np.concatenate(self._indices)[start:end]
        # obtain the batch data
        X_batch = [elem[batch_indices] for elem in self._X]
        y_batch = self._y[batch_indices]
        return X_batch, y_batch

    def __len__(self) -> int:
        """Get the number of the batches."""
        if self._balance:
            return self._datalen // int(self._batch_size / 2)
        else:
            return self._datalen // self._batch_size

    def on_epoch_end(self) -> None:
        """Shuffle samples randomly by the end of an epoch."""
        for c in range(self._num_classes):
            np.random.shuffle(self._indices[c])


def train_test_split(
    df: pd.DataFrame, frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train set and validation set."""
    if frac < 0.0 or frac > 1.0:
        raise ValueError("Incorrect value for frac, which should be in [0, 1]")
    pid = np.unique(df["PAT_ENC_CSN_ID"].values)
    N = len(pid)
    pid_valid = np.random.choice(pid, int(frac * N))
    valid = df.loc[df["PAT_ENC_CSN_ID"].isin(pid_valid)]
    train = df.loc[~df["PAT_ENC_CSN_ID"].isin(pid_valid)]
    return (train, valid)


def circular_slicing(arr: np.ndarray, start: int, size: int):
    """Slice the given array in a circular way.

    The cursor will be reset to the beginning for large end indices so that the extended portion is included.
    """
    arr_len = len(arr)
    start = start % arr_len
    if size > arr_len:
        raise ValueError("maybe consider smaller batch size.")
    if start + size > arr_len:
        return np.append(arr[start:], arr[: (start + size - arr_len)])
    else:
        end = start + size
        return arr[start:end]
