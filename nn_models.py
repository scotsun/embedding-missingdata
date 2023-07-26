"""NN Models."""

from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from keras import Model
from keras.layers import (
    Dense,
    Dropout,
    LSTM,
    TimeDistributed,
    Concatenate,
    Input,
    Lambda,
    Reshape,
    CategoryEncoding,
    Embedding,
    Layer,
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.optimizers import Adam
from keras.metrics import AUC
from keras import backend as K
from keras.utils import to_categorical

from data_utils import DataLoader, DataSeqLoader, DataTableLoader


class NN:
    """Abstract class for deep learning classes."""

    def __init__(
        self, dataloader: DataLoader | None, embed_categorical_data: bool, embed_size=2
    ) -> None:
        """Init."""
        self._dataloader = dataloader
        self._embed_categorical_data = embed_categorical_data
        self._embed_size = embed_size
        self._model: Model | None = None

    @abstractmethod
    def build(self) -> None:
        """Build the model."""
        ...

    @property
    def model(self) -> Model:
        """Get the model."""
        if self._model is None:
            raise ValueError("Model has not been build yet.")
        return self._model

    def load_weight(self, filepath: str) -> None:
        """Load model weights."""
        self.model.load_weights(filepath)

    def fit(
        self, val_data: DataLoader, filepath: str, metric_curve: str, epochs=50, lr=1e-4
    ) -> History:
        """Fit the NN model."""
        if self._dataloader is None:
            raise AttributeError(
                "Fitting the data is not applicable as the self._dataloader is None."
            )
        if self._model is None:
            raise ValueError("Model has not been built yet.")
        auc = AUC(curve=metric_curve, name="auc")
        self._model.compile(
            loss="binary_crossentropy", optimizer=Adam(learning_rate=lr), metrics=[auc]
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        save_best = ModelCheckpoint(
            filepath,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
        )
        return self._model.fit(
            x=self._dataloader,
            epochs=epochs,
            callbacks=[early_stopping, save_best],
            validation_data=val_data,
        )

    def outcome_data(self, dataloader: DataLoader | None = None) -> pd.DataFrame:
        """Output a dataframe with columns: {outcome, time_index, probability}."""
        if self._dataloader is None or self._model is None:
            raise AttributeError("Model is not complete yet.")
        if dataloader is None:
            dataloader = self._dataloader
        if isinstance(dataloader, DataSeqLoader):
            y = np.vstack(
                [np.array(elem.data["outcome"]).reshape(-1, 1)]
                for elem in dataloader.data_seqs
            )  # type: ignore
            count = np.vstack(
                [
                    np.array(elem.data["count"]).reshape((-1, 1))
                    for elem in dataloader.data_seqs
                ]
            )
            p_ragged = self.model.predict(dataloader)
            p = p_ragged.flat_values.numpy()
        elif isinstance(dataloader, DataTableLoader):
            y = dataloader._y
            count = dataloader._rawX["count"].values
            # note: use _X (not adjusted for class imbalance) instead of the dataloader
            p = self.model.predict(dataloader._X)
        else:
            pass
        outcome_data = pd.DataFrame()
        outcome_data["p"] = p.reshape(-1)
        outcome_data["count"] = count.reshape(-1)
        outcome_data["outcome"] = y.reshape(-1)
        return outcome_data
