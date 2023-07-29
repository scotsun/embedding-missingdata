"""NN Models."""

from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd

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


class EmbeddingModel(NN):
    """Implement embedding method to solve missingness."""

    def build(self) -> None:
        """Build the model."""
        if self._dataloader is None:
            raise AttributeError(
                "Building the model is not applicable as the dataloader is empty."
            )
        self._is_built = True
        dataloader = self._dataloader
        embed_size = self._embed_size
        sub_layers: list[Layer] = []
        # set up input layer
        ins: list[Input] = []
        # continuous part
        transformed_input_cont_list: list[Layer] = []
        var_name: str
        needs_embed: bool
        for var_name, needs_embed in dataloader.continuous_vars_info.items():
            var_name = var_name.lower()
            _input_cont = Input(shape=(1,), name=var_name)
            ins.append(_input_cont)
            if needs_embed:
                _input_cont_indct = Input(shape=(1,), name=var_name + "_indct")
                ins.append(_input_cont_indct)
                _cont_indct_embed = Embedding(
                    2, embed_size, input_length=1, name=var_name + "_indctr_embedded"
                )(_input_cont_indct)
                _cont_indct_embed = Reshape(
                    target_shape=(embed_size,),
                    name=var_name + "_indctr_embedded_reshape",
                )(_cont_indct_embed)
                _input_cont = Lambda(rescale, name=var_name + "_cont_rescale")(
                    (_input_cont, _cont_indct_embed)
                )
            else:
                _input_cont = Reshape(target_shape=(1,), name=var_name + "_reshape")(
                    _input_cont
                )
            transformed_input_cont_list.append(_input_cont)
        transformed_input_cont = Concatenate(name="transformed_input_cont")(
            transformed_input_cont_list
        )
        sub_layers.append(transformed_input_cont)
        # categorical part
        for var_name, k in dataloader.categorical_vars_info.items():
            var_name = var_name.lower()
            _input_cat = Input(shape=(1,), name=var_name)
            ins.append(_input_cat)
            if self._embed_categorical_data:
                _input_cat = Embedding(
                    input_dim=k,
                    output_dim=embed_size,
                    input_length=1,
                    name=var_name + "_embedded",
                )(_input_cat)
                _input_cat = Reshape(
                    target_shape=(embed_size,), name=var_name + "embedded_reshape"
                )(_input_cat)
            else:  # otherwise, apply one-hot encoding
                _input_cat = CategoryEncoding(
                    num_tokens=k, output_mode="one_hot", name=var_name + "_one_hot"
                )(_input_cat)
                sub_layers.append(_input_cat)
        # concat both parts
        x = Concatenate(name="last_concat")(sub_layers)
        for i in range(4):
            x = Dense(100, activation="relu", name="dense" + str(i))(x)
            x = Dropout(0.5)(x)
        out = Dense(1, activation="sigmoid", name="output")(x)
        self._model = Model(inputs=ins, outputs=out)


class MLPModel(NN):
    """MLP Classifier."""

    def build(self) -> None:
        """Build the model."""
        if self._dataloader is None:
            raise AttributeError(
                "Building the model is not applicable as the dataloader is empty."
            )
        self._is_built = True
        dataloader = self._dataloader
        embed_size = self._embed_size
        sub_layers: list[Layer] = []
        # set up input layers
        ins: list[Input] = []
        # continuous part
        var_name: str
        for var_name, _ in dataloader.continuous_vars_info.items():
            var_name = var_name.lower()
            _input_cont = Input(shape=(1,), name=var_name)
            ins.append(_input_cont)
            sub_layers.append(_input_cont)
        # categorical part
        for var_name, k in dataloader.categorical_vars_info.items():
            var_name = var_name.lower()
            _input_cat = Input(shape=(1,), name=var_name)
            ins.append(_input_cat)
            if self._embed_categorical_data:
                _input_cat = Embedding(
                    input_dim=k,
                    output_dim=embed_size,
                    input_length=1,
                    name=var_name + "_embedded",
                )(_input_cat)
                _input_cat = Reshape(
                    target_shape=(embed_size,), name=var_name + "_embedded_reshape"
                )(_input_cat)
            else:  # otherwise, one-hot encode
                _input_cat = CategoryEncoding(
                    num_tokens=k, output_mode="one_hot", name=var_name + "_one_hot"
                )(_input_cat)
            sub_layers.append(_input_cat)
        # concat both parts
        x = Concatenate(name="last_concat")(sub_layers)
        for i in range(4):
            x = Dense(100, activation="relu", name="dense" + str(i))(x)
            x = Dropout(0.5)(x)
        out = Dense(1, activation="sigmoid", name="output")(x)
        self._model = Model(inputs=ins, outputs=out)


class EmbeddingLSTMModel(NN):
    """Implement embedding method to solve missingness with LSTM seq2seq classifier."""

    def build(self) -> None:
        """Build the model."""
        if self._dataloader is None:
            raise AttributeError(
                "Building the model is not applicable as the dataloader is empty."
            )
        self._is_built = True
        dataloader = self._dataloader
        embed_size = self._embed_size
        sub_layers: list[Layer] = []
        # set up input layer
        ins: list[Input] = []
        # continuous part
        transformed_input_cont_list: list[Layer] = []
        var_name: str
        needs_embed: bool
        for var_name, needs_embed in dataloader.continuous_vars_info.items():
            var_name = var_name.lower()
            _input_cont = Input(
                shape=(
                    None,
                    1,
                ),
                name=var_name,
            )
            ins.append(_input_cont)
            if needs_embed:
                _input_cont_indct = Input(
                    shape=(
                        None,
                        1,
                    ),
                    name=var_name + "_indct",
                )
                ins.append(_input_cont_indct)
                _cont_indct_embed = TimeDistributed(
                    Embedding(2, embed_size, input_length=1),
                    name=var_name + "_indctr_embedded",
                )(_input_cont_indct)
                _cont_indct_embed = TimeDistributed(
                    Reshape(target_shape=(embed_size,)),
                    name=var_name + "_indctr_embedded_reshape",
                )(_cont_indct_embed)
                _input_cont = TimeDistributed(
                    Lambda(rescale), name=var_name + "_cont_rescale"
                )((_input_cont, _cont_indct_embed))
            else:
                _input_cont = TimeDistributed(
                    Reshape(target_shape=(1,)), name=var_name + "_reshape"
                )(_input_cont)
            transformed_input_cont_list.append(_input_cont)
        transformed_input_cont = TimeDistributed(
            Concatenate(), name="transformed_input_cont"
        )(transformed_input_cont_list)
        sub_layers.append(transformed_input_cont)
        # categorical part
        for var_name, k in dataloader.categorical_vars_info.items():
            var_name = var_name.lower()
            _input_cat = Input(
                shape=(
                    None,
                    1,
                ),
                name=var_name,
            )
            ins.append(_input_cat)
            if self._embed_categorical_data:
                _input_cat = TimeDistributed(
                    Embedding(input_dim=k, output_dim=embed_size, input_length=1),
                    name=var_name + "_embedded",
                )(_input_cat)
                _input_cat = TimeDistributed(
                    Reshape(target_shape=(embed_size,)),
                    name=var_name + "embedded_reshape",
                )(_input_cat)
            else:  # otherwise, apply one-hot encoding
                _input_cat = TimeDistributed(
                    CategoryEncoding(num_tokens=k, output_mode="one_hot"),
                    name=var_name + "_one_hot",
                )(_input_cat)
                sub_layers.append(_input_cat)
        # concat both parts
        x = TimeDistributed(Concatenate(), name="last_concat")(sub_layers)
        for i in range(3):
            x = LSTM(128, return_sequences=True, name="lstm" + str(i))(x)
        out = TimeDistributed(Dense(1, activation="sigmoid"), name="output")(x)
        self._model = Model(inputs=ins, outputs=out)


class LSTMModel(NN):
    """LSTM seq2seq classifier."""

    def build(self) -> None:
        """Build the model."""
        if self._dataloader is None:
            raise AttributeError(
                "Building the model is not applicable as the dataloader is empty."
            )
        self._is_built = True
        dataloader = self._dataloader
        embed_size = self._embed_size
        sub_layers: list[Layer] = []
        # set up input layers
        ins: list[Input] = []
        # continuous part
        var_name: str
        for var_name, _ in dataloader.continuous_vars_info.items():
            var_name = var_name.lower()
            _input_cont = Input(
                shape=(
                    None,
                    1,
                ),
                name=var_name,
            )
            ins.append(_input_cont)
            sub_layers.append(_input_cont)
        # categorical part
        for var_name, k in dataloader.categorical_vars_info.items():
            var_name = var_name.lower()
            _input_cat = Input(
                shape=(
                    None,
                    1,
                ),
                name=var_name,
            )
            ins.append(_input_cat)
            if self._embed_categorical_data:
                _input_cat = TimeDistributed(
                    Embedding(input_dim=k, output_dim=embed_size, input_length=1),
                    name=var_name + "_embedded",
                )(_input_cat)
                _input_cat = TimeDistributed(
                    Reshape(target_shape=(embed_size,)),
                    name=var_name + "_embedded_reshape",
                )(_input_cat)
            else:  # otherwise, one-hot encode
                _input_cat = TimeDistributed(
                    CategoryEncoding(num_tokens=k, output_mode="one_hot"),
                    name=var_name + "_one_hot",
                )(_input_cat)
            sub_layers.append(_input_cat)
        # concat both parts
        x = TimeDistributed(Concatenate(), name="last_concat")(sub_layers)
        for i in range(3):
            x = LSTM(128, return_sequences=True, name="lstm" + str(i))(x)
        out = TimeDistributed(Dense(1, activation="sigmoid"), name="output")(x)
        self._model = Model(inputs=ins, outputs=out)


def rescale(x: tuple[Any, Any]):
    """Rescale embeddings by the raw data (if observed). This function is used for Lambda layer."""
    input_value = x[0]
    embedding = x[1]
    # force input value -999.0 represent missing data
    mask = K.cast(input_value == -999.0, dtype=K.floatx())
    return embedding * mask + tf.multiply(embedding, input_value) * (1 - mask)
