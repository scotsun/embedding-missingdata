"""Embedding methods."""


import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import (
    Dense,
    Concatenate,
    Input,
    Lambda,
    Reshape,
    Layer,
    CategoryEncoding,
)
from keras.layers.embeddings import Embedding
from keras import backend as K

# TODO: self._X_train has not developed yet.


class EmbeddingModel:
    """Implement embedding method to solve data missingness. This only has not been generalized yet."""

    def __init__(
        self, predictors: pd.DataFrame, target: np.ndarray, embed_cat: bool
    ) -> None:
        """Init."""
        self._predictors = predictors
        self._target = target
        self._embed_cat = embed_cat
        self._is_built = False
        self._X_train: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def rescale(x):
        """Rescale embedding vectors by raw data (used for Lambda layer)."""
        input_value = x[0]
        embedding = x[1]
        # force cat_code 0 represent missing data
        mask = K.cast(input_value == 0, dtype=K.floatx())
        return embedding * mask + tf.multiply(embedding, input_value) * (1 - mask)

    def _prepare_inputs(self) -> None:
        """Prepare inputs for the embedding model."""
        pass

    def build(self) -> None:
        """Build a pseud-model."""
        embed_size = 4
        sub_layers: list[Layer] = []
        continuous_dim = 4
        continuous_vars = {"count": False, "AGE": False, "WBC": True, "Platelets": True}
        categorical_dim = 3
        categorical_vars = {"ADMIT_TYPE": 4, "SEX": 3, "RACE": 12}
        # set up input layers
        ins: list[Input] = []
        input_cont = Input(shape=(continuous_dim,), name="continous")
        input_cont_indct = Input(shape=(continuous_dim,), name="continous_indctr")
        input_cat = Input(shape=(categorical_dim,), name="categorical")
        for elem in [input_cont, input_cont_indct, input_cat]:
            ins.append(elem)
        # contiunous part
        transformed_input_cont_list: list[Layer] = []
        for i, (var_name, needs_embed) in enumerate(continuous_vars.items()):
            _input_cont = Lambda(lambda x: x[:, i], name=var_name)(input_cont)
            if needs_embed:
                _input_cont_indct = Lambda(
                    lambda x: x[:, i], name=var_name + "_indctr"
                )(input_cont_indct)
                _cont_indct_embed = Embedding(
                    2 + 1,
                    embed_size,
                    input_length=1,
                    name=var_name + "_indctr_embedded",
                )(_input_cont_indct)
                _cont_indct_embed = Reshape(
                    target_shape=(embed_size,),
                    name=var_name + "_indctr_embedded_reshape",
                )(_cont_indct_embed)
                _input_cont = Lambda(self.rescale, name=var_name + "_cont_rescale")(
                    [_input_cont, _cont_indct_embed]
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
        for j, (var_name, k) in enumerate(categorical_vars.items()):
            _input_cat = Lambda(lambda l: l[:, j], name=var_name)(input_cat)
            if self._embed_cat:
                _input_cat = Embedding(
                    input_dim=k + 1,
                    output_dim=embed_size,
                    input_length=1,
                    name=var_name + "_embedded",
                )(_input_cat)
                _input_cat = Reshape(
                    target_shape=(embed_size,), name=var_name + "_embedded_reshape"
                )(_input_cat)
            else:
                _input_cat = CategoryEncoding(
                    num_tokens=k, output_mode=var_name + "_one_hot"
                )(_input_cat)
            sub_layers.append(_input_cat)
        # concat both parts
        x = Concatenate(name="last_concat")(sub_layers)
        for i in range(4):
            x = Dense(200, activation="relu", name="dense_" + str(i))
        out = Dense(1, activation="sigmoid", name="output")(x)
        self._model = Model(inputs=ins, outputs=out)
        self._is_built = True

    @property
    def model(self):
        """Get the model."""
        if not self._is_built:
            raise ValueError("model not built yet.")
        return self._model

    @staticmethod
    def static_build(embed_cat) -> Model:
        """Build a embedding model."""
        embed_size = 4
        sub_layers: list[Layer] = []
        continuous_dim = 4
        continuous_vars = {"count": False, "AGE": False, "WBC": True, "Platelets": True}
        categorical_dim = 3
        categorical_vars = {"ADMIT_TYPE": 4, "SEX": 3, "RACE": 12}
        # set up input layers
        ins: list[Input] = []
        input_cont = Input(shape=(continuous_dim,), name="continous")
        input_cont_indct = Input(shape=(continuous_dim,), name="continous_indctr")
        input_cat = Input(shape=(categorical_dim,), name="categorical")
        for elem in [input_cont, input_cont_indct, input_cat]:
            ins.append(elem)
        # contiunous part
        transformed_input_cont_list: list[Layer] = []
        for i, (var_name, needs_embed) in enumerate(continuous_vars.items()):
            _input_cont = Lambda(lambda x: x[:, i], name=var_name)(input_cont)
            if needs_embed:
                _input_cont_indct = Lambda(
                    lambda x: x[:, i], name=var_name + "_indctr"
                )(input_cont_indct)
                _cont_indct_embed = Embedding(
                    2 + 1,
                    embed_size,
                    input_length=1,
                    name=var_name + "_indctr_embedded",
                )(_input_cont_indct)
                _cont_indct_embed = Reshape(
                    target_shape=(embed_size,),
                    name=var_name + "_indctr_embedded_reshape",
                )(_cont_indct_embed)
                _input_cont = Lambda(
                    EmbeddingModel.rescale, name=var_name + "_cont_rescale"
                )([_input_cont, _cont_indct_embed])
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
        for j, (var_name, k) in enumerate(categorical_vars.items()):
            _input_cat = Lambda(lambda l: l[:, j], name=var_name)(input_cat)
            if embed_cat:
                _input_cat = Embedding(
                    input_dim=k + 1,
                    output_dim=embed_size,
                    input_length=1,
                    name=var_name + "_embedded",
                )(_input_cat)
                _input_cat = Reshape(
                    target_shape=(embed_size,), name=var_name + "_embedded_reshape"
                )(_input_cat)
            else:
                _input_cat = CategoryEncoding(
                    num_tokens=k,
                    output_mode="one_hot",
                    name=var_name + "embedded_reshape",
                )(_input_cat)
            sub_layers.append(_input_cat)
        # concat both parts
        x = Concatenate(name="last_concat")(sub_layers)
        for i in range(4):
            x = Dense(200, activation="relu", name="dense_" + str(i))(x)
        out = Dense(1, activation="sigmoid", name="output")(x)
        return Model(inputs=ins, outputs=out)


def extract_embeddings(model: EmbeddingModel):
    """Extract embedding weights from the corresponding layers."""
    dfs: list[pd.DataFrame] = []
    for layer in model.model.layers:
        if isinstance(layer, Embedding):
            M = layer.get_weights()[0]
            k = M.shape[0]
            layer_name = layer.name
            cat_var = layer_name[:-9]
            _df = pd.DataFrame(
                pd.get_dummies([cat_var]).values.dot(M),
                columns=[layer_name + str(i) for i in range(4)],
                index=model._X_train.index,
            )
            dfs.append(_df)
