"""NN Models."""

from abc import abstractmethod
from typing import Any

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
