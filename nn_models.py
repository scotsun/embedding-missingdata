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
