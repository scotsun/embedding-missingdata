"""Slice layer."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Concatenate, Input, Flatten, Lambda

layer = Lambda(lambda l: l[:, 0], name="slice")
arr = np.array([[1, 1, 0], [2, 1, 8], [1, 1, 1], [1, 3, 9]])

print(layer(arr))
