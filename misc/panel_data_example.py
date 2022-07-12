"""Panel Data Example."""

from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dense, LSTM


def generate_data(n_obs: int, timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate simulated data."""
    X = np.zeros((n_obs, timesteps * 2))
    X[: int(n_obs / 2), 1] = 1
    Y = np.zeros((n_obs, 1))
    for i in range(n_obs):
        for j in range(0, timesteps * 2, 2):
            X[i, j] = np.random.random()
        if X[i, 1] == 0:
            Y[i] = X[i, -2] * 3
        if X[i, 1] == 1:
            Y[i] = X[i, -2] * 9
    X = X.reshape((n_obs, timesteps, 2))
    return X, Y


def main() -> None:
    """Build and fit NN."""
    X, Y = generate_data(10000, 10)
    lstm_nn = models.Sequential()
    lstm_nn.add(LSTM(4, activation="tanh", input_shape=(10, 2)))
    lstm_nn.add(Dense(1))
    lstm_nn.compile(optimizer="adam", loss="mean_squared_error")
    lstm_nn.fit(X, Y, validation_split=0.15, epochs=50, batch_size=256)


if __name__ == "__main__":
    main()
