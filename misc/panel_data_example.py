"""Panel Data Example."""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.metrics import AUC
from keras.utils import to_categorical


def train_generator():
    """Generate simulated training data."""
    while True:
        sequence_length = np.random.randint(1, 14)
        x_train = np.random.random((248, sequence_length, 5))
        # y_train will depend on past 5 timesteps of x
        y_train = x_train[:, :, 0]
        for i in range(1, 5):
            y_train[:, i:] += x_train[:, :-i, i]
        y_train = to_categorical(y_train > 2.5)
        yield x_train, y_train


def auc_and_N(grouped: pd.DataFrame) -> pd.Series:
    """Summarize the grouped data-frame."""
    output: dict[str, np.float64 | int] = dict()
    y = grouped["outcome"]
    p = grouped["p"]
    auc = AUC()
    auc.update_state(y, p)
    auc_i = auc.result().numpy()
    output["auc_i"] = 1 if auc_i == 0 else auc_i
    output["N_i"] = len(y)
    return pd.Series(output, index=["auc_i", "N_i"])


def stratified_auc(outcome_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the stratified auc scores."""
    return outcome_data.groupby("count").apply(auc_and_N)


def auc_w(stratified_auc: pd.DataFrame) -> np.float64:
    """Calculate the weighted auc."""
    weighted_auc = (
        stratified_auc["auc_i"].values
        * stratified_auc["N_i"].values
        / stratified_auc["N_i"].values.sum()
    ).sum()
    return weighted_auc


def main():
    """Build main."""
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(None, 5)))
    model.add(LSTM(8, return_sequences=True))
    model.add(TimeDistributed(Dense(2, activation="softmax")))
    print(model.summary())
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.fit(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)


if __name__ == "__main__":
    main()
