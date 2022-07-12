"""Panel Data Example."""

import numpy as np
import pandas as pd
from keras.metrics import AUC


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


def auc_and_N(grouped: pd.DataFrame) -> pd.Series:
    """Summarize the grouped data-frame."""
    output: dict[str, np.float64 | int] = dict()
    y = grouped["outcome"]
    p = grouped["p"]
    auc = AUC()
    auc.update_state(y, p)
    output["auc_i"] = auc.result().numpy()
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
