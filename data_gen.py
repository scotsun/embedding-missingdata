"""Module that generate simulated data for testing purpose."""

import numpy as np
import pandas as pd


def data_gen(n: int, p: int, max_t: int):
    """
    Generate fake longitudinal data with `n` subjects and `p` predictors and one categorical var.

    Note: outcome variable is included. `max_t` is the maximum num of obs from a subject
    """
    _pids = range(n)
    time_idx = np.random.randint(1, max_t + 1, n)
    pid = []
    for i in range(n):
        pid += [_pids[i]] * time_idx[i]
    X = np.random.randn(time_idx.sum(), p)
    _y = np.random.binomial(1, p=0.1, size=n)
    y = []
    for i in range(n):
        elem = [0] * time_idx[i]
        elem[-1] = _y[i]
        y += elem
    X.ravel()[np.random.choice(X.size, int(X.size * 0.3), replace=False)] = np.nan

    df = pd.DataFrame()
    df["id"] = pid
    df["y"] = y
    df = pd.concat(
        [df, pd.DataFrame(X, columns=["X" + str(i) for i in range(p)])], axis=1
    )

    df_cat = pd.DataFrame()
    df_cat["id"] = _pids
    df_cat["C1"] = np.random.choice(6, n, replace=True)
    df_cat["C2"] = np.random.choice(3, n, replace=True)
    df_cat["C1"] = df_cat["C1"].astype("object")
    df_cat["C2"] = df_cat["C2"].astype("object")

    return df.merge(df_cat, how="left", on="id")
