"""Test Embedding NN Models."""
import numpy as np
import pandas as pd

from data_utils import DataSeqLoader, DataTableLoader
from nn_models import EmbeddingLSTMModel, EmbeddingModel


def data_gen(n: int, p: int) -> pd.DataFrame:
    """Simulate a data set."""
    _pids = range(n)
    time_idx = np.random.randint(1, 15, n)
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
    return df


def main():  # noqa
    # prepare dataloaders
    df1 = data_gen(n=2000, p=20)
    df2 = data_gen(n=400, p=20)
    dl1 = DataSeqLoader(
        X=df1.drop(["id", "y"], axis=1), y=df1["y"], ids=df1["id"], batch_size=8
    )
    dl2 = DataSeqLoader(
        X=df2.drop(["id", "y"], axis=1), y=df2["y"], ids=df2["id"], batch_size=8
    )
    # build model
    em = EmbeddingLSTMModel(dl1, True, 2)
    em.build()
    # fit
    em.fit(val_data=dl2, filepath="./example.hdf5", metric_curve="ROC", epochs=5)


if __name__ == "__main__":
    main()
