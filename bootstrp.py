"""Methods to obtain CIs of eval metrics."""

import numpy as np
import pandas as pd
from tqdm import trange
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from keras.metrics import AUC


def metrics_and_N(grouped: pd.DataFrame) -> pd.Series:
    """Calculate the stratum-specific AUROC and num_counts."""
    output: dict[str, np.float64 | int] = dict()
    y = grouped["outcome"].values
    p = grouped["p"].values
    e = 1e-10
    logit = np.log((p + e) / (1 - p - e)).reshape((-1, 1))
    # auroc
    auc = AUC(num_thresholds=5000)
    auc.update_state(y, p)
    auc_i = auc.result().numpy()
    output["auc_i"] = 1 if auc_i == 0 else auc_i
    # calibration slope & intercept cox's method
    calib_lgr = LogisticRegression()
    calib_lgr.fit(logit, y)
    output["slope_i"] = calib_lgr.coef_[0][0]
    output["intercept_i"] = calib_lgr.intercept_[0]
    # num_obs
    output["N_i"] = len(y)
    return pd.Series(output, index=["auc_i", "slope_i", "intercept_i", "N_i"])


class Boot:
    """A single bootstrap sample."""

    def __init__(self, boot_data: pd.DataFrame) -> None:
        """
        Init.

        boot_data: DataFrame defined as {p, count, outcome, boot_patid}
        """
        self._boot_data = boot_data
        self._metric_by_count = boot_data.groupby("count").apply(metrics_and_N)

    @property
    def metric_stratified(self) -> np.ndarray:
        """Calculate stratum-specific metrics. Data are stratified by time index (count)."""
        return self._metric_by_count[["auc_i", "slope_i", "intercept_i"]].values

    @property
    def metrics_w(self) -> np.ndarray:
        """Calculate weighted sum of stratum-specific metric scores."""
        metrics = self._metric_by_count
        metrics["w_i"] = metrics["N_i"].values / metrics["N_i"].values.sum()
        metrics_w = (
            metrics[["auc_i", "slope_i", "intercept_i"]]
            .multiply(metrics["w_i"], axis=0)
            .sum(axis=0)
        )
        return metrics_w.values


class Bootstrapping:
    """Bootstrapping process."""

    def __init__(self, outcome_data: pd.DataFrame) -> None:
        """
        Init.

        outcome_data: DataFrame defined by {p, count, outcome}
        """
        n_patients = outcome_data.loc[outcome_data["count"] == 1].shape[0]
        outcome_data["boot_patid"] = np.nan
        self._patids = np.arange(1, n_patients + 1)
        outcome_data.loc[outcome_data["count"] == 1, "boot_patid"] = self._patids
        outcome_data["boot_patid"] = (
            outcome_data["boot_patid"].fillna(method="ffill").astype("int")
        )
        self._outcome_data = outcome_data

    def bootstrap(self, B: int) -> None:
        """Bootstrap B boot samples."""
        self._B = B
        self._m_ws: np.ndarray = np.array([[np.nan] * 3] * self._B)

        # nested function
        def _collect(i: int):
            """Collect boot statistics."""
            boot = self.generate_boot()
            self._m_ws[i, :] = boot.metrics_w

        # parallel computing
        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(_collect)(i) for i in trange(self._B)
        )

    def generate_original_boot(self) -> Boot:
        """Generate the original outcome data wrapped in a Boot object."""
        return Boot(boot_data=self._outcome_data)

    def generate_boot(self) -> Boot:
        """Generate a single boot."""
        boot_patid = np.random.choice(self._patids, self._patids.size, True)
        boot_data = self._outcome_data.copy().merge(
            pd.Series(boot_patid, name="boot_patid")
        )
        return Boot(boot_data)

    def metrics_w_CI(self, alpha: float = 0.05) -> np.ndarray:
        """Calculate CI of metrics."""
        _alpha = 100 * alpha
        return np.percentile(self._m_ws, q=(_alpha / 2, 100 - _alpha / 2), axis=0)
